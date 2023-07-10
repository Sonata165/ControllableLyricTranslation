import itertools
import json
import linecache
import math
import os
import random
import sys
import pickle
import socket
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import git
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler

from .sentence_splitter import add_newline_to_end_of_each_sentence
from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer
from transformers.file_utils import cached_property
from transformers.models.bart.modeling_bart import shift_tokens_right
from datasets import load_metric

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils_common.utils import TextCorrupterEn, TextCorrupterCh

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


def get_dataset_by_type(dataset_class):
    print(dataset_class)
    return eval(dataset_class)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


def calculate_sacrebleu(out, ref, zh_tokenize=True):
    ref = [[i] for i in ref]
    metric = load_metric('sacrebleu')
    if zh_tokenize == True:
        result = metric.compute(predictions=out, references=ref, tokenize='zh')
    else:
        result = metric.compute(predictions=out, references=ref)
    ret = {'bleu': round(result['score'], 4)}
    return ret


def calculate_sentence_bleu(out, ref):
    out = [[i] for i in out]
    ref = [[[i]] for i in ref]
    ret = []
    metric = load_metric('sacrebleu')
    for i in range(len(out)):
        t = metric.compute(predictions=out[i], references=ref[i], tokenize='zh', use_effective_order=True)
        ret.append(t['score'])
    return ret


def build_compute_metrics_fn(task_name: str, tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({"gen_len": summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
    return compute_metrics_fn


def trim_batch(
        input_ids,
        pad_token_id,
        attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


# class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
#     def __getitem__(self, index) -> Dict[str, torch.Tensor]:
#         """Call tokenizer on src and tgt_lines"""
#         index = index + 1  # linecache starts at 1
#         source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
#         tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
#         assert source_line, f"empty source line for index {index}"
#         assert tgt_line, f"empty tgt line for index {index}"
#         source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
#         target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)
#
#         source_ids = source_inputs["input_ids"].squeeze()
#         target_ids = target_inputs["input_ids"].squeeze()
#         src_mask = source_inputs["attention_mask"].squeeze()
#         return {
#             "input_ids": source_ids,
#             "attention_mask": src_mask,
#             "labels": target_ids,
#         }
#
#     def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
#         """Only used by LegacyDataset"""
#         return tokenizer(
#             [line],
#             max_length=max_length,
#             padding="max_length" if pad_to_max_length else None,
#             truncation=True,
#             return_tensors=return_tensors,
#             **self.dataset_kwargs,
#         )
#
#     def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
#         input_ids = torch.stack([x["input_ids"] for x in batch])
#         masks = torch.stack([x["attention_mask"] for x in batch])
#         target_ids = torch.stack([x["labels"] for x in batch])
#         pad_token_id = self.pad_token_id
#         y = trim_batch(target_ids, pad_token_id)
#         source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
#         batch = {
#             "input_ids": source_ids,
#             "attention_mask": source_mask,
#             "labels": y,
#         }
#         return batch

class Seq2SeqDatasetEmbStr(AbstractSeq2SeqDataset):
    """
    Dataset class for embedding control of stress pattern
    Meanwhile, provide prompt for output lengtin in as encoder's prefix
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print('constraint path:', t)
        assert t.exists()
        self.tgt_cons_file = t

        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + "_stress.target")
        assert t.exists()
        self.tgt_cons_stress_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_stress_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]

        # Construct constraint for embedding control
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels
        model_inputs["input_ids"] = model_inputs['input_ids']
        model_inputs['attention_mask'] = model_inputs['attention_mask']

        # Process format constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']

        # Process target rhyme constraint
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_rhymes = t2['input_ids']
        attn_rhy = t2['attention_mask']

        # Convert stress pattern from string to tensor
        emb_ids = self.convert_constraint_to_tensor(labels, [x['tgt_stress'][::-1] for x in batch])
        model_inputs['emb_ids'] = self.shift_constraints(emb_ids)

        # Concat length and rhyme constraints with input ids, for length control
        input_ids = torch.cat((tgt_lens, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat rhyme constraints with label (and decoder input)
        labels = torch.cat((tgt_rhymes, labels), dim=1)
        model_inputs['labels'] = labels

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                      [x['tgt_stress'] for x in batch]]

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding

    def convert_constraint_to_tensor(self, decoder_input_ids, constraints):
        '''
        Receive not-even-len constraint, pad it with zero, convert to tensor
        constraints: a list of string, each string represent the constraint for one sentence
        '''
        assert decoder_input_ids.dim() == 2
        assert isinstance(constraints[0], str)

        max_len = 0
        for constraint in constraints:
            max_len = max(max_len, len(constraint))

        # print('max_len:', max_len)

        constraints = [[int(i) for i in list(constraint)] for constraint in constraints]
        # print(constraints)
        # constraints = [[int(l) for l in list(c)] for c in constraints]
        for constraint in constraints:
            cur_len = len(constraint)
            if cur_len < max_len:
                constraint += [0 for i in range(max_len - cur_len)]

            # elif cur_len > max_len:
            #     print(decoder_input_ids)
            #     constraint = constraint[-max_len:]
        # print('ha')
        constraints = torch.tensor(data=constraints, dtype=torch.long, device=decoder_input_ids.device)
        return constraints

    def shift_constraints(self, constraint_tensor, num_shift=1):
        '''
        Shift constraint tensor by num_shift tokens
        A normal decoder input ids start with: [</s>, zh_CN, token1, ...]
        We need shift the constraints by 1
        '''
        assert constraint_tensor.dim() == 2
        t1, t2 = constraint_tensor.shape
        shifted_constraints = torch.zeros(size=[t1, t2 + num_shift], dtype=torch.long)
        # constraint_tensor.new_zeros(constraint_tensor.shape)
        shifted_constraints[:, num_shift:] = constraint_tensor.clone()
        return shifted_constraints


class Seq2SeqDatasetEmbBdr(AbstractSeq2SeqDataset):
    """
    Dataset class for embedding control of stress pattern
    Meanwhile, provide prompt for output lengtin in as encoder's prefix
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print('constraint path:', t)
        assert t.exists()
        self.tgt_cons_file = t

        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + "_boundary.target")
        assert t.exists()
        self.tgt_cons_stress_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_stress_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]

        # Construct constraint for embedding control
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels
        model_inputs["input_ids"] = model_inputs['input_ids']
        model_inputs['attention_mask'] = model_inputs['attention_mask']

        # Process format constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']

        # Process target rhyme constraint
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_rhymes = t2['input_ids']
        attn_rhy = t2['attention_mask']

        # Convert stress pattern from string to tensor
        emb_ids = self.convert_constraint_to_tensor(labels, [x['tgt_stress'][::-1] for x in batch])
        model_inputs['emb_ids'] = self.shift_constraints(emb_ids)

        # Concat length and rhyme constraints with input ids, for length control
        input_ids = torch.cat((tgt_lens, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat rhyme constraints with label (and decoder input)
        labels = torch.cat((tgt_rhymes, labels), dim=1)
        model_inputs['labels'] = labels

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                      [x['tgt_stress'] for x in batch]]

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding

    def convert_constraint_to_tensor(self, decoder_input_ids, constraints):
        '''
        Receive not-even-len constraint, pad it with zero, convert to tensor
        constraints: a list of string, each string represent the constraint for one sentence
        '''
        assert decoder_input_ids.dim() == 2
        assert isinstance(constraints[0], str)

        max_len = 0
        for constraint in constraints:
            max_len = max(max_len, len(constraint))

        # print('max_len:', max_len)

        constraints = [[int(i) for i in list(constraint)] for constraint in constraints]
        # print(constraints)
        # constraints = [[int(l) for l in list(c)] for c in constraints]
        for constraint in constraints:
            cur_len = len(constraint)
            if cur_len < max_len:
                constraint += [0 for i in range(max_len - cur_len)]

            # elif cur_len > max_len:
            #     print(decoder_input_ids)
            #     constraint = constraint[-max_len:]
        # print('ha')
        constraints = torch.tensor(data=constraints, dtype=torch.long, device=decoder_input_ids.device)
        return constraints

    def shift_constraints(self, constraint_tensor, num_shift=1):
        '''
        Shift constraint tensor by num_shift tokens
        A normal decoder input ids start with: [</s>, zh_CN, token1, ...]
        We need shift the constraints by 1
        '''
        assert constraint_tensor.dim() == 2
        t1, t2 = constraint_tensor.shape
        shifted_constraints = torch.zeros(size=[t1, t2 + num_shift], dtype=torch.long)
        # constraint_tensor.new_zeros(constraint_tensor.shape)
        shifted_constraints[:, num_shift:] = constraint_tensor.clone()
        return shifted_constraints


class Seq2SeqDatasetEmbLen(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]

        # Construct constraint for embedding control
        emb_ids = list(range(min(20, length), 0, -1))
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'emb_ids': emb_ids}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels
        model_inputs["input_ids"] = model_inputs['input_ids']
        model_inputs['attention_mask'] = model_inputs['attention_mask']

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        tgt_rhymes = t2['input_ids']
        attn_len = t1['attention_mask']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Convert numeric length constraints to a string
        emb_ids = self.convert_constraint_to_tensor(labels, [x['emb_ids'] for x in batch])
        model_inputs['emb_ids'] = self.shift_constraints(emb_ids)

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding

    def convert_constraint_to_tensor(self, decoder_input_ids, constraints):
        '''
        Receive not-even-len constraint, pad it with zero, convert to tensor
        constraints: a list of string, each string represent the constraint for one sentence
        '''
        assert decoder_input_ids.dim() == 2
        # max_len = decoder_input_ids.shape[1]
        max_len = 0
        for constraint in constraints:
            max_len = max(max_len, len(constraint))
        # constraints = [[int(l) for l in list(c)] for c in constraints]
        for constraint in constraints:
            cur_len = len(constraint)
            if cur_len < max_len:
                constraint += [0 for i in range(max_len - cur_len)]
            # elif cur_len > max_len:
            #     print(decoder_input_ids)
            #     constraint = constraint[-max_len:]
        constraints = torch.tensor(data=constraints, dtype=torch.long, device=decoder_input_ids.device)
        return constraints

    def shift_constraints(self, constraint_tensor, num_shift=1):
        '''
        Shift constraint tensor by num_shift tokens
        A normal decoder input ids start with: [</s>, zh_CN, token1, ...]
        We need shift the constraints by 1
        '''
        assert constraint_tensor.dim() == 2
        t1, t2 = constraint_tensor.shape
        shifted_constraints = torch.zeros(size=[t1, t2 + num_shift], dtype=torch.long)
        # constraint_tensor.new_zeros(constraint_tensor.shape)
        shifted_constraints[:, num_shift:] = constraint_tensor.clone()
        return shifted_constraints


class Seq2SeqDatasetEmbRhy(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for decoder side embedding control
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]

        # Construct constraint for embedding control
        emb_ids = [rhyme]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'emb_ids': emb_ids}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels
        model_inputs["input_ids"] = model_inputs['input_ids']
        model_inputs['attention_mask'] = model_inputs['attention_mask']

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        tgt_rhymes = t2['input_ids']
        attn_len = t1['attention_mask']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Convert numeric length constraints to a string
        emb_ids = torch.tensor([x['emb_ids'] for x in batch])  # [bs, 1]
        model_inputs['emb_ids'] = emb_ids

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetPrefixEncoder(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        tgt_rhymes = t2['input_ids']
        attn_len = t1['attention_mask']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Concat length and rhyme constraints with target ids
        input_ids = torch.cat((tgt_lens, tgt_rhymes, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, attn_rhy, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetWithConstraints(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        tgt_rhymes = t2['input_ids']
        attn_len = t1['attention_mask']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # # Concat length and rhyme constraints with target ids
        # input_ids = torch.cat((tgt_lens, tgt_rhymes, model_inputs['input_ids']), dim=1)
        # attention_mask = torch.cat((attn_len, attn_rhy, model_inputs['attention_mask']), dim=1)
        # model_inputs["input_ids"] = input_ids
        # model_inputs['attention_mask'] = attention_mask

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetConstraintCommon(AbstractSeq2SeqDataset):
    """
    S2s dataset with additional returning elements in __getitem__ function
    including target length, target rhyme, target boundary
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        if not t.exists():
            print('WARNING: target file does not exist: ', t)
        self.tgt_cons_file = t

        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + "_boundary.target")
        if not t.exists():
            print('WARNING: ', t, "doesn't exist")
        self.tgt_cons_stress_file = t
        self.split = type_path

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_stress_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Common part
        model_inputs = self.collate_common(batch)
        self.return_tensors = "pt"

        # Process constraints
        tgt_lens, attn_len = self.collate_tgt_length(batch)
        tgt_rhymes, attn_rhy = self.collate_tgt_rhyme(batch)
        tgt_stress, attn_str = self.collate_tgt_stress(batch)

        # Concat length and stress constraints with encoder input ids
        input_ids = torch.cat((tgt_lens, tgt_stress, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, attn_str, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat rhyme constraints with label and decoder input
        labels = torch.cat((tgt_rhymes, model_inputs["labels"]), dim=1)
        model_inputs['labels'] = labels

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                      [x['tgt_stress'] for x in batch]]

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


    def collate_tgt_length(self, batch):
        tgt_lens = ['len_{}'.format(min(20, x["tgt_len"])) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=self.return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']
        return tgt_lens, attn_len

    def collate_tgt_rhyme(self, batch):
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=self.return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_rhymes = t2['input_ids']
        attn_rhy = t2['attention_mask']
        return tgt_rhymes, attn_rhy

    def collate_tgt_stress(self, batch):
        tgt_stress = [''.join(['str_{}'.format(i) for i in x['tgt_stress'][::-1]]) for x in batch]
        t3 = self.tokenizer(
            tgt_stress,
            return_tensors=self.return_tensors,
            add_special_tokens=False,
            padding=True,
        )
        # add zero padding to 20 (max length) here
        tgt_stress = t3['input_ids']
        attn_str = t3['attention_mask']
        assert tgt_stress.dim() == 2
        pad_bit = 20 - tgt_stress.shape[1]
        tgt_stress = F.pad(tgt_stress, (0, pad_bit, 0, 0), value=1)
        attn_str = F.pad(attn_str, (0, pad_bit, 0, 0), value=1)
        return tgt_stress, attn_str

    def collate_common(self, batch):
        '''
        Common parts in collate function
        '''

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        return model_inputs


class Seq2SeqDatasetLenEncRhyEnc(Seq2SeqDatasetConstraintCommon):
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        # Common part
        model_inputs = self.collate_common(batch)
        self.return_tensors = "pt"

        # Process constraints
        tgt_lens, attn_len = self.collate_tgt_length(batch)
        tgt_rhymes, attn_rhy = self.collate_tgt_rhyme(batch)
        tgt_stress, attn_str = self.collate_tgt_stress(batch)

        # Concat length and rhyme constraints with encoder input ids
        input_ids = torch.cat((tgt_lens, tgt_rhymes, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, attn_rhy, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                      [x['tgt_stress'] for x in batch]]

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetPrefixEncoderStr(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + "_stress.target")
        if not t.exists():
            print(t, "doesn't exist")
            exit(100)
        self.tgt_cons_stress_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_stress_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)

        # Process target rhyme constraint
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_rhymes = t2['input_ids']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Process target stress constraint
        tgt_stress = [''.join(['str_{}'.format(i) for i in x['tgt_stress'][::-1]]) for x in batch]
        t3 = self.tokenizer(
            tgt_stress,
            return_tensors=return_tensors,
            add_special_tokens=False,
            padding=True,
        )
        # add zero padding to 20 (max length) here
        tgt_stress = t3['input_ids']
        attn_str = t3['attention_mask']
        assert tgt_stress.dim() == 2
        pad_bit = 20 - tgt_stress.shape[1]
        tgt_stress = F.pad(tgt_stress, (0, pad_bit, 0, 0), value=1)
        attn_str = F.pad(attn_str, (0, pad_bit, 0, 0), value=1)

        # Concat length and stress constraints with encoder input ids
        input_ids = torch.cat((tgt_lens, tgt_stress, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, attn_str, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat rhyme constraints with label and decoder input
        labels = torch.cat((tgt_rhymes, labels), dim=1)
        model_inputs['labels'] = labels

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                      [x['tgt_stress'] for x in batch]]

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetPrefixEncoderBdr(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + "_boundary.target")
        if not t.exists():
            print(t, "doesn't exist")
            exit(100)
        self.tgt_cons_stress_file = t
        self.split = type_path

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_stress_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)

        # Process target rhyme constraint
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_rhymes = t2['input_ids']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Process target stress constraint
        tgt_stress = [''.join(['str_{}'.format(i) for i in x['tgt_stress'][::-1]]) for x in batch]
        t3 = self.tokenizer(
            tgt_stress,
            return_tensors=return_tensors,
            add_special_tokens=False,
            padding=True,
        )
        # add zero padding to 20 (max length) here
        tgt_stress = t3['input_ids']
        attn_str = t3['attention_mask']
        assert tgt_stress.dim() == 2
        pad_bit = 20 - tgt_stress.shape[1]
        tgt_stress = F.pad(tgt_stress, (0, pad_bit, 0, 0), value=1)
        attn_str = F.pad(attn_str, (0, pad_bit, 0, 0), value=1)

        # Concat length and stress constraints with encoder input ids
        input_ids = torch.cat((tgt_lens, tgt_stress, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, attn_str, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat rhyme constraints with label and decoder input
        labels = torch.cat((tgt_rhymes, labels), dim=1)
        model_inputs['labels'] = labels

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        try:
            model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                          [x['tgt_stress'] for x in batch]]
        except:
            import traceback
            traceback.print_exc()
            print([x['tgt_stress'] for x in batch])
            exit(100)

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding

class Seq2SeqDatasetPrefixEncoderBdrNoRhy(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Control boundary by prompt at prefix of encoder's input
    Cancel the rhyme control
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + "_boundary.target")
        if not t.exists():
            print(t, "doesn't exist")
            exit(100)
        self.tgt_cons_stress_file = t
        self.split = type_path

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_stress_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        rhyme = 0
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)

        # Process target rhyme constraint
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_rhymes = t2['input_ids']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Process target stress constraint
        tgt_stress = [''.join(['str_{}'.format(i) for i in x['tgt_stress'][::-1]]) for x in batch]
        t3 = self.tokenizer(
            tgt_stress,
            return_tensors=return_tensors,
            add_special_tokens=False,
            padding=True,
        )
        # add zero padding to 20 (max length) here
        tgt_stress = t3['input_ids']
        attn_str = t3['attention_mask']
        assert tgt_stress.dim() == 2
        pad_bit = 20 - tgt_stress.shape[1]
        tgt_stress = F.pad(tgt_stress, (0, pad_bit, 0, 0), value=1)
        attn_str = F.pad(attn_str, (0, pad_bit, 0, 0), value=1)

        # Concat length and stress constraints with encoder input ids
        input_ids = torch.cat((tgt_lens, tgt_stress, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, attn_str, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat rhyme constraints with label and decoder input
        labels = torch.cat((tgt_rhymes, labels), dim=1)
        model_inputs['labels'] = labels

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        try:
            model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                          [x['tgt_stress'] for x in batch]]
        except:
            import traceback
            traceback.print_exc()
            print([x['tgt_stress'] for x in batch])
            exit(100)

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding

class Seq2SeqDatasetPrefixEncoderBdrNoRhyBiased(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Control boundary by prompt at prefix of encoder's input
    Cancel the rhyme control and boundary control
    For biased decoding of boundary control
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + "_boundary.target")
        if not t.exists():
            print(t, "doesn't exist")
            exit(100)
        self.tgt_cons_stress_file = t
        self.split = type_path

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_stress_file), index).rstrip('\n')
        bdr_pos = []
        for i in range(len(stress_line)):
            if stress_line[i] == '1':
                bdr_pos.append(i+1)

        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        rhyme = 0
        stress_line = ''.join(['0' for i in range(length)])
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line, 'bdr_pos': bdr_pos}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)

        # Process target rhyme constraint
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_rhymes = t2['input_ids']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Process target stress constraint
        tgt_stress = [''.join(['str_{}'.format(i) for i in x['tgt_stress'][::-1]]) for x in batch]
        t3 = self.tokenizer(
            tgt_stress,
            return_tensors=return_tensors,
            add_special_tokens=False,
            padding=True,
        )
        # add zero padding to 20 (max length) here
        tgt_stress = t3['input_ids']
        attn_str = t3['attention_mask']
        assert tgt_stress.dim() == 2
        pad_bit = 20 - tgt_stress.shape[1]
        tgt_stress = F.pad(tgt_stress, (0, pad_bit, 0, 0), value=1)
        attn_str = F.pad(attn_str, (0, pad_bit, 0, 0), value=1)

        # Concat length and stress constraints with encoder input ids, no stress constraints
        input_ids = torch.cat((tgt_lens, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat rhyme constraints with label and decoder input
        labels = torch.cat((tgt_rhymes, labels), dim=1)
        model_inputs['labels'] = labels

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        try:
            model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                          [x['tgt_stress'] for x in batch]]
        except:
            import traceback
            traceback.print_exc()
            print([x['tgt_stress'] for x in batch])
            exit(100)

        model_inputs['bdr_pos'] = [x['bdr_pos'] for x in batch]

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding



class Seq2SeqDatasetPrefixEncoderBdrRev(Seq2SeqDatasetPrefixEncoderBdr):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_stress_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line[::-1]}


class Seq2SeqDatasetPrefixEncoderBdrDenoise(Seq2SeqDatasetPrefixEncoderBdr):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        if self.split == 'train':
            p = random.uniform(0, 1)  # corrupt 50% source sentence for training
            if p > 0.5:
                source_line = TextCorrupterEn.corrupt_sentence(source_line)  # corrupt source sentence

        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_stress_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line}


class Seq2SeqDatasetPrefixEncoderLength(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    TODO: add code Read rhyme constraints to batch, but doesn't add to input as prefix
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        # t2 = self.tokenizer(
        #     tgt_rhymes,
        #     add_special_tokens=False,
        #     return_tensors=return_tensors,
        #     max_length=1,
        #     padding=False,
        #     truncation=True,
        # )
        tgt_lens = t1['input_ids']
        # tgt_rhymes = t2['input_ids']
        attn_len = t1['attention_mask']
        # attn_rhy = t2['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Concat length and rhyme constraints with target ids
        input_ids = torch.cat((tgt_lens, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetPrefixEncoderRhyme(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        tgt_rhymes = t2['input_ids']
        attn_len = t1['attention_mask']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Concat length and rhyme constraints with target ids
        input_ids = torch.cat((tgt_rhymes, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_rhy, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetLenEncRhyEmb(Seq2SeqDatasetConstraintCommon):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for decoder side embedding control
    """

    def __getitem__(self, index) -> Dict[str, str]:
        ret = super().__getitem__(index)

        # Construct constraint for embedding control
        emb_ids = [ret['tgt_rhyme']]
        ret['emb_ids'] = emb_ids
        return ret

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        # Common part
        model_inputs = self.collate_common(batch)
        self.return_tensors = "pt"

        # Process constraints
        tgt_lens, attn_len = self.collate_tgt_length(batch)
        tgt_rhymes, attn_rhy = self.collate_tgt_rhyme(batch)
        tgt_stress, attn_str = self.collate_tgt_stress(batch)

        # Concat length and rhyme constraints with encoder input ids
        input_ids = torch.cat((tgt_lens, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Convert numeric length constraints to a string
        emb_ids = torch.tensor([x['emb_ids'] for x in batch])  # [bs, 1]
        model_inputs['emb_ids'] = emb_ids

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                      [x['tgt_stress'] for x in batch]]

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetPrefixDecoder(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of target text
    Dataset class for Prefix decoders
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        tgt_lens = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )['input_ids']
        model_inputs['tgt_lens'] = tgt_lens

        tgt_rhymes = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )['input_ids']
        model_inputs['tgt_rhymes'] = tgt_rhymes

        # Concat length and rhyme constraints with target ids
        labels = torch.cat((tgt_lens, tgt_rhymes, labels), dim=1)
        model_inputs["labels"] = labels

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


class Seq2SeqDatasetPrefixLengthRhyme(AbstractSeq2SeqDataset):
    """
    Encoder-side prompt for length control,
    Decoder-side prompt for rhyme control
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']

        tgt_rhymes = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )['input_ids']
        # model_inputs['tgt_rhymes'] = tgt_rhymes

        # Concat length constraints with input ids
        input_ids = torch.cat((tgt_lens, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat length and rhyme constraints with target ids
        labels = torch.cat((tgt_rhymes, labels), dim=1)
        model_inputs["labels"] = labels
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding

    def prepare_decoder_input_ids_for_generation(self, tgt_lens_tensor):
        '''
        tgt_lens_tensor: tokenized length token, in batch   shape: [bs]
        '''
        assert tgt_lens_tensor.dim() == 1

        # Prepare decoder input ids
        batch_tgt_len = tgt_lens_tensor
        print(batch_tgt_len.shape)
        decoder_input_ids = torch.cat(
            (batch_tgt_len.unsqueeze(-1),
             torch.ones((batch_tgt_len.shape[0], 1), dtype=torch.int64, device=tgt_lens_tensor.device) * 2), dim=1
        )
        return decoder_input_ids


class Seq2SeqDatasetPrefixDecoderStr(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(str(self.tgt_cons_file).replace('.target', '_stress.target'), index).rstrip(
            '\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)

        # Process target rhyme constraint
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_rhymes = t2['input_ids']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Process target stress constraint
        tgt_stress = [''.join(['str_{}'.format(i) for i in x['tgt_stress'][::-1]]) for x in batch]
        t3 = self.tokenizer(
            tgt_stress,
            return_tensors=return_tensors,
            add_special_tokens=False,
            padding=True,
        )
        # add zero padding to 20 (max length) here
        tgt_stress = t3['input_ids']
        attn_str = t3['attention_mask']
        assert tgt_stress.dim() == 2
        pad_bit = 20 - tgt_stress.shape[1]
        tgt_stress = F.pad(tgt_stress, (0, pad_bit, 0, 0), value=1)

        # Concat length constraints with encoder input ids
        input_ids = torch.cat((tgt_lens, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat rhyme and stress constraints with label (and decoder input)
        labels = torch.cat((tgt_rhymes, tgt_stress, labels), dim=1)
        model_inputs['labels'] = labels

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                      [x['tgt_stress'] for x in batch]]

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetPrefixDecoderBdr(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

        self.tgt_bdr_file = str(self.tgt_cons_file).replace('.target', '_boundary.target')
        assert os.path.exists(self.tgt_bdr_file)

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        stress_line = linecache.getline(self.tgt_bdr_file, index).rstrip(
            '\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1,
                'tgt_len': length, 'tgt_rhyme': rhyme, 'tgt_stress': stress_line}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_lens = t1['input_ids']
        attn_len = t1['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)

        # Process target rhyme constraint
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t2 = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        tgt_rhymes = t2['input_ids']
        attn_rhy = t2['attention_mask']
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Process target stress constraint
        tgt_stress = [''.join(['str_{}'.format(i) for i in x['tgt_stress'][::-1]]) for x in batch]
        t3 = self.tokenizer(
            tgt_stress,
            return_tensors=return_tensors,
            add_special_tokens=False,
            padding=True,
        )
        # add zero padding to 20 (max length) here
        tgt_stress = t3['input_ids']
        attn_str = t3['attention_mask']
        assert tgt_stress.dim() == 2
        pad_bit = 20 - tgt_stress.shape[1]
        tgt_stress = F.pad(tgt_stress, (0, pad_bit, 0, 0), value=1)

        # Concat length constraints with encoder input ids
        input_ids = torch.cat((tgt_lens, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Concat rhyme and stress constraints with label (and decoder input)
        labels = torch.cat((tgt_rhymes, tgt_stress, labels), dim=1)
        model_inputs['labels'] = labels

        # Add constraints to batch data
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)
        model_inputs['tgt_stress'] = [[int(i) for i in list(constraint)] for constraint in
                                      [x['tgt_stress'] for x in batch]]

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding


class Seq2SeqDatasetPrefixDecoderLength(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of target text
    Dataset class for Prefix decoders
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        # tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        tgt_lens = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )['input_ids']
        model_inputs['tgt_lens'] = tgt_lens

        # tgt_rhymes = self.tokenizer(
        #     tgt_rhymes,
        #     add_special_tokens=False,
        #     return_tensors=return_tensors,
        #     max_length=1,
        #     padding=False,
        #     truncation=True,
        # )['input_ids']
        # model_inputs['tgt_rhymes'] = tgt_rhymes

        # Concat length and rhyme constraints with target ids
        labels = torch.cat((tgt_lens, labels), dim=1)
        model_inputs["labels"] = labels
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding

    def prepare_decoder_input_ids_for_generation(self, tgt_lens_tensor):
        '''
        tgt_lens_tensor: tokenized length token, in batch   shape: [bs]
        '''
        assert tgt_lens_tensor.dim() == 1

        # Prepare decoder input ids
        batch_tgt_len = tgt_lens_tensor
        print(batch_tgt_len.shape)
        decoder_input_ids = torch.cat(
            (batch_tgt_len.unsqueeze(-1),
             torch.ones((batch_tgt_len.shape[0], 1), dtype=torch.int64, device=tgt_lens_tensor.device) * 2), dim=1
        )
        return decoder_input_ids


class Seq2SeqDatasetPrefixDecoderRhyme(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of target text
    Dataset class for Prefix decoders
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        # tgt_lens = self.tokenizer(
        #     tgt_lens,
        #     add_special_tokens=False,
        #     return_tensors=return_tensors,
        #     max_length=1,
        #     padding=False,
        #     truncation=True,
        # )['input_ids']

        tgt_rhymes = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )['input_ids']

        # Concat length and rhyme constraints with target ids
        labels = torch.cat((tgt_rhymes, labels), dim=1)
        model_inputs["labels"] = labels
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


class Seq2SeqDatasetPrefixDecoderDoc(AbstractSeq2SeqDataset):
    """
    A dataset that calls prepare_seq2seq_batch.
    Additional length and rhyme constraints will be added to the labels as prefix
    sentences
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        assert tgt_texts != None
        # if tgt_texts is None:
        #     return model_inputs

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        # model_inputs["labels"] = labels["input_ids"]

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        tgt_lens = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )['input_ids']
        model_inputs['tgt_lens'] = tgt_lens
        tgt_rhymes = self.tokenizer(
            tgt_rhymes,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )['input_ids']
        model_inputs['tgt_rhymes'] = tgt_rhymes

        # Concat length and rhyme constraints with target ids
        labels = torch.cat((tgt_lens, tgt_rhymes, labels), dim=1)
        model_inputs["labels"] = labels

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        # assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        # assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


class Seq2SeqDatasetAdapt(AbstractSeq2SeqDataset):
    """
    Dataset for monolingual adaptation
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        constraint_path = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(
            type_path + ".target")
        if os.path.exists(constraint_path):
            print("WARNING: constraint path doesn't exist!")
        self.tgt_cons_file = constraint_path
        self.split = type_path

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        source_text = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")  #
        if self.split == 'train':
            source_line = TextCorrupterCh.corrupt_sentence(tgt_line)
        else:
            source_line = source_text

        # assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        if self.split != 'train':
            self.dataset_kwargs['src_lang'] = 'en_XX'

        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        if self.split != 'train':
            self.dataset_kwargs['src_lang'] = 'zh_CN'

        return batch_encoding


def add_constraint_prefix(input_ids: torch.Tensor,
                          pad_token_id: int,
                          decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i: i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i: i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank: self.total_size: self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w", encoding='utf8') as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
        data = json.loads(data)
    return data


def save_json_sort(data, path):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False))


def print_json(data):
    print(json.dumps(data, indent=4, ensure_ascii=False))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    try:
        repo = git.Repo(search_parent_directories=True)
        repo_infos = {
            "repo_id": str(repo),
            "repo_sha": str(repo.head.object.hexsha),
            "repo_branch": str(repo.active_branch),
            "hostname": str(socket.gethostname()),
        }
        return repo_infos
    except TypeError:
        return {
            "repo_id": None,
            "repo_sha": None,
            "repo_branch": None,
            "hostname": None,
        }


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def calculate_rouge(
        pred_lns: List[str],
        tgt_lns: List[str],
        use_stemmer=True,
        rouge_keys=ROUGE_KEYS,
        return_precision_and_recall=False,
        bootstrap_aggregation=True,
        newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def unfreeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def unfreeze_embeds(model):
    model_type = model.config.model_type

    if model_type == "t5":
        unfreeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            unfreeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            unfreeze_params(d.embed_positions)
            unfreeze_params(d.embed_tokens)
    else:
        unfreeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            unfreeze_params(d.embed_positions)
            unfreeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad / npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it

    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir

    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
            os.path.exists(args.output_dir)
            and len(os.listdir(args.output_dir)) > expected_items
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )
