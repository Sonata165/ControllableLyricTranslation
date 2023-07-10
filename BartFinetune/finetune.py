'''
Finetune mBART translation model
Author: Longshen Ou
2022/6/23

Merge of training script for different models
'''
# !/usr/bin/env python

import argparse
import glob
import logging
import os
import sys
import time
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import MBartTokenizer, T5ForConditionalGeneration, MBart50TokenizerFast, LogitsProcessor
from transformers.models.bart.modeling_bart import shift_tokens_right

from metrics import BoundaryRecall
from utils.callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from utils.utils import (
    ROUGE_KEYS,
    Seq2SeqDataset,
    Seq2SeqDatasetAdapt,
    Seq2SeqDatasetWithConstraints,
    Seq2SeqDatasetPrefixEncoder,
    Seq2SeqDatasetPrefixDecoder,
    Seq2SeqDatasetPrefixEncoderLength,
    Seq2SeqDatasetPrefixEncoderRhyme,
    Seq2SeqDatasetPrefixDecoderLength,
    Seq2SeqDatasetPrefixDecoderRhyme,
    Seq2SeqDatasetEmbLen,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    check_output_dir,
    flatten_list,
    freeze_embeds,
    freeze_params,
    unfreeze_params,
    unfreeze_embeds,
    get_git_info,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    save_json_sort,
    print_json,
    use_task_specific_params,
    calculate_sacrebleu,
    read_json,
    get_dataset_by_type,
)

# Add the parent dir to path
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from utils.lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa
from models.MBarts import (
    shift_tokens_right_prefix_1,
    shift_tokens_right_prefix_2,
    shift_tokens_right_prefix_20,
    shift_tokens_right_prefix_21,
    ForcedBOSTokenLogitsProcessorPrefixDecoder,
    ForcedBOSTokenLogitsProcessorPrefixDecoderLength,
    ForcedBOSTokenLogitsProcessorPrefixDecoderN,
)

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils_common.utils import calculate_acc, RhymeUtil, RhymeCaculator, PosSeg, calculate_acc_2d

# from LM.ngram.language_model import LanguageModel

logger = logging.getLogger(__name__)

from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(3)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


class TranslationModule(BaseTransformer):
    mode = "translation"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, args, tokenizer=None):
        if args.sortish_sampler and args.gpus > 1:
            args.replace_sampler_ddp = False
        elif args.max_tokens_per_batch is not None:
            if args.gpus > 1:
                raise NotImplementedError(
                    "Dynamic Batch size does not remove_multi_character_word for multi-gpu training")
            if args.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")
        super().__init__(args, num_labels=None, mode=self.mode, tokenizer=tokenizer,
                         model_class_name=args.model_class_name)

        # Extend model's embedding size
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))

        use_task_specific_params(self.model, "summarization")
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "args.json"
        save_json_sort(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

        print('data_dir:', self.hparams.data_dir)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        self.dataset_kwargs["src_lang"] = args.src_lang
        self.dataset_kwargs["tgt_lang"] = args.tgt_lang
        print(args.src_lang, args.tgt_lang)
        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        if self.hparams.prompt_tuning:
            freeze_params(self.model.get_encoder())
            freeze_params(self.model.get_decoder())
            unfreeze_embeds(self.model)

        # self.args.git_sha = None
        self.num_workers = args.num_workers
        # self.decoder_start_token_id = None  # default to config
        # if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
        #     self.decoder_start_token_id = self.tokenizer.lang_code_to_id[args.tgt_lang]
        #     self.model.config.decoder_start_token_id = self.decoder_start_token_id
        #     print('BOS start token id specified to ', self.tokenizer.lang_code_to_id[args.tgt_lang])
        self.forced_bos_token_id = self.tokenizer.lang_code_to_id[args.tgt_lang]
        print('\n', 'BOS start token id specified to ', self.tokenizer.lang_code_to_id[args.tgt_lang], '\n')

        # self.dataset_class = eval(self.hparams.dataset_class)
        self.dataset_class = get_dataset_by_type(self.hparams.dataset_class)

        self.already_saved_batch = False
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        print('val metric:', self.val_metric)
        print('Max length when generating: {}\n'.format(self.eval_max_length))

        # Metric for checkpointing
        # self.best_bleu = 0 # Update 9.18: use validation loss
        self.best_loss = 100

        # Metric for logging
        self.avg_train_loss = 0
        self.best_valid_metrics = None

    def calc_generative_metrics(self, preds, target, zh_tokenize) -> dict:
        '''
        Calculate metrics for generation
        '''
        # return calculate_bleu(preds, target)
        return calculate_sacrebleu(preds, target, zh_tokenize=zh_tokenize)

    def save_readable_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        '''
        Save a batch of data for debugging
        '''
        # readable_batch = {
        #     k: self.tokenizer.batch_decode(v.tolist()) if ("ids" in k or 'labels' in k) and 'emb' not in k else v.tolist() for k, v in batch.items()
        # }
        readable_batch = {}
        for k, v in batch.items():
            if ('ids' in k or 'labels' in k) and 'emb' not in k:
                readable_batch[k] = self.tokenizer.batch_decode(v.tolist())
            elif 'emb_ids' in k:
                readable_batch[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                readable_batch[k] = v.shape
            else:
                readable_batch[k] = v

        save_json(readable_batch, Path(self.output_dir) / "text_batch.json")
        # save_json({k: v.tolist() for k, v in batch.items()}, Path(self.output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, input_ids, **kwargs):
        '''
        Compute model output of a forward step
        '''
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        '''
        A step of forward pass in TRAINING, computing the logits in every position and return the loss
        A generative step will also call this function, but only for computing loss
        '''
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        # tgt_lens, tgt_rhymes = batch['tgt_lens'], batch['tgt_rhymes']

        # Prepare decoder input for training (right-shifted target)
        decoder_start_token_id = self.model._get_decoder_start_token_id()
        if self.hparams.dataset_class == "Seq2SeqDatasetPrefixDecoder":  # 2 decoder prefix
            shift_func = shift_tokens_right_prefix_2
        elif self.hparams.dataset_class in [  # 1 decoder prefix
            "Seq2SeqDatasetPrefixDecoderLength",
            "Seq2SeqDatasetPrefixDecoderRhyme",
            'Seq2SeqDatasetPrefixLengthRhyme',
            'Seq2SeqDatasetEmbStr',
            'Seq2SeqDatasetEmbBdr',
            'Seq2SeqDatasetPrefixEncoderStr',
            'Seq2SeqDatasetPrefixEncoderBdr',
            'Seq2SeqDatasetPrefixEncoderBdrRev',
            'Seq2SeqDatasetPrefixEncoderBdrDenoise',
        ]:
            shift_func = shift_tokens_right_prefix_1
        elif self.hparams.dataset_class in [  # no decoder prefix
            "Seq2SeqDataset",
            'Seq2SeqDatasetAdapt',
            'Seq2SeqDatasetPrefixEncoder',
            'Seq2SeqDatasetPrefixEncoderLength',
            'Seq2SeqDatasetPrefixEncoderRhyme',
            'Seq2SeqDatasetEmbLen',
            'Seq2SeqDatasetEmbRhy',
            'Seq2SeqDatasetLenEncRhyEmb',
            'Seq2SeqDatasetLenEncRhyEnc',
        ]:
            shift_func = shift_tokens_right
        elif self.hparams.dataset_class in [  # 21 decoder prefix
            'Seq2SeqDatasetPrefixDecoderStr',
            'Seq2SeqDatasetPrefixDecoderBdr',
        ]:
            shift_func = shift_tokens_right_prefix_21
        else:
            raise Exception("Incorrect dataset_class: {}".format(self.hparams.dataset_class))

        # Prepare decoder input
        decoder_input_ids = shift_func(tgt_ids, pad_token_id, decoder_start_token_id)
        if not self.already_saved_batch:  # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch(batch)

        # Forward pass
        if self.hparams.dataset_class in [  # for embedding constraints, do forward pass for encoder by hand
            'Seq2SeqDatasetEmbLen',
            'Seq2SeqDatasetEmbRhy',
            'Seq2SeqDatasetEmbStr',
            'Seq2SeqDatasetEmbBdr',
            'Seq2SeqDatasetLenEncRhyEmb',
        ]:
            emb_ids = batch['emb_ids']
            decoder_input_embeds = self.model.prepare_decoder_inputs_embeds_for_training(decoder_input_ids, emb_ids)
            outputs = self(src_ids, attention_mask=src_mask, decoder_inputs_embeds=decoder_input_embeds,
                           use_cache=False)
        else:  # If no constraints is in embedding form, do forward pass for the whole model directly
            outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)

        lm_logits = outputs["logits"]

        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # , weight=loss_weight)
            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            # Original implementation
            # lprobs = nn.functional.log_softmax(lm_logits, dim=-1)
            # loss, nll_loss = label_smoothed_nll_loss(
            #     lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            # )

            # Longshen's implementation
            print('new label smoothing')
            ce_loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=self.hparams.label_smoothing)
            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        '''
        Full forward pass and metric computing in training
        '''
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()

        # Log in wandb, but I don't know where are the logs
        for k in logs:
            self.log(k, logs[k])

        # Update average train loss
        self.avg_train_loss = (self.avg_train_loss * batch_idx + loss_tensors[0].item()) / (batch_idx + 1)

        return {"loss": loss_tensors[0], "log": logs}

    def training_epoch_end(self, outputs):
        # Clear metric cache
        self.avg_train_loss = 0

    # def backward(
    #         self, loss, optimizer, optimizer_idx, *args, **kwargs
    # ) -> None:
    #     '''
    #     Freeze the embedding for original vocab,
    #     But not freeze for additional vocab (special tokens)
    #     '''
    #     loss.backward(*args, **kwargs)
    #
    #     # Zero out gradient for tokens in the original vocabularies
    #     original_vocab_size = 250054
    #
    #     if self.model.model.shared.weight.grad != None:
    #         self.model.model.shared.weight.grad[:original_vocab_size] = 0

    def validation_step(self, batch, batch_idx) -> Dict:
        '''
        Full forward pass and metric computing in validating
        '''
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        ks = self.metric_names + ["gen_time", "gen_len"]

        if self.hparams.dataset_class not in ['Seq2SeqDataset', 'Seq2SeqDatasetAdapt']:
            ks += ['len_acc', 'rhyme_acc', 'boundary_recall']

        if self.hparams.dataset_class in [
            'Seq2SeqDatasetEmbStr',
            'Seq2SeqDatasetPrefixDecoderStr',
            'Seq2SeqDatasetPrefixEncoderStr',
        ]:
            ks += ['stress_acc']

        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in ks
        }

        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count

        all_metrics['epoch'] = self.current_epoch
        all_metrics['step'] = self.global_step

        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])

        # Log in wandb
        for k in all_metrics:
            self.log(k, all_metrics[k])

        # Log myself
        self.valid_metrics = all_metrics

        print('Validation end\n')
        # Save checkpoint
        if prefix == 'val':
            if self.global_step > 0:  # Don't log for sanity check
                # Save checkpoint
                # if all_metrics['val_avg_bleu'] > self.best_bleu and self.global_step > 0:
                if all_metrics['val_avg_loss'] < self.best_loss and self.global_step > 0:
                    save_path = self.output_dir.joinpath("best_tfmr")
                    self.best_loss = all_metrics['val_avg_loss']
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)

                    # checkpoint_name = 'epoch: {}, step: {}, loss: {:.4f}, bleu: {:.4f}'.format(
                    #     self.current_epoch, self.global_step, all_metrics['val_avg_loss'], all_metrics['val_avg_bleu'])
                    # history_path = os.path.join(save_path, 'history')
                    # if not os.path.exists(history_path):
                    #     os.mkdir(history_path)
                    # with open(os.path.join(save_path, 'history', checkpoint_name), 'w') as f:
                    #     f.write('Good luck Larry!')
                    # print('Saving checkpoint: {}\n'.format(checkpoint_name))

                    self.best_valid_metrics = all_metrics.copy()

                # Logging
                if self.best_valid_metrics == None:
                    self.best_valid_metrics = all_metrics.copy()
                self.my_log(all_metrics, self.output_dir.joinpath('log.txt'))
                self.my_log(self.best_valid_metrics, self.output_dir.joinpath('log_best.txt'))

        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def my_log(self, valid_metrics, log_fn):
        '''
        Log training loss and validation metric to file at the end of each validation.
        '''
        log_str = 'Epoch: {}, step: {} | Train loss: {:.4f} | Val loss: {:.4f} | Bleu: {:.4f}'.format(
            self.current_epoch, self.global_step, self.avg_train_loss,
            valid_metrics['val_avg_loss'], valid_metrics['val_avg_bleu']
        )
        if 'val_avg_len_acc' in valid_metrics:
            log_str += ' | Len: {:.4f}'.format(valid_metrics['val_avg_len_acc'])
        if 'val_avg_rhyme_acc' in valid_metrics:
            log_str += ' | Rhy: {:.4f}'.format(valid_metrics['val_avg_rhyme_acc'])
        # if 'val_avg_stress_acc' in valid_metrics:
        #     log_str += ' | Str: {:.4f}'.format(valid_metrics['val_avg_stress_acc'])
        if 'val_avg_boundary_recall' in valid_metrics:
            log_str += ' | Bdr: {:.4f}'.format(valid_metrics['val_avg_boundary_recall'])
        log_str += '\n'
        with open(log_fn, 'a') as f:
            f.write(log_str)

    def _generative_step(self, batch: dict) -> dict:
        '''
        A forward step in validating and testing
        '''
        t0 = time.time()
        if self.hparams.dataset_class == 'Seq2SeqDatasetPrefixDecoder':  # if 2 prefix for decoder
            # Prepare decoder input ids
            decoder_input_ids = batch['labels'][:, :3].clone()  # [BS, 3]
            decoder_input_ids[:, 2] = 2  # set the 3rd col to decoder_start_token_id

            # Prepare logits processor for forced bos token id
            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoder(bos_token_id=self.forced_bos_token_id)

            generated_ids = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                # decoder_start_token_id=self.decoder_start_token_id, #  no need to set for mBART-50
                # forced_bos_token_id=self.forced_bos_token_id,  # should be set for mBART-50
                decoder_input_ids=decoder_input_ids,
                logits_processor=[bos_processor],
                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
            )  # [bs, max_gen_length]
        elif self.hparams.dataset_class in [  # if 1 prefix for decoder
            'Seq2SeqDatasetPrefixDecoderLength',
            'Seq2SeqDatasetPrefixDecoderRhyme',
            'Seq2SeqDatasetPrefixLengthRhyme',
            'Seq2SeqDatasetPrefixEncoderStr',
            'Seq2SeqDatasetPrefixEncoderBdr',
            'Seq2SeqDatasetPrefixEncoderBdrRev',
            'Seq2SeqDatasetPrefixEncoderBdrDenoise',
        ]:
            # Prepare decoder input ids
            decoder_input_ids = batch['labels'][:, :2].clone()  # [BS, 3]
            decoder_input_ids[:, 1] = 2  # set the 2nd col to decoder_start_token_id

            # Prepare logits processor for forced bos token id
            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderLength(bos_token_id=self.forced_bos_token_id)

            generated_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,

                decoder_input_ids=decoder_input_ids,
                logits_processor=[bos_processor],

                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
            )  # [bs, max_gen_length]
        elif self.hparams.dataset_class in [  # if 21 prefix for decoder
            'Seq2SeqDatasetPrefixDecoderStr',
            'Seq2SeqDatasetPrefixDecoderBdr',
        ]:
            # Prepare decoder input ids
            n = 21
            decoder_input_ids = batch['labels'][:, :n + 1].clone()  # [BS, 3]
            decoder_input_ids[:, n] = 2  # set the 2nd col to decoder_start_token_id

            # Prepare logits processor for forced bos token id
            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderN(
                bos_token_id=self.forced_bos_token_id,
                prefix_length=n,
            )

            generated_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,

                decoder_input_ids=decoder_input_ids,
                logits_processor=[bos_processor],

                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
            )  # [bs, max_gen_length]
        elif self.hparams.dataset_class in [  # if no prefix for decoder
            "Seq2SeqDataset",
            'Seq2SeqDatasetAdapt',
            'Seq2SeqDatasetPrefixEncoder',
            'Seq2SeqDatasetPrefixEncoderLength',
            'Seq2SeqDatasetPrefixEncoderRhyme',
            'Seq2SeqDatasetLenEncRhyEnc',
        ]:
            generated_ids = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                # decoder_start_token_id=self.decoder_start_token_id, #  no need to set for mBART-50
                forced_bos_token_id=self.forced_bos_token_id,  # should be set for mBART-50
                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
            )
        elif self.hparams.dataset_class in [  # embedding constraints with no decoder prefix
            'Seq2SeqDatasetEmbLen',
            'Seq2SeqDatasetEmbRhy',
            'Seq2SeqDatasetLenEncRhyEmb',
        ]:
            generated_ids = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                # decoder_start_token_id=self.decoder_start_token_id, #  no need to set for mBART-50
                forced_bos_token_id=self.forced_bos_token_id,  # should be set for mBART-50
                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
                emb_ids=batch["emb_ids"],
            )
        elif self.hparams.dataset_class in [  # embedding constraints with 1 decoder prefix
            'Seq2SeqDatasetEmbStr',
            'Seq2SeqDatasetEmbBdr',
        ]:
            # Prepare decoder input ids
            n = 1  # num of prefix
            decoder_input_ids = batch['labels'][:, :n + 1].clone()  # [BS, 3]
            decoder_input_ids[:, n] = 2  # set the 2nd col to decoder_start_token_id

            # Prepare logits processor for forced bos token id
            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderLength(bos_token_id=self.forced_bos_token_id)

            generated_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,

                decoder_input_ids=decoder_input_ids,
                logits_processor=[bos_processor],

                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
                emb_ids=batch["emb_ids"],
            )
        else:
            raise Exception("Incorrect dataset_class: {}".format(self.hparams.dataset_class))

        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)

        # Validation metrics
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}  # {loss: loss_value}
        # print(len(preds), len(target))
        zh_tokenize = True if self.tgt_lang == 'zh_CN' else False
        bleu: Dict = self.calc_generative_metrics(preds, target, zh_tokenize=zh_tokenize)
        summ_len = np.mean(lmap(len, generated_ids))

        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **bleu)

        if self.hparams.dataset_class not in ['Seq2SeqDataset', 'Seq2SeqDatasetAdapt']:
            # Compute format accuracy
            out_lens = [len(i.strip()) for i in preds]
            len_acc = calculate_acc(out=out_lens, tgt=batch['tgt_lens'].squeeze().tolist())

            # Compute rhyme accuracy
            rhyme_util = RhymeCaculator
            if 'rev' in self.hparams.output_dir:
                print('Compute rhyme metric in reverse order.\n')
                out_rhymes = [rhyme_util.get_rhyme_type_of_line(line[::-1]) for line in preds]
            else:
                print('Compute rhyme metric in normal order.\n')
                out_rhymes = [rhyme_util.get_rhyme_type_of_line(line) for line in preds]
            rhyme_acc = calculate_acc(out=out_rhymes, tgt=batch['tgt_rhymes'].squeeze().tolist())

            # Compute boundary recall
            if 'tgt_stress' in batch:
                if 'rev' in self.hparams.output_dir:
                    output_lines = [line[::-1] for line in preds]
                else:
                    output_lines = preds
                boundary_util = BoundaryRecall()
                boundary_recall = boundary_util.boundary_recall_batch(output_lines, [''.join(str(i) for i in l) for l in
                                                                                     batch['tgt_stress']])
            else:
                boundary_recall = 0
            base_metrics.update(len_acc=len_acc, rhyme_acc=rhyme_acc, boundary_recall=boundary_recall)

        # Compute stress pattern accuracy for some models
        if self.hparams.dataset_class in [
            'Seq2SeqDatasetEmbStr',
            'Seq2SeqDatasetPrefixDecoderStr',
            'Seq2SeqDatasetPrefixEncoderStr',
        ]:
            stress_util = PosSeg()
            if 'rev' in self.hparams.output_dir:
                gen_stress_patterns = [stress_util.get_stress_pattern_list(i[::-1]) for i in preds]
            else:
                gen_stress_patterns = [stress_util.get_stress_pattern_list(i) for i in preds]
            tgt_stress_patterns = batch['tgt_stress']
            stress_acc = calculate_acc_2d(gen_stress_patterns, tgt_stress_patterns)
            base_metrics.update(stress_acc=stress_acc)

        return base_metrics

    def test_step(self, batch, batch_idx):
        '''
        Full forward pass and metric computing in testing
        '''
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        '''
        Construct datasets
        '''
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        if self.hparams.dataset_class == "Seq2SeqDatasetPrefixDecoder":
            dataset = self.dataset_class(
                self.tokenizer,
                type_path=type_path,
                n_obs=n_obs,
                max_target_length=max_target_length,
                constraint_type=self.hparams.constraint_type,
                **self.dataset_kwargs,
            )
        else:
            dataset = self.dataset_class(
                self.tokenizer,
                type_path=type_path,
                n_obs=n_obs,
                max_target_length=max_target_length,
                **self.dataset_kwargs,
            )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        '''
        Construct dataloaders
        '''
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test" and type_path != "val":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test" and type_path != "val":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self) -> DataLoader:
        '''
        Build dataloader for training
        '''
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        '''
        Build dataloader for validating
        '''
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        '''
        Build dataloader for testing
        '''
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)

        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--model_class_name",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--dataset_class",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            ),
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            ),
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            ),
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            ),
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--prompt_tuning", action="store_true", default=False)
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--overwrite_output_dir", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--tokenizer", type=str, default="", required=True)
        parser.add_argument("--constraint_type", type=str, default="", required=True)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help=(
                "-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So"
                " val_check_interval will effect it."
            ),
        )
        parser.add_argument("--config_json", type=str, default='', required=False, help="Path for yaml config file")
        return parser


def train_with_args(args, model=None):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if 'debug' not in args.output_dir:
        check_output_dir(args, expected_items=3)

    # Load tokenizer
    print('Tokenizer path:', args.tokenizer)
    tokenizer = MBart50TokenizerFast.from_pretrained(args.tokenizer)
    tokenizer.src_lang = 'en_XX'
    tokenizer.tgt_lang = 'zh_CN'

    # Construct model
    if model is None:
        assert args.task == 'translation'
        model = TranslationModule(args, tokenizer=tokenizer)

    # Construct logger
    dataset_dir = Path(args.data_dir).name
    if (
            args.logger_name == "default"
            or args.fast_dev_run
            or str(args.output_dir).startswith("/tmp")
            or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset_dir)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset_dir}")

    # Config early stopping
    print(args.early_stopping_patience, type(args.early_stopping_patience))
    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    # Training
    lower_is_better = args.val_metric == "loss"
    trainer: pl.Trainer = generic_train(
        model,
        args,

        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(
            args.output_dir, model.val_metric, args.save_top_k, lower_is_better
        ),
        early_stopping_callback=es_callback,
        logger=logger,
    )

    # Test best model in .ckpt file
    if args.do_predict:
        model.hparams.test_checkpoint = ""
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
        if checkpoints:
            model.hparams.test_checkpoint = checkpoints[-1]
            trainer.resume_from_checkpoint = checkpoints[-1]
        trainer.logger.log_hyperparams(model.hparams)
        trainer.test()

    return model


def parse_arg():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TranslationModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'].strip().split(',')
    args.gpus = len(gpu_list)
    return args


if __name__ == "__main__":
    args = parse_arg()

    if 'debug' in args.output_dir:
        args.data_dir = args.data_dir.replace('full', 'mini')
        args.output_dir = args.output_dir.replace('full', 'mini')
        args.logger_name = 'default'
        args.max_epochs = 2
        args.val_check_interval = 0.99
        args.learning_rate = 1e-3

    print(args.config_json)
    if args.config_json != '' and os.path.exists(args.config_json):
        print('Additional config exists!')
        config = read_json(args.config_json)
        iteration = config['iteration']
        model_direction = config['direction']

        # Update output dir
        output_dir = '../results/full/ibt/iter_{}/model_{}'.format(iteration, model_direction)
        args.output_dir = output_dir

        # Save next iteration num to the json file
        config['iteration'] += 1
        save_json(config, args.config_json)

    train_with_args(args)
