#!/usr/bin/env python

import os
import sys

import utils.utils

sys.path.insert(1, os.path.join(sys.path[0], '../'))

import re
import json
import time
import nltk
import torch
import cmudict
import warnings
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from logging import getLogger
from pathlib import Path
from typing import Dict, List
from datasets import load_metric
from models.MBarts import get_model
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBart50TokenizerFast

from utils_common.utils import RhymeUtil, read_json, jpath, RhymeCaculator, ls
from utils.utils import (
    calculate_rouge,
    chunks,
    parse_numeric_n_bool_cl_kwargs,
    use_task_specific_params,
    calculate_sacrebleu,
    get_dataset_by_type,
)

from models.MBarts import ForcedBOSTokenLogitsProcessorPrefixDecoder, ForcedBOSTokenLogitsProcessorPrefixDecoderLength, \
    ForcedBOSTokenLogitsProcessorPrefixDecoderN

# sns.set_theme(font_scale=0.8)
# plt.style.use('seaborn')

logger = getLogger(__name__)


def _main():
    run_generate(verbose=True)
    # test_trimmer()


def generate_translations(
        examples: List[str],
        out_file: str,
        model_name: str,
        batch_size: int = 8,
        device: str = 'none',
        fp16=False,
        task="translation",
        prefix=None,
        ref=None,
        model_class_name=None,
        constraints=None,
        tokenizer_path=None,
        src_lang='en_XX',
        args=None,
        **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""

    # 初始化model和tokenizer
    model_name = str(model_name)
    model = get_model(model_class_name, model_name, None, None).to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = 'zh_CN'
    print('Uncontrol rhyme id:', tokenizer.convert_tokens_to_ids('rhy_0'))

    if fp16:
        model = model.half()

    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    start_time = time.time()

    # Update config with task specific params
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""

    # Never generate <unk>
    bad_words_ids = [[3]]

    # Generate kwargs
    generate_kwargs['bad_words_ids'] = bad_words_ids
    print('Generate kwargs:', generate_kwargs)

    do_generate = True
    do_recommend_rhyme = True
    do_collate = True

    print('You are testing: ' + model_name)

    # Prepare path
    out_root = os.path.dirname(out_file)
    dataset_root = '../Dataset/datasets/real' #_letitgo' | _5
    songs = ls(dataset_root)
    songs.sort()
    for song in songs:
        song_path = jpath(dataset_root, song)
        pars = ls(song_path)
        pars.sort()
        for par in pars:
            out_dir_song = jpath(out_root, song)
            if not os.path.exists(out_dir_song):
                os.mkdir(out_dir_song)
            out_dir = jpath(out_root, song, par)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            print('Output dir:', out_dir)
            dataset_path = jpath(dataset_root, song, par)
            out_file = jpath(out_dir, 'output.txt')  # output file path
            args.input_path = dataset_path

            if do_generate == True:
                fout = Path(out_file).open("w", encoding="utf-8")

                # Prepare dataset
                dataset_kwargs: dict = dict(
                    src_lang=args.src_lang,
                    tgt_lang=args.tgt_lang,
                )
                dataset_class = get_dataset_by_type(args.dataset_class)
                dataset = dataset_class(
                    tokenizer=tokenizer,
                    data_dir=args.input_path,
                    max_source_length=30,
                    max_target_length=30,
                    type_path='test',
                    constraint_type=args.constraint_type,
                    **dataset_kwargs,
                )
                dataloader = DataLoader(
                    dataset=dataset,
                    collate_fn=dataset.collate_fn,
                    batch_size=args.bs,
                    shuffle=False,
                )

                decode_step = 0

                # Decoder-side prompt to control length and rhyme

                if args.dataset_class == 'Seq2SeqDatasetPrefixEncoderBdr': # for final model
                    for idx, batch in enumerate(tqdm(dataloader)):
                        decode_step += 1

                        for k in batch:
                            if isinstance(batch[k], torch.Tensor):
                                batch[k] = batch[k].to(device)

                        if do_recommend_rhyme == True:
                            # Rhyme recommendation
                            rhyme_dist, rhyme_rec = recommend_rhyme(model, batch)
                            # print('rec:', rhyme_rec)
                            x = ['0']
                            for k in RhymeCaculator.rhyme_14_dic:
                                x.append(str(k) + ': ' + str(RhymeCaculator.rhyme_14_dic[k]))

                            # Plot rhyme prob distribution
                            for file_type in ['png', 'pdf']:
                                fig = plt.figure(figsize=(5,4))
                                fig.set_tight_layout(True)
                                bars = plt.barh(x, rhyme_dist, edgecolor="none")
                                plt.bar_label(bars, fmt='%.3f')
                                # plt.yticks(rotation=45)
                                plt.savefig(jpath(out_dir, 'rhyme_dist.{}'.format(file_type)))

                            # Write rhyme results to constraint file
                            constraint_path = jpath(dataset_path, 'constraints', 'source', 'test.target')
                            with open(constraint_path) as f:
                                cons = f.readlines()
                            cons_new = []
                            for l in cons:
                                length, _ = l.strip().split('\t')
                                length = int(length)
                                cons_new.append('{}\t{}\n'.format(length, rhyme_rec))
                            with open(constraint_path, 'w') as f:
                                f.writelines(cons_new)

                            # Force rhyme type
                            # rhyme_rec = 13

                            # Prepare decoder input ids
                            n = 1
                            decoder_input_ids = batch['labels'][:, :n + 1].clone()  # [BS, 3]
                            target_rhyme_type = rhyme_rec
                            target_rhyme_id = tokenizer.convert_tokens_to_ids('rhy_{}'.format(target_rhyme_type))
                            decoder_input_ids[:, 0] = target_rhyme_id
                            decoder_input_ids[:, n] = 2  # set the 2nd col to decoder_start_token_id
                        else:
                            rhyme_rec = batch['tgt_rhymes']
                            # print(rhyme_rec)

                            # Prepare decoder input ids
                            n = 1
                            decoder_input_ids = batch['labels'][:, :n + 1].clone()  # [BS, 3]
                            target_rhyme_type = rhyme_rec
                            target_rhyme_id = tokenizer.convert_tokens_to_ids(['rhy_{}'.format(i) for i in target_rhyme_type])
                            decoder_input_ids[:, 0] = torch.tensor(target_rhyme_id)
                            decoder_input_ids[:, n] = 2  # set the 2nd col to decoder_start_token_id

                        # Prepare logits processor for forced bos token id
                        bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderN(
                            bos_token_id=250025,
                            prefix_length=n,
                        )

                        assert args.force == 'no'
                        output = model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=True,
                            no_repeat_ngram_size=2,

                            decoder_input_ids=decoder_input_ids,
                            logits_processor=[bos_processor],

                            num_beams=generate_kwargs['num_beams'],
                            max_length=generate_kwargs['max_length'],
                        )

                        translation = output
                        dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                        for hypothesis in dec:
                            fout.write(hypothesis[::-1] + "\n")
                            fout.flush()
                elif '/baseline_' in model_name: # for baseline
                    for idx, batch in enumerate(tqdm(dataloader)):
                        for k in batch:
                            if isinstance(batch[k], torch.Tensor):
                                batch[k] = batch[k].to(device)
                        assert args.force == 'no'
                        translation = model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            forced_bos_token_id=250025,
                            **generate_kwargs,
                        )
                        dec = tokenizer.batch_decode(translation, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
                        for hypothesis in dec:
                            fout.write(hypothesis + "\n")
                            fout.flush()
                elif args.dataset_class == 'Seq2SeqDatasetPrefixLengthRhyme': # len + rhy
                    print('Inference function: length control, decoder prefix')
                    for idx, batch in enumerate(tqdm(dataloader)):
                        for k in batch:
                            if isinstance(batch[k], torch.Tensor):
                                batch[k] = batch[k].to(device)

                        bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderLength(bos_token_id=250025)

                        # Prepare decoder input ids
                        decoder_input_ids = batch['labels'][:, :2].clone()  # [BS, 3]
                        decoder_input_ids[:, 1] = 2

                        assert args.force == 'no'
                        output = model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            decoder_input_ids=decoder_input_ids,
                            logits_processor=[bos_processor],
                            bad_words_ids=bad_words_ids,
                            num_beams=generate_kwargs['num_beams'],
                            max_length=generate_kwargs['max_length'],
                        )

                        translation = output

                        dec = tokenizer.batch_decode(translation, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)

                        for hypothesis in dec:
                            fout.write(hypothesis[::-1] + "\n")
                            fout.flush()
                elif args.dataset_class == 'Seq2SeqDatasetPrefixEncoderLength': # for len only
                    for idx, batch in enumerate(tqdm(dataloader)):
                        for k in batch:
                            if isinstance(batch[k], torch.Tensor):
                                batch[k] = batch[k].to(device)

                        output = model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            forced_bos_token_id=250025,  # should be set for mBART-50
                            bad_words_ids=bad_words_ids,
                            num_beams=generate_kwargs['num_beams'],
                            max_length=generate_kwargs['max_length'],
                        )

                        dec = tokenizer.batch_decode(output, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
                        for hypothesis in dec:
                            fout.write(hypothesis + "\n")
                            fout.flush()
                fout.close()

    if do_collate == True:
        print('Collating ...')
        for song in tqdm(songs):
            text_song = []
            out_dir_song = jpath(out_root, song)
            pars = ls(out_dir_song)
            pars.sort()
            for par in pars:
                if 'txt' in par:
                    continue
                out_dir_par = jpath(out_dir_song, par)
                out_file_par = jpath(out_dir_par, 'output.txt')
                with open(out_file_par) as f:
                    text_par = f.readlines()
                text_song += text_par
                text_song.append('\n')
            out_file_song = jpath(out_dir_song, 'output.txt')
            with open(out_file_song, 'w') as f:
                f.writelines(text_song)


def recommend_rhyme(model, batch):
    # Prepare decoder input
    # print(batch['labels'])
    # decoder_input_ids = batch['labels'][:, :4].clone()  # [BS, 3]
    # print(decoder_input_ids.shape)
    bs = batch['labels'].shape[0]
    decoder_input_ids = torch.zeros(size=(bs, 4), dtype=torch.long, device=batch['labels'].device)
    decoder_input_ids[:, 0] = 250076  # Uncontrol rhyme
    decoder_input_ids[:, 1] = 2  # set the 2nd col to decoder_start_token_id
    decoder_input_ids[:, 2] = 250025
    decoder_input_ids[:, 3] = 6

    # Forward pass
    outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"],
                    decoder_input_ids=decoder_input_ids, use_cache=False)
    out = outputs['logits']
    prob = torch.softmax(out[:, 3, :], 1) # [BS, vocab_size]

    rhyme_prob = token_prob_to_rhyme_prob(prob) # list, len=14
    rhyme_prob.pop(0)
    rhyme_prob = torch.softmax(torch.tensor(rhyme_prob), dim=0)
    rhyme_prob = rhyme_prob.tolist()
    rhyme_prob.insert(0, 0)

    # Get index of the k-th largest element
    recommend_k_th_largest = 1
    k = recommend_k_th_largest
    i = [rhyme_prob.index(x) for x in sorted(rhyme_prob, reverse=True)[:k]][-1]

    return rhyme_prob, i


def token_prob_to_rhyme_prob(prob):
    '''
    Convert token distribution to rhymeclass distribution
    '''
    rhyme_dic = read_json(os.path.dirname(__file__) + '/tokenizers/misc/rhyme_type_dic.json')
    ret = [0 for i in range(15)]
    for rhyme in rhyme_dic:
        ids = [int(i) for i in rhyme_dic[rhyme]]
        rhyme = int(rhyme)
        for i in range(prob.shape[0]):  # for each sentence
            j = i + 1  # for last j'th sentence
            if j % 2 == 1:
                weight = 1
            else:
                weight = 1
            prob_s = prob[-j]
            ret[rhyme] += weight * sum(prob_s[ids]).item()
    return ret


def rhyme_prob_weighted_sum(prob):
    return


def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_generate(verbose=True):
    """

    Takes input text, generates output, and then using reference calculates the BLEU scores.

    The results are saved to a file and returned to the caller, and printed out unless ``verbose=False`` is passed.

    Args:
        verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): print results to stdout

    Returns:
        a tuple: ``(scores, params}``
        - ``scores``: a dict of scores data ``{'bleu': 39.6501, 'n_obs': 2000, 'runtime': 186, 'seconds_per_sample': 0.093}``
        - ``params``: a dict of custom params, e.g. ``{'num_beams': 5, 'length_penalty': 0.8}``
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("save_path", type=str, help="where to save summaries")
    parser.add_argument(
        "--model_class_name",
        required=True,
        type=str,
    )
    parser.add_argument("--reference_path", type=str, required=False, default='none', help="like cnn_dm/test.target")
    parser.add_argument("--constraint_path", type=str, required=True, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=device, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--tokenizer", type=str, default="", required=True, help="path of tokenizer")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--src_lang", type=str, default="en_XX", required=True)
    parser.add_argument("--tgt_lang", type=str, default="zh_CN", required=False)
    parser.add_argument("--dump-args", action="store_true", help="print the custom args with the results")
    parser.add_argument("--constraint_type", type=str, default="reference", required=True)
    parser.add_argument("--dataset_class", type=str, default="", required=False)
    parser.add_argument("--force", type=str, default="", required=True)
    parser.add_argument(
        "--info",
        nargs="?",
        type=str,
        const=datetime_now(),
        help="use in conjunction w/ --dump-args to print with the results whatever other info you'd like, e.g. lang=en-ru. If no value is passed, the current datetime string will be used.",
    )
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    print(rest)
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")

    # Read input and target files
    # with open(args.input_path) as f:
    #     examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in f.readlines()]
    examples = None

    if args.reference_path != 'none':
        with open(args.reference_path) as f:
            references = [x.rstrip() for x in f.readlines()]
    else:
        references = None

    if os.path.exists(args.constraint_path):
        # args.constraint_path.exists():
        with open(args.constraint_path) as f:
            constraints = [x.rstrip() for x in f.readlines()]
    else:
        constraints = None

    if args.n_obs > 0:
        examples = examples[: args.n_obs]
        references = references[: args.n_obs]

    Path(args.save_path).parent.mkdir(exist_ok=True, parents=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")

    generate_translations(
        examples,
        args.save_path,
        args.model_name,
        model_class_name=args.model_class_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        prefix=args.prefix,
        ref=references,
        constraints=constraints,
        tokenizer_path=args.tokenizer,
        src_lang=args.src_lang,
        args=args,
        **parsed_args,
    )

    return


class BeamTrimmer:
    '''
    Receive output of multiple beams
    Trim beams that doesn't fulfill the constraints
    Return the best beam
    (Operate on batch)
    '''

    def __init__(self, target, beam_size):
        assert target in ['length', 'rhyme_end', 'rhyme_start']
        self.target = target  # length, rhyme_end, rhyme_start
        self.beam_size = beam_size

        # Select proper trim function
        if self.target == 'length':
            self.trim_func = self.trim_by_length
        elif self.target == 'rhyme_end':
            self.trim_func = self.trim_by_rhyme_end
        elif self.target == 'rhyme_start':
            self.trim_func = self.trim_by_rhyme_start

    def trim(self, batch_seqs, batch_scores, constraints):
        # constraint_list = constraints.tolist()
        # batch_scores = batch_scores.tolist()
        # Trim sentence sample-by-sample
        batch_output = []
        for i in range(0, len(batch_seqs), self.beam_size):
            j = i + self.beam_size  # end index of a sample
            seqs = batch_seqs[i:j]  # extract multiple beams of sequence from one sample
            scores = batch_scores[i:j]
            res = self.trim_func(seqs, scores, constraints.pop(0))
            batch_output.append(res)
        return batch_output

    def trim_by_length(self, seqs, scores, constraint):
        '''
        seqs:
        scores: tensor
        constraint: a number
        '''
        # print(seqs, scores, constraint)
        seq_n_scores = list(zip(seqs, scores))
        seq_n_scores = [seq_n_score for seq_n_score in seq_n_scores if len(seq_n_score[0]) == constraint]
        # sorted(seq_n_scores.sorted()
        seq_n_scores = sorted(seq_n_scores, key=lambda tup: tup[1],
                              reverse=True)  # sort sequence by score from high to low
        if len(seq_n_scores) > 0:
            return seq_n_scores[0][0]
        else:
            print(seqs)
            print(constraint)
            raise Exception('no valid sequence')

    def trim_by_rhyme_end(self, seqs, scores, constraint):
        pass

    def trim_by_rhyme_start(self, seqs, scores, constraint):
        pass


def test_trimmer():
    # tokenizer = MBart50TokenizerFast.from_pretrained('./tokenizers/mbart_tokenizer_fast_ch')
    # tokenizer.tgt_lang = 'zh_CN'

    # with tokenizer.as_target_tokenizer():
    #     ids = tokenizer([
    #         '哈哈', '哈哈大笑', '傻呵呵', '嘿嘿'])['input_ids']
    batch_seqs = [
        '哈哈',
        '哈嘿',
        '哈哈大笑',
        '傻呵呵',
        '嘿嘿',
        '傻乎乎',
    ]
    scores = [
        0.5,
        0.4,
        0.2,
        0.3,
        1,
        0.4,
    ]
    constraints = [
        2, 3,
    ]

    trimmer = BeamTrimmer(target='length', beam_size=3)
    ret = trimmer.trim(batch_seqs=batch_seqs,
                       batch_scores=scores,
                       constraints=constraints)
    print(ret)


if __name__ == "__main__":
    _main()
