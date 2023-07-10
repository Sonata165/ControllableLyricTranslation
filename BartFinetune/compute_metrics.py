'''
Compute metrics for output of trained models
'''

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../'))

import argparse
import datetime
import json
import re
import torch
import nltk
import numpy as np
import time
import warnings
import cmudict
import pyter

from logging import getLogger
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from models.MBarts import get_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBart50TokenizerFast

# from LM.ngram.language_model import LanguageModel
from metrics import (
    BoundaryRecall
)
from utils_common.utils import (
    RhymeUtil, save_json, PosSeg,
    FluencyCaculator,
    # , # PerpCaculator #, FluencyCaculator2
    calculate_acc_2d,
)
from utils.utils import (
    calculate_rouge,
    chunks,
    parse_numeric_n_bool_cl_kwargs,
    use_task_specific_params,
    calculate_sacrebleu,
    calculate_sentence_bleu,
)


def compute_scores(args):
    """
    Compute metrics values for output text (given reference text).
    Reference text is optional.
    """

    # if os.path.exists(args.score_path):
    #     raise Exception('score_path {} exists!'.format(args.score_path))
    # ----------

    # Read output and reference
    with open(args.output_path) as f:
        outputs = [x.rstrip() for x in f.readlines()]
    if args.reference_path != None:
        with open(args.reference_path) as f:
            references = [x.rstrip() for x in f.readlines()]
    else:
        references = None

    output_lns = outputs
    reference_lns = references
    if os.path.exists(args.constraint_path):
        constraint_lns = [x.rstrip() for x in open(args.constraint_path).readlines()]
        constraint_stress_lns = [x.rstrip() for x in
                                 open(args.constraint_path.replace('.target', '_boundary.target')).readlines()]
    else:
        print('Constraint path not exist: {}'.format(args.constraint_path))
        constraint_lns = None
    if constraint_lns != None:
        assert len(output_lns) == len(reference_lns) == len(constraint_lns)

    # Compute scores
    scores: dict = calculate_sacrebleu(output_lns, reference_lns)

    if constraint_lns != None:
        # Read constraint target
        tgt_lens, tgt_rhymes = [], []
        for l in constraint_lns:
            t1, t2 = [int(i) for i in l.split('\t')]
            tgt_lens.append(t1)
            tgt_rhymes.append(t2)

        # Compute format accuracy
        out_lens = [len(i.strip()) for i in output_lns]
        len_acc = calculate_acc(out=out_lens, tgt=tgt_lens)
        scores['format_accuracy'] = len_acc

        # Compute rhyme accuracy
        rhyme_util = RhymeUtil()
        if 'rev' in args.output_path:
            print('rhyme in reverse order')
            print(output_lns[0])
            out_rhymes = [rhyme_util.get_rhyme_type_of_line(line[::-1]) for line in output_lns]
        else:
            print('rhyme in normal order')
            print(output_lns[0])
            out_rhymes = [rhyme_util.get_rhyme_type_of_line(line) for line in output_lns]
        print(out_rhymes[:10], tgt_rhymes[:10])
        rhyme_acc = calculate_acc(out=out_rhymes, tgt=tgt_rhymes) # no need unconstrained token, because no 0 in rhyme target of valid and test set
        scores['rhyme_accuracy'] = rhyme_acc

        # Compute stress pattern accuracy
        if 'rev' in args.output_path:
            out_lines = [i[::-1] for i in output_lns] # convert the output to normal order
        else:
            out_lines = output_lns
        boundary_util = BoundaryRecall()
        boundary_recall = boundary_util.boundary_recall_batch(out_lines, constraint_stress_lns)
        scores['boundary_recall'] = boundary_recall

    # Compute Translate Edit Rate (TER)
    ters = [pyter.ter(out, ref) for out, ref in zip(output_lns, reference_lns)]
    ter = sum(ters) / len(ters)
    scores['TER'] = ter

    # # Compute fluency (SLOR metric)
    # if 'rev' in args.output_path:
    #     output_lns = [s[::-1] for s in output_lns]
    # slors, perps, lm_probs, uni_probs = FluencyCaculator.compute_slor(output_lns)
    # # slors = FluencyCaculator.compute_lm_probability(output_lns)
    # v_min, v_max = np.min(slors), np.max(slors)
    # # print(v_min, v_max)
    # # slors = FluencyCaculator.normalize_to_0_and_1(slors, -3.796, 10.232) # min and max need to be updated each time
    # slor = sum(slors) / len(slors)

    def geo_mean(iterable):
        a = np.array(iterable)
        return a.prod() ** (1.0 / len(a))

    # Compute perplexity
    # calc = PerpCaculator()
    # perp = geo_mean(perps)
    # perp = calc(output_lns)
    perp = 0

    # lm_prob = sum(lm_probs) / len(lm_probs)
    # uni_prob = sum(uni_probs) / len(uni_probs)
    # scores['SLOR'] = (slor, v_min, v_max)
    scores['Perplexity'] = perp
    # scores['LM prob'] = lm_prob
    # scores['Unigram Prob'] = uni_prob

    # Save result
    save_json(scores, args.score_path)

    # Metric for result comparison file
    bleus = calculate_sentence_bleu(output_lns, reference_lns)  # Sentence-level BLEU
    scores = {'bleu': ['{:.4f}'.format(i) for i in bleus]}
    if constraint_lns != None:
        ch_count = ['{} / {}'.format(i, j) for i, j in zip(out_lens, tgt_lens)]
        rhy_result = ['{} / {}'.format(i, j) for i, j in zip(out_rhymes, tgt_rhymes)]
        scores['len'] = ch_count
        scores['rhyme'] = rhy_result
    generate_result_comparison_file(args.source_path, args.output_path, args.reference_path, scores=scores)

    return


def count_characters(outputs, refs):
    '''
    given all output and reference lines
    count the number of characters of each line
    return actual_count / target_count
    '''
    pass


def calculate_acc(out, tgt, unconstrained_token=None):
    '''
    Calculate the ratio of same elements
    '''
    assert len(out) == len(tgt)
    cnt_same = 0
    for i in range(len(out)):
        if out[i] == tgt[i]:
            cnt_same += 1
        elif unconstrained_token != None:
            if tgt[i] == unconstrained_token:
                cnt_same += 1
    return cnt_same / len(out)


def calculate_len_dif_with_target_len(output_lns, desired_length):
    '''
    Compute average len dif and maximum len dif, between the output and the desired length
    pred: 一个batch的输出
    tgt： 一个batch对应的target length
    return: [avg_len_dif, max_len_dif]
    '''
    assert len(output_lns) == len(desired_length)
    sentence_num = min(len(output_lns), len(desired_length))
    dif_sum = 0
    max_dif = 0
    for out_s, tgt_s in zip(output_lns, desired_length):
        out_s = re.sub(r"(?![\u4e00-\u9fa5]).", "", out_s).strip()
        dif = abs(len(out_s) - tgt_s)
        dif_sum += dif
        if dif > max_dif:
            max_dif = dif
    return np.array([dif_sum / sentence_num, max_dif])


def generate_result_comparison_file(src_path, output_path, ref_path, scores):
    '''
    Generate a result comparison file to compare the generation result with ground truth.
    scores is a dict that looks like:
        scores = {
            bleu: [bleu of all sentences],
            ter: [ter of all sentences],
        }
    '''
    with open(src_path, 'r') as f:
        srcs = f.readlines()
    with open(output_path, 'r') as f:
        outputs = f.readlines()
    with open(ref_path, 'r') as f:
        refs = f.readlines()
    srcs = [s.strip() for s in srcs]
    outputs = [s.strip() for s in outputs]
    refs = [s.strip() for s in refs]
    if 'rev' in output_path:
        # print('rev in output path')
        # print(refs[:10])
        refs = [s[::-1] for s in refs]
        outputs = [s[::-1] for s in outputs]
    # print('rev not in out path')
    # print(refs[:10])
    # exit(10)

    # Construct file content
    ret = '----------------------------------------\n'
    for i in range(len(outputs)):
        ret += 'Sentence {}'.format(i + 1)
        for k in scores:
            ret += ' | {}: {}'.format(k, scores[k][i])
        ret += '\n'
        src_s = srcs[i]
        ref_s = refs[i]
        out_s = outputs[i]
        ret += 'src: {}\n' \
               'ref: {}\n' \
               'out: {}\n' \
               '----------------------------------------\n'.format(src_s, ref_s, out_s)

    with open(output_path.replace('output.txt', 'result.txt'), 'w') as f:
        f.write(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_path", type=str, required=True, help="like cnn_dm/test_output.txt.")
    parser.add_argument("--output_path", type=str, required=True, help="like cnn_dm/test_output.txt.")
    parser.add_argument("--constraint_path", type=str, required=True, help="like cnn_dm/test.target")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=True, default="metrics.json", help="where to save metrics")

    args = parser.parse_args()

    compute_scores(args)
