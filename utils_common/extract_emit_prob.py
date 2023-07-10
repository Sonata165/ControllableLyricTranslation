'''
Convert emit prob dict from python to json file
Extract only useful part from emit dict
'''

import os
import re
import sys
from utils_common.utils import save_json, read_json

def _main():
    # obtain_statistics()
    # extract_emit_for_by_ids()
    create_emit_tensor()

def _procedures():
    obtain_statistics()
    extract_emit_for_by_ids()

def obtain_statistics():
    '''
    obtain median value of P['B']
    '''
    from prob_emit import P
    pb = P['B']
    l = [pb[i] for i in pb]
    l.sort()
    median = l[len(l)//2]
    print(l[0], median, l[-1]) # -17.334481908575043 -11.044766337666047 -3.6544978750449433


def extract_emit_for_by_ids():
    # P = read_json('./emit_dict.json')
    from prob_emit import P

    tk = read_json('../BartFinetune/tokenizers/mbart_tokenizer_fast_ch/tokenizer.json')
    vocab = tk['model']['vocab']
    pattern_ch = '[\u4e00-\u9fff]'
    m = -11.0447

    dic = {}
    for i in range(len(vocab)):
        entry = vocab[i]
        if re.search(pattern_ch, entry[0]):  # If contain chinese character
            if len(entry[0]) == 1:
                # print(i)
                # print(entry)
                ch = entry[0]
                if ch in P['B']:
                    log_p = P['B'][ch]
                elif ch in P['S']:
                    log_p = P['S'][ch]
                else:
                    log_p = m
                dic[i] = log_p
    save_json(dic, './emit_dict.json')

def create_emit_tensor():
    import torch
    emit_dict = read_json('./emit_dict.json')
    emit_ts = torch.zeros(size=(250090,), dtype=torch.float)
    for i in range(emit_ts.shape[0]):
        if str(i) in emit_dict:
            emit_ts[i] += emit_dict[str(i)]
        else:
            emit_ts[i] = -17.4
    torch.save(emit_ts, './emit_ts.json')

if __name__ == '__main__':
    _main()
