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
from torch.utils.data import DataLoader

from logging import getLogger
from pathlib import Path
from typing import Dict, List
from datasets import load_metric
from models.MBarts import get_model
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBart50TokenizerFast

from utils_common.utils import RhymeUtil
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
    fout = Path(out_file).open("w", encoding="utf-8")

    # 初始化model和tokenizer
    model_name = str(model_name)
    model = get_model(model_class_name, model_name, None, None).to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = 'zh_CN'

    if fp16:
        model = model.half()

    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    start_time = time.time()

    # Update config with task specific params
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""

    # Force related
    if args.force == 'length':
        args.bs = 1

    # Never generate <unk>
    bad_words_ids = [[3]]

    # Generate kwargs
    generate_kwargs['bad_words_ids'] = bad_words_ids
    print('Generate kwargs:', generate_kwargs)

    # Prepare dataset
    print('Dataset Class: ', args.dataset_class)
    dataset_kwargs: dict = dict(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )
    dataset_class = get_dataset_by_type(args.dataset_class)
    dataset = dataset_class(
        tokenizer=tokenizer,
        data_dir=os.path.dirname(args.input_path),
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

    print('You are testing: ' + model_name)
    # Decoder-side prompt to control length and rhyme
    if '/prefix_decoder_' in model_name:  # Prefix Decoder
        print('Inference function: length and rhyme control by prefix decoder')
        for t in tqdm(list(zip(
                list(chunks(examples, batch_size)), list(chunks(ref, batch_size)), list(chunks(constraints, batch_size))
        ))):
            batch_input_line, batch_ref_line, batch_constraint_line = t
            batch_input_line = [prefix + text for text in batch_input_line]  # a batch of input sentences
            batch_input = tokenizer(batch_input_line,
                                    return_tensors="pt",
                                    truncation=True,
                                    padding="longest").to(device)
            batch_tgt_len, batch_tgt_rhyme = [], []
            for line in batch_constraint_line:
                t1, t2 = line.split('\t')
                tgt_len, tgt_rhyme = 'len_{}'.format(t1), 'rhy_{}'.format(t2)
                batch_tgt_len.append(tgt_len)
                batch_tgt_rhyme.append(tgt_rhyme)
            batch_tgt_len = tokenizer(batch_tgt_len,
                                      return_tensors="pt",
                                      add_special_tokens=False,
                                      max_length=1,
                                      padding=False,
                                      truncation=True, ).to(device).input_ids
            batch_tgt_rhyme = tokenizer(batch_tgt_rhyme,
                                        return_tensors="pt",
                                        add_special_tokens=False,
                                        max_length=1,
                                        padding=False,
                                        truncation=True, ).to(device).input_ids

            # Prepare decoder input ids
            decoder_input_ids = torch.cat(
                (batch_tgt_len, batch_tgt_rhyme,
                 torch.ones((batch_tgt_len.shape[0], 1), device=device, dtype=torch.int64) * 2), dim=1
            ).to(device)

            # Prepare logits processor for forced bos token id
            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoder(bos_token_id=250025)

            translation = model.generate(
                input_ids=batch_input.input_ids,
                attention_mask=batch_input.attention_mask,
                decoder_input_ids=decoder_input_ids,
                logits_processor=[bos_processor],
                # forced_bos_token_id=250025,
                # length_penalty=1.0,
                **generate_kwargs,
            )
            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    # Decoder-side prompt to control length
    elif ('/len_decoder_prefix' in model_name or
          args.dataset_class == 'Seq2SeqDatasetPrefixLengthRhyme'
    ):
        print('Inference function: length control, decoder prefix')
        for idx, batch in enumerate(tqdm(dataloader)):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderLength(bos_token_id=250025)

            # Prepare decoder input ids
            decoder_input_ids = batch['labels'][:, :2].clone()  # [BS, 3]
            decoder_input_ids[:, 1] = 2

            if args.force == 'length':
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    logits_processor=[bos_processor],
                    bad_words_ids=bad_words_ids,
                    num_beams=generate_kwargs['num_beams'],
                    min_length=batch['tgt_lens'].item() + 5,  # force minimum gen len, only work when bs=1
                    max_length=batch['tgt_lens'].item() + 5,  # s/a
                )
            elif args.force == 'no':
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    logits_processor=[bos_processor],
                    bad_words_ids=bad_words_ids,
                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                )

            else:
                raise Exception('Wrong force type.')

            translation = output

            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    # Decoder-side prompt to control rhyme
    elif '/rhy_decoder_prefix' in model_name:
        print('Inference function: length control, decoder prefix')
        for t in tqdm(list(zip(
                list(chunks(examples, batch_size)), list(chunks(ref, batch_size)), list(chunks(constraints, batch_size))
        ))):
            batch_input_line, batch_ref_line, batch_constraint_line = t
            batch_input_line = [prefix + text for text in batch_input_line]  # a batch of input sentences
            batch_input = tokenizer(batch_input_line,
                                    return_tensors="pt",
                                    truncation=True,
                                    padding="longest").to(device)
            batch_tgt_len, batch_tgt_rhyme = [], []
            for line in batch_constraint_line:
                t1, t2 = line.split('\t')
                tgt_len, tgt_rhyme = 'len_{}'.format(t1), 'rhy_{}'.format(t2)
                batch_tgt_len.append(tgt_len)
                batch_tgt_rhyme.append(tgt_rhyme)

            # batch_tgt_len = tokenizer(batch_tgt_len,
            #                           return_tensors="pt",
            #                           add_special_tokens=False,
            #                           max_length=1,
            #                           padding=False,
            #                           truncation=True, ).to(device).input_ids
            batch_tgt_rhyme = tokenizer(batch_tgt_rhyme,
                                        return_tensors="pt",
                                        add_special_tokens=False,
                                        max_length=1,
                                        padding=False,
                                        truncation=True, ).to(device).input_ids

            # Prepare decoder input ids
            decoder_input_ids = torch.cat(
                (batch_tgt_rhyme,
                 torch.ones((batch_tgt_rhyme.shape[0], 1), device=device, dtype=torch.int64) * 2), dim=1
            ).to(device)

            # Prepare logits processor for forced bos token id
            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderLength(bos_token_id=250025)

            translation = model.generate(
                input_ids=batch_input.input_ids,
                attention_mask=batch_input.attention_mask,
                decoder_input_ids=decoder_input_ids,
                logits_processor=[bos_processor],
                **generate_kwargs,
            )
            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    # Encoder-side prompt to control length and rhyme
    elif '/encoder_prefix_' in model_name:  #
        for t in tqdm(list(zip(
                list(chunks(examples, batch_size)), list(chunks(constraints, batch_size))
        ))):
            batch_input_line, batch_constraint_line = t
            batch_input_line = [prefix + text for text in batch_input_line]  # a batch of input sentences
            batch_input = tokenizer(batch_input_line,
                                    return_tensors="pt",
                                    truncation=True,
                                    padding="longest").to(device)
            batch_tgt_len, batch_tgt_rhyme = [], []
            for line in batch_constraint_line:
                t1, t2 = line.split('\t')
                tgt_len, tgt_rhyme = 'len_{}'.format(t1), 'rhy_{}'.format(t2)
                batch_tgt_len.append(tgt_len)
                batch_tgt_rhyme.append(tgt_rhyme)
            t1 = tokenizer(batch_tgt_len,
                           return_tensors="pt",
                           add_special_tokens=False,
                           max_length=1,
                           padding=False,
                           truncation=True, ).to(device)
            t2 = tokenizer(batch_tgt_rhyme,
                           return_tensors="pt",
                           add_special_tokens=False,
                           max_length=1,
                           padding=False,
                           truncation=True, ).to(device)

            # Prepare prompted encoder input
            batch_tgt_len = t1.input_ids
            batch_tgt_len_attn = t1.attention_mask
            batch_tgt_rhyme = t2.input_ids
            batch_tgt_rhyme_attn = t2.attention_mask
            input_ids = torch.cat((batch_tgt_len, batch_tgt_rhyme, batch_input.input_ids), dim=1)
            attention_mask = torch.cat((batch_tgt_len_attn, batch_tgt_rhyme_attn, batch_input.attention_mask), dim=1)

            translation = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=250025,
                **generate_kwargs,
            )
            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    # Encoder-side prompt to control length
    elif '/len_encoder_prefix' in model_name:
        for idx, batch in enumerate(tqdm(dataloader)):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            if args.force == 'length':
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                    forced_bos_token_id=250025,  # should be set for mBART-50
                    bad_words_ids=bad_words_ids,
                    num_beams=generate_kwargs['num_beams'],
                    min_length=batch['tgt_lens'].item() + 4,  # force minimum gen len, only work when bs=1
                    max_length=batch['tgt_lens'].item() + 4,  # s/a
                )
            elif args.force == 'no':
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    forced_bos_token_id=250025,  # should be set for mBART-50
                    bad_words_ids=bad_words_ids,
                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                )
            elif args.force == 'rhyme_first':
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    forced_bos_token_id=250025,  # should be set for mBART-50
                    bad_words_ids=bad_words_ids,
                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                    tgt_rhymes=batch['tgt_rhymes']
                )
            elif args.force == 'rhyme_last':
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    forced_bos_token_id=250025,  # should be set for mBART-50
                    bad_words_ids=bad_words_ids,
                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                    tgt_rhymes=batch['tgt_rhymes'],
                    tgt_length=batch['tgt_lens'],
                )
            elif args.force == 'bdr':
                # Biased decoding for boundary control
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    forced_bos_token_id=250025,  # should be set for mBART-50
                    bad_words_ids=bad_words_ids,
                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                    bdr_pos=batch['bdr_pos']
                )
            else:
                raise Exception('Wrong force type.')

            dec = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    # Encoder-side prompt to control rhyme
    elif '/rhy_encoder_prefix' in model_name or args.dataset_class in [
        'Seq2SeqDatasetLenEncRhyEnc',
    ]:
        for idx, batch in enumerate(tqdm(dataloader)):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            output = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                forced_bos_token_id=250025,  # should be set for mBART-50
                num_beams=generate_kwargs['num_beams'],
                max_length=generate_kwargs['max_length'],
                min_length=1 + 4,
            )
            dec = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    # Baseline and other unconstrained model
    elif ('/baseline_' in model_name
          or '/unconstrained-' in model_name
          or '/pretrain_' in model_name
          or '/denoise_ft_' in model_name
          or '/ft_in_domain_pt_' in model_name
          or '/ft_denoise_ft_' in model_name
          or 'mmt' in model_name
    ):
        for idx, batch in enumerate(tqdm(dataloader)):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            if args.force == 'no':
                translation = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    forced_bos_token_id=250025,
                    **generate_kwargs,
                )
            elif args.force == 'length':
                translation = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    forced_bos_token_id=250025,
                    num_beams=generate_kwargs['num_beams'],
                    min_length=batch['tgt_lens'].item() + 4,  # force minimum gen len, only work when bs=1
                    max_length=batch['tgt_lens'].item() + 4,  # s/a
                )
            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    elif ('/len_emb_' in model_name or args.dataset_class in [
        'Seq2SeqDatasetLenEncRhyEmb',
    ]):
        # trimmer = BeamTrimmer(target='length', beam_size=5)
        for idx, batch in enumerate(tqdm(dataloader)):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            if args.force == 'length':
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                    forced_bos_token_id=250025,  # should be set for mBART-50
                    emb_ids=batch["emb_ids"],
                    num_beams=generate_kwargs['num_beams'],
                    min_length=batch['tgt_lens'].item() + 4,  # force minimum gen len, only work when bs=1
                    max_length=batch['tgt_lens'].item() + 4,  # s/a
                )
                # If we want return multiple beams, we need
                # num_return_sequences=5,
                # output_scores=True,
                # return_dict_in_generate=True,
            elif args.force == 'no':
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                    forced_bos_token_id=250025,  # should be set for mBART-50
                    emb_ids=batch["emb_ids"],
                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                )
            else:
                raise Exception('Wrong force type.')

            translation = output

            # translation = output.sequences
            # scores = output.sequences_scores.cpu().tolist()
            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Check force length fulfilled
            if args.force == 'length':
                if len(dec[0]) != batch['tgt_lens'].item():
                    print(translation)
                    print(dec)
                    print('target length:', batch['tgt_lens'].item())
                    print(batch["emb_ids"])
                    exit(10)

            # # Trim output
            # try:
            #     dec = trimmer.trim(batch_seqs=dec,
            #                        batch_scores=scores,
            #                        constraints=batch['tgt_lens'].cpu().tolist())
            # except:
            #     print(batch)
            #     print(dec)
            #     exit(20)

            # dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
            # exit(10)
    elif ('/rhy_emb_' in model_name):
        for idx, batch in enumerate(tqdm(dataloader)):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            if args.force == 'length':
                raise Exception
            elif args.force == 'no':
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                    forced_bos_token_id=250025,  # should be set for mBART-50
                    emb_ids=batch["emb_ids"],
                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                )
            else:
                raise Exception('Wrong force type.')

            translation = output
            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    elif ('/stress_emb_' in model_name
        or args.dataset_class in [
        'Seq2SeqDatasetEmbBdr',
    ]):
        for idx, batch in enumerate(tqdm(dataloader)):
            decode_step += 1

            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            # Prepare decoder input ids
            n = 1
            decoder_input_ids = batch['labels'][:, :n + 1].clone()  # [BS, 3]
            decoder_input_ids[:, n] = 2

            # Prepare logits processor for forced bos token id
            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderLength(bos_token_id=250025)

            if args.force == 'length':
                raise NotImplementedError
            elif args.force == 'no':
                # print(batch['emb_ids'].shape)
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,

                    decoder_input_ids=decoder_input_ids,
                    logits_processor=[bos_processor],

                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                    emb_ids=batch["emb_ids"],
                )
            else:
                raise Exception('Wrong force type.')

            # if decode_step == 3:
            #     exit(100)

            translation = output
            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    elif args.dataset_class in [
        'Seq2SeqDatasetPrefixDecoderStr',
        'Seq2SeqDatasetPrefixDecoderBdr',
    ]:
        for idx, batch in enumerate(tqdm(dataloader)):
            decode_step += 1

            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            # Prepare decoder input ids
            # Prepare decoder input ids
            n = 21
            decoder_input_ids = batch['labels'][:, :n + 1].clone()  # [BS, 3]
            decoder_input_ids[:, n] = 2  # set the 2nd col to decoder_start_token_id

            # Prepare logits processor for forced bos token id
            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderN(
                bos_token_id=250025,
                prefix_length=n,
            )

            if args.force == 'length':
                raise NotImplementedError
            elif args.force == 'no':
                # print(batch['emb_ids'].shape)
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,

                    decoder_input_ids=decoder_input_ids,
                    logits_processor=[bos_processor],

                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                )
            else:
                raise Exception('Wrong force type.')

            translation = output
            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    elif args.dataset_class in [
        'Seq2SeqDatasetPrefixEncoderStr',
        'Seq2SeqDatasetPrefixEncoderBdr',
        'Seq2SeqDatasetPrefixEncoderBdrDenoise',
        'Seq2SeqDatasetPrefixEncoderBdrNoRhy',
    ]:
        for idx, batch in enumerate(tqdm(dataloader)):
            decode_step += 1

            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            # Prepare decoder input ids
            n = 1
            decoder_input_ids = batch['labels'][:, :n + 1].clone()  # [BS, 3]
            decoder_input_ids[:, n] = 2  # set the 2nd col to decoder_start_token_id

            # Prepare logits processor for forced bos token id
            bos_processor = ForcedBOSTokenLogitsProcessorPrefixDecoderN(
                bos_token_id=250025,
                prefix_length=n,
            )

            if args.force == 'length':
                raise NotImplementedError
            elif args.force == 'no':
                # print(batch['emb_ids'].shape)
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,

                    decoder_input_ids=decoder_input_ids,
                    logits_processor=[bos_processor],

                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                )
            elif args.force == 'bdr':
                # Biased decoding for boundary control
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,

                    decoder_input_ids=decoder_input_ids,
                    logits_processor=[bos_processor],

                    num_beams=generate_kwargs['num_beams'],
                    max_length=generate_kwargs['max_length'],
                    bdr_pos=batch['bdr_pos']
                )
            else:
                raise Exception('Wrong force type.')

            translation = output
            dec = tokenizer.batch_decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()
    else:
        die

    fout.close()
    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(examples)
    # return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4)), desired_length_list

    out_path = str(out_file)
    if 'rev' in out_path and 'real' in out_path:
        with open(out_path) as f:
            text = f.readlines()
        test_r = [i.strip()[::-1]+'\n' for i in text]
        with open(out_path.replace('.txt', 'normal.txt'), 'w') as f:
            f.writelines(test_r)

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
    with open(args.input_path) as f:
        examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in f.readlines()]

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
