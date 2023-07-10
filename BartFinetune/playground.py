import os
import sys
import torch
import torch.nn.functional as F

from models.MBarts import MBartForConditionalGenerationCharLevel, MBart50TokenizerFast


def _main():
    # Load model from local checkpoints
    # tokenizer = MBart50TokenizerFast.from_pretrained("./tokenizers/mbart_tokenizer_fast_ch_prefix_lrs2")
    # model = MBartForConditionalGenerationCharLevel.from_pretrained('../results/final_rev_data_parallel_1/best_tfmr')

    # Load from huggingface
    model = MBartForConditionalGenerationCharLevel.from_pretrained('LongshenOu/lyric-trans-en2zh')
    tokenizer = MBart50TokenizerFast.from_pretrained('LongshenOu/lyric-trans-en2zh')

    device = 'cuda'
    model.to(device)
    print('Load complete')
    text = [
        "There's only one song left for you",
        'Get me off the streets of this city',
        'You only left one kiss for me',
    ]
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = 'zh_CN'
    encoded_input = tokenizer(text, return_tensors="pt", padding=True).to(device)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    print('----- Input -----')
    for line in text:
        print(line)
    print('-----------------')

    # Length constraints
    desired_length = [12, 9, 8]
    tgt_lens = ['len_{}'.format(x) for x in desired_length]
    t1 = tokenizer(
        tgt_lens,
        add_special_tokens=False,
        return_tensors='pt',
        max_length=1,
        padding=False,
        truncation=True,
    )
    tgt_lens = t1['input_ids'].to(device)
    attn_len = t1['attention_mask'].to(device)

    # Rhyme constraint
    desired_rhyme = [1, 1, 1]  # type 1 for all three sentences: rhyme {a, ia, ua}
    tgt_rhymes = ['rhy_{}'.format(x) for x in desired_rhyme]
    t2 = tokenizer(
        tgt_rhymes,
        add_special_tokens=False,
        return_tensors='pt',
        max_length=1,
        padding=False,
        truncation=True,
    )
    tgt_rhymes = t2['input_ids'].to(device)

    # Process target stress constraint
    desired_boundary = [
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
    ]
    tgt_stress = [''.join(['str_{}'.format(i) for i in x[::-1]]) for x in desired_boundary]
    t3 = tokenizer(
        tgt_stress,
        return_tensors='pt',
        add_special_tokens=False,
        padding=True,
    )
    # add zero padding to 20 (max length) here
    tgt_stress = t3['input_ids'].to(device)
    attn_str = t3['attention_mask'].to(device)
    assert tgt_stress.dim() == 2
    pad_bit = 20 - tgt_stress.shape[1]
    tgt_stress = F.pad(tgt_stress, (0, pad_bit, 0, 0), value=1).to(device)
    attn_str = F.pad(attn_str, (0, pad_bit, 0, 0), value=1).to(device)

    # Concat length and stress constraints with encoder input ids
    input_ids = torch.cat((tgt_lens, tgt_stress, input_ids), dim=1).to(device)
    attention_mask = torch.cat((attn_len, attn_str, attention_mask), dim=1).to(device)

    # Prepare decoder input ids, put rhyme info here
    decoder_input_ids = torch.zeros(size=(3,2), dtype=torch.long).to(device)
    decoder_input_ids[:, 0] = tgt_rhymes.squeeze()
    decoder_input_ids[:, 1] = 2  # set the 3rd col to decoder_start_token_id

    # Generate translation
    generated_tokens = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        num_beams=5,
        max_length=36,
        forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
    )

    # decode to text
    for line in generated_tokens:
        print([tokenizer.decode(i) for i in line])
    print('----- Output -----')
    for line in tokenizer.batch_decode(generated_tokens, skip_special_tokens=True):
        print(line[::-1])
    print('------------------')
    # 现在只剩下一首歌为你留下
    # 离开这城市的街道吧
    # 只留给我一个吻吧



if __name__ == '__main__':
    _main()
