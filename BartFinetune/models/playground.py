import torch
from transformers import (MBartForConditionalGeneration,
                          MBartModel,
                          MBartConfig,
                          MBart50TokenizerFast,
                          BeamScorer,
                          LogitsProcessorList,
                          StoppingCriteriaList, BertTokenizer)
from transformers.models.mbart50.tokenization_mbart50 import MBart50Tokenizer
from BartFinetune.utils.utils import *


def _main():
    pass


def play_mbart():
    mbart, info = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt",
                                                                output_loading_info=True)
    print(mbart.config)

    # play_mbart_inference(mbart)
    play_mbart_training(mbart)


def play_mbart_inference(model):
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    device = 'cuda'
    model.to(device)
    print('load complete')
    text = [
        "There's only one song left for you",
        'Get me off the streets of this city',
        'You only left one kiss for me',
        "You're laughing so brightly",
        'Keeps me in my head for the rest of my life',
        "How come I'm still stuck and you're long gone",
        "You're laughing so brightly"
    ]
    # model.to(device)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = 'zh_CN'
    encoded_input = tokenizer(text, return_tensors="pt", padding=True).to(
        device)  # shape: [bs, max_input_seq_len]  [7, 15]
    desired_length = torch.tensor([6, 5, 4, 5, 6, 7, 3], dtype=torch.int64, device=device).unsqueeze(1)
    generated_tokens = model.generate(**encoded_input,
                                      max_length=48,
                                      forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"],
                                      desired_length=desired_length)  # zh_CN, en_XX
    # print(tokenizer.convert_ids_to_tokens(2))
    for line in generated_tokens:
        print([tokenizer.decode(i) for i in line])
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


def play_mbart_training(model):
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    device = 'cuda'
    model.to(device)
    print('load complete')
    text = [
        "There's only one song left for you",
        # 'Get me off the streets of this city',
        # 'You only left one kiss for me',
        # "You're laughing so brightly",
        # 'Keeps me in my head for the rest of my life',
        # "How come I'm still stuck and you're long gone",
        # "You're laughing so brightly"
    ]
    # model.to(device)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = 'zh_CN'
    encoded_input = tokenizer(text, return_tensors="pt", padding=True).to(
        device)  # shape: [bs, max_input_seq_len]  [7, 15]
    desired_length = torch.tensor([6, 5, 4, 5, 6, 7, 3], dtype=torch.int64, device=device).unsqueeze(1)
    generated_tokens = model.generate(
        **encoded_input,
        max_length=48,
        # decoder_start_token_id=tokenizer.lang_code_to_id["zh_CN"],
        forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"],
    )  # zh_CN, en_XX
    # desired_length = desired_length
    # print(tokenizer.convert_ids_to_tokens(2))
    for line in generated_tokens:
        print([tokenizer.decode(i) for i in line])
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    # NOTICE: decoder_start_token_id shouldn't be set for mbart generation


if __name__ == '__main__':
    _main()
