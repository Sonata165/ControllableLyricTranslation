# Controllable Lyric Translation
This is the official implementation of the paper [Songs Across Borders: Singable and Controllable Neural Lyric Translation](https://arxiv.org/abs/2305.16816) (ACL 2023).
Through explicit control using prompt-based fine-tuning, we achieve effective singability quality gain in the translation output. Please check out 
our [translation demo here](https://www.oulongshen.xyz/lyric_translation).
This implementation is in the direction of English-to-Chinese.

The repository contains:
- Link to dataset / model checkpoints / paper / demo

- Code for training / inference / evaluation

## Step-by-step Instructions
### Environment
    # Note: the environment is only tested on Linux (with CUDA) 

    # Create conda env
    conda create -n lyric_trans python=3.9
    
    # Install PyTorch
    pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    
    # Install transformers
    cd transformers
    pip install -e .
    pip install setuptools==58.2.0 # a bug fix for Transformers

    # Other packages
    pip install -r requirements.txt

### Play with the model

**Note**: Due to historical reasons, variables related word boundaries in the code
are sometimes referred to with "str" instead of "bdr".

BartFintune/playground.py is a brief demo of using the model for translation. Run it with
    
    cd BartFinetune
    python playground.py

There are some pre-defined source sentences and corresponding constraints in the script. 
After running it, translations will be generated for the lyrics under these constraints.

### Training
#### Prepare dataset
Find the dataset line at the "Resources" section. If you agree with the terms of this [License](https://huggingface.co/datasets/LongshenOu/lyric-trans-en2zh-data/tree/main),
proceed the download by clicking the download button of the "datasets.zip" file. Unzip it, and put the "datasets" folder under 
the Dataset directory.

#### Run the script
1. First, finetune the mBART model with the back-translated monolingual dataset:
   
       cd BartFinetune
       sh train.sh hparams/with_bt.sh

    During training, logs and checkpoints will be saved to the "results" folder of the root directory.

2. Then, continue fine-tuning with the parallel dataset:

       (under BartFinetune director)
       sh train.sh hparams/with_parallel.sh

### Inference
#### Inference for the test set
After either of the step 1 or 2 of training, you can test the resulting model by
    
    sh infer.sh hparams/with_bt.sh          # After step 1
    sh infer.sh hparams/with_paralle.sh     # After step 2

Outputs and evaluation results will be generated at corresponding folders under the "results" directory.

#### Inference for full songs
After the training step 2, you can use below commands to call the model to generate trans

    cd Dataset/data_sources/Real
    python prepare_dataset.py

This script will process the full song for subjective evaluation into format that are compatible 
with model inference, save into "Dataset/datasets/real" directory. Then,
   
    cd BartFinetune
    sh infer_rec.sh hparams/with_parallel.sh

This command will generate rhyme distribution for each paragraph, and use the 1st-ranked rhyme as 
the rhyme constraints for inference. Results will be saved to "results" directory.


## Resources
- Dataset: [[huggingface](https://huggingface.co/datasets/LongshenOu/lyric-trans-en2zh-data/tree/main)]
- Model: [[huggingface](https://huggingface.co/LongshenOu/lyric-trans-en2zh)]
- Paper: [[ArXiv Ver](https://arxiv.org/abs/2305.16816)] [[ACL Anthology](https://aclanthology.org/2023.acl-long.27/)]
- Demo & subjective evaluation outputs: [[Demo](https://www.oulongshen.xyz/lyric_translation)]

**Note**: The model and dataset is for research purpose only, hence are shared under license [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
**Commercial usage is strictly prohibited**. When using our model or dataset, please cite our ACL 2023 paper following the below section.

## Citation
Our ACL Paper:

    @inproceedings{ou-etal-2023-songs,
        title = "Songs Across Borders: Singable and Controllable Neural Lyric Translation",
        author = "Ou, Longshen  and
          Ma, Xichu  and
          Kan, Min-Yen  and
          Wang, Ye",
        booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.acl-long.27",
        pages = "447--467",
        abstract = "The development of general-domain neural machine translation (NMT) methods has advanced significantly in recent years, but the lack of naturalness and musical constraints in the outputs makes them unable to produce singable lyric translations. This paper bridges the singability quality gap by formalizing lyric translation into a constrained translation problem, converting theoretical guidance and practical techniques from translatology literature to prompt-driven NMT approaches, exploring better adaptation methods, and instantiating them to an English-Chinese lyric translation system. Our model achieves 99.85{\%}, 99.00{\%}, and 95.52{\%} on length accuracy, rhyme accuracy, and word boundary recall. In our subjective evaluation, our model shows a 75{\%} relative enhancement on overall quality, compared against naive fine-tuning (Code available at https://github.com/Sonata165/ControllableLyricTranslation).",
    }

mBART paper:

    @article{tang2020multilingual,
        title={Multilingual Translation with Extensible Multilingual Pretraining and Finetuning},
        author={Yuqing Tang and Chau Tran and Xian Li and Peng-Jen Chen and Naman Goyal and Vishrav Chaudhary and Jiatao Gu and Angela Fan},
        year={2020},
        eprint={2008.00401},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }