## Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation

- [paper link](http://arxiv.org/abs/2004.11867)
- source code is given in the [multilingual_laln_lalt branch](https://github.com/bzhangGo/zero/tree/multilingual_laln_lalt)

We study massively multilingual translation in this paper, 
with a goal of improving zero-shot translation.

* On massively multilingual translation, we collected OPUS-100, a 100-language multilingual dataset from OPUS.
We release this dataset to help further study on this direction.

  - [the OPUS-100 dataset, Edinburgh](https://github.com/EdinburghNLP/opus-100-corpus); or
  - [the OPUS website, OPUS](http://opus.nlpl.eu/opus-100.php)

* Multilingual model requires stronger capacity to support various language pairs. We propose to 
devise language-specific components: *1. language-aware layer normalization and 2. language-aware linear mapping* 
and deepen the NMT architecture: *deep Transformer*.

  - Improving capacity clearly improves multilingual translation, and benefits specifically low-resource translation.

* On zero-shot translation, we observe that traditional multilingual model 
often suffer from `off-target translation` [1,2].

    The examples below are for French->German translation, while multilingual models are trained on English<->French and 
    English<->German dataset:
    
    |           |                                                                                              |
    |-----------|----------------------------------------------------------------------------------------------|
    | Source    | Jusqu'à ce qu'on trouve le moment clé, celui où tu as su que tu l'aimais.                    |
    | Reference | Bis wir den unverkennbaren Moment gefunden haben, den Moment, wo du wusstest, du liebst ihn. |
    | Zero-Shot | `Jusqu'à ce qu'on trouve le moment clé, celui où tu as su que tu l'aimais.`                    |
    |-----------|----------------------------------------------------------------------------------------------|
    |   Source  | Les États membres ont été consultés et ont approuvé cette proposition.                       |
    | Reference | Die Mitgliedstaaten wurden konsultiert und sprachen sich für diesen Vorschlag aus.           |
    | Zero-Shot | Les `Member States have been` consultedés `and have approved this proposal.`              |
    
    Multilingual model tends to copy the source input or translate into English on zero-shot directions. To support
    enormous zero-shot directions (~10000), we introduce *random online backtranslation* which operates backtranslation
    by randomly pick an intermediate language.
    
  - Stronger capacity fails to solve the off-target translation issue.
  - Random online backtranslation works and scales very well to massively zero-shot translations.
  
### Pretrained Multilingual Models (many-to-many)

Training a multilingual model is time-consuming. We spent several weeks for one model with 4 P100 GPUs. We offer our
[pretrained models](http://data.statmt.org/bzhang/acl2020_multilingual/) below to ease the study of massively multilingual translation:

Model Description | Download
---|---
`Base + 6 layer` | [download](http://data.statmt.org/bzhang/acl2020_multilingual/Base-L6.tar.gz)
`Base + 6 layer + RoBT` | [download](http://data.statmt.org/bzhang/acl2020_multilingual/Base-L6-RoBT.tar.gz)
`Ours + 6 layer` | [download](http://data.statmt.org/bzhang/acl2020_multilingual/Ours-L6.tar.gz)
`Ours + 6 layer + RoBT` | [download](http://data.statmt.org/bzhang/acl2020_multilingual/Ours-L6-RoBT.tar.gz)
`Ours + 12 layer` | [download](http://data.statmt.org/bzhang/acl2020_multilingual/Ours-L12.tar.gz)
`Ours + 12 layer + RoBT` | [download](http://data.statmt.org/bzhang/acl2020_multilingual/Ours-L12-RoBT.tar.gz)
`Ours + 24 layer` | [download](http://data.statmt.org/bzhang/acl2020_multilingual/Ours-L24.tar.gz)
`Ours + 24 layer + RoBT` | [download](http://data.statmt.org/bzhang/acl2020_multilingual/Ours-L24-RoBT.tar.gz)

- `Base`: transformer. Notice that to use these base models, you need the [master version of zero](https://github.com/bzhangGo/zero)
rather than the multilingual one.

- `Ours`: transformer + [merged attention](https://github.com/bzhangGo/zero/blob/master/docs/depth_scale_init_and_merged_attention) + LaLn + LaLT

- Some different preprocessing used in our experiments (**not suggested!**)
    * We adopted "--character_coverage 0.9995 --input_sentence_size=10000000" for 
    sentencepiece model training. This coverage rate results in messy code for some languages like Chinese.
    * We cut the length of test sentence to 100 during evaluation to avoid memory issue.

- The sentencepiece model and vocabulary used in our experiments are given [here](http://data.statmt.org/bzhang/acl2020_multilingual/submodels.tar.gz).
We also provide an example [evaluation script](http://data.statmt.org/bzhang/acl2020_multilingual/example_evaluation.sh).

#### Going-Through Example

To ease others quickly using the pretrained models and our source code, below we show a simple script. *Note it's used
to decoding the opus test set with the above pretrained models.*

```
#! /bin/bash

# below is an example showing how to use our pretrained models to perform evaluation


# download the zero-toolkit 
git clone --branch multilingual_laln_lalt https://github.com/bzhangGo/zero.git

# download opus-100 to opus-100-corpus
# slow
# bash zero/scripts/data/download_opus100.sh zero/ opus-100-corpus
# faster, note this downloading might not be stable, if you cannot download the data, try multiple times or other alternative methods (like the above
 slow one)
wget https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz
tar xfvz opus-100-corpus-v1.0.tar.gz

# adjust document structure (to meet requirement of example_evaluation.sh)
if [[ ! -d opus-100-corpus ]]; then
    echo 'please downloading the opus-100-corpus'
    exit 1
fi

cd opus-100-corpus/
ln -s v1.0/* .
cd ..

# download our preprocessed subword models
wget http://data.statmt.org/bzhang/acl2020_multilingual/submodels.tar.gz
tar xfvz submodels.tar.gz

# download Ours-L6-RoBT model for example
wget http://data.statmt.org/bzhang/acl2020_multilingual/Ours-L6-RoBT.tar.gz 
tar xfvz Ours-L6-RoBT.tar.gz

# download the evaluation script
wget http://data.statmt.org/bzhang/acl2020_multilingual/example_evaluation.sh

# install sacrebleu, sentencepiece if necessary
pip3 install sacrebleu sentencepiece --user
# notice that we use tensorflow, so install tensorflow if necessary
# pip install tensorflow_gpu==1.13.1 --user


# perform decoding
bash example_evaluation.sh
```

 
### Code

We implement models in our paper in a new branch of zero: 
[multilingual_laln_lalt](https://github.com/bzhangGo/zero/tree/multilingual_laln_lalt).

Please go to the new branch for code viewing:
```
random online backtranslation => main.py
language-aware modeling => models/transformer_multilingual
```

### Data preprocessing

We provide a simple script to download and preprocess the OPUS-100
* Step 1: download zero code `git clone --branch multilingual_laln_lalt https://github.com/bzhangGo/zero.git`
    - set code path `zero_path=path-to-the-zero-code/zero`, such as ```zero_path=`pwd`/zero```
* Step 2: download opus dataset `bash $zero_path/scripts/data/download_opus100.sh $zero_path opus-100`
    - Notice, speed is slow!
    - set data path `opus_path=path-to-opus-100/opus-100`, such as ```opus_path=`pwd`/opus-100```
  
  Or instead, using wget: `wget -r -np -nH -R "index.html*" -e robots=off http://data.statmt.org/opus-100-corpus/v1.0/`
  and set data path ```opus_path=`pwd`/opus-100-corpus/v1.0```
* Step 3: preprocess and prepare for one-to-many translation and many-to-many translation
`bash $zero_path/scripts/data/prepare_multilingual_translation.sh $opus_path $zero_path yes preprocessed_data`
    - Requirement
        - python3, with sentencepiece installed
        - python2
    - !! Very very slow! (mainly due to subword processing part)
    - set preprocessed data path `data_path=path-to-preprocessed-opus-100/preprocessed_data`, 
    such as ```data_path=`pwd`/preprocessed_data```
    - Two corpus are automatically generated: `$data_path/one-to-many` and `$data_path/many-to-many`
    - Note that we employ a `checkpoint` file to trace the preprocessing, which might not be robust enough for handling
    all exceptional cases. In case you fail, consider to delete this file before reruning the script.
    - By default, we should use all development dataset for model selection. However, OPUS-100 in multilingual setting
    contains ~370k sentences for decoding, which could consume too many time. Instead, we offer another option to only
    include top-`n` sentences in each devset as follows (`n` is set to 100):
    `bash $zero_path/scripts/data/prepare_multilingual_translation.sh $opus_path $zero_path yes preprocessed_data 100`

### Training

We provide an example to show how to train the model, as given in 
[example_training.sh](https://github.com/bzhangGo/zero/blob/multilingual_laln_lalt/scripts/data/example_training.sh).

* remember to set the data_path in example_training.py to `$data_path/one-to-many` for one-to-many translation 
or `$data_path/many-to-many` for many-to-many translation.

### *Finetuning*

For many-to-many translation, we propose `random online backtranslation` to enable thousands of zero-shot translation
directions. We show how to perform finetuning in 
[example_finetuning.sh](https://github.com/bzhangGo/zero/blob/multilingual_laln_lalt/scripts/data/example_finetuning.sh).

### Evaluation

Multilingual evaluation in OPUS-100 include `xx-en`, `en-xx` (only for one-to-many translation), and `xx-xx`.
We show how to do evaluation in 
[example_evaluation.sh](https://github.com/bzhangGo/zero/blob/multilingual_laln_lalt/scripts/data/example_evaluation.sh).

We employ `sacrebleu` for all evaluation.

### More Results

We format our paper by only reporting average BLEU scores. For those interested in concrete results for each 
language pair, we provide more [detailed results](many-to-many-full-results-per-language.md) for many-to-many translation.
We also provide an [excel document](many-to-many.xlsx) for downloading.

### Citation

Please consider cite our paper as follows:
>Biao Zhang; Philip Williams; Ivan Titov; Rico Sennrich (2020). 
Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation. 
In Proceedings of the 2020 Annual Conference of the Association for Computational Linguistics.
```
@inproceedings{zhang2020,
title = "Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation",
author = "Zhang, Biao and
Williams, Philip and
Titov, Ivan and
Sennrich, Rico",
booktitle = "Proceedings of the 2020 Annual Conference of the Association for Computational Linguistics",
month = jul,
year = "2020",
publisher = "Association for Computational Linguistics"
}
```

### References

[1] Naveen Arivazhagan, Ankur Bapna, Orhan Firat, RoeeAharoni, Melvin Johnson, and Wolfgang Macherey.2019a.   The missing ingredient in zero-shot neuralmachine translation.CoRR, abs/1903.07091.

[2] Jiatao  Gu,  Yong  Wang,  Kyunghyun  Cho,  and  Vic-tor O.K. Li. 2019.   Improved zero-shot neural ma-chine translation via ignoring spurious correlations.InProceedings  of  the  57th  Annual  Meeting  of  theAssociation  for  Computational  Linguistics,  pages1258–1268, Florence, Italy. Association for Compu-tational Linguistics.