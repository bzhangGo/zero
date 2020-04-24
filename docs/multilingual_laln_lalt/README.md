## Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation

[paper](#)

We study massively multilingual translation in this paper, 
with a goal of improving zero-shot translation.

* On massively multilingual translation, we collected OPUS-100, a 100-language multilingual dataset from OPUS.
We release this dataset to help further study on this direction.

  - [the OPUS-100 dataset](https://github.com/EdinburghNLP/opus-100-corpus)

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
   
### Code

We implement the model in [transformer_l0drop](../../models/transformer_l0drop.py) 
and [l0norm](../../modules/l0norm.py)

### Training

It's possible to train Transformer with L0Drop from scratch by setting proper schedulers for `\lambda`, 
a hyperparameter loosely controling the sparsity rate of L0Drop. Unfortunately, the optimal scheduler is
data&task-dependent.

We suggest first pre-train a normal Transformer model, and then finetune the Transfomer+L0Drop. This could
save a lot of efforts.

* Step 1. train a normal Transformer model as described [here](../../docs/usage/README.md). Below is 
an example on WMT14 En-De for reference:
```
data_dir=the preprocessed data diretory
zero=the path of this code base
python $zero/run.py --mode train --parameters=hidden_size=512,embed_size=512,filter_size=2048,\
dropout=0.1,label_smooth=0.1,attention_dropout=0.1,\
max_len=256,batch_size=80,eval_batch_size=32,\
token_size=6250,batch_or_token='token',\
initializer="uniform_unit_scaling",initializer_gain=1.,\
model_name="transformer",scope_name="transformer",buffer_size=60000,\
clip_grad_norm=0.0,\
num_heads=8,\
lrate=1.0,\
process_num=3,\
num_encoder_layer=6,\
num_decoder_layer=6,\
warmup_steps=4000,\
lrate_strategy="noam",\
epoches=5000,\
update_cycle=4,\
gpus=[0],\
disp_freq=1,\
eval_freq=5000,\
sample_freq=1000,\
checkpoints=5,\
max_training_steps=300000,\
beta1=0.9,\
beta2=0.98,\
epsilon=1e-8,\
random_seed=1234,\
src_vocab_file="$data_dir/vocab.zero.en",\
tgt_vocab_file="$data_dir/vocab.zero.de",\
src_train_file="$data_dir/train.32k.en.shuf",\
tgt_train_file="$data_dir/train.32k.de.shuf",\
src_dev_file="$data_dir/dev.32k.en",\
tgt_dev_file="$data_dir/dev.32k.de",\
src_test_file="$data_dir/newstest2014.32k.en",\
tgt_test_file="$data_dir/newstest2014.de",\
output_dir="train"
```

* Step 2. finetune L0Drop using the following command:
```
data_dir=the preprocessed data directory
zero=the path of this code base
python $zero/run.py --mode train --parameters=\
l0_norm_reg_scalar=0.2,\
l0_norm_warm_up=False,\
model_name="transformer_l0drop",scope_name="transformer",\
pretrained_model="path-to-pretrained-transformer",\
max_training_steps=320000,\
src_vocab_file="$data_dir/vocab.zero.en",\
tgt_vocab_file="$data_dir/vocab.zero.de",\
src_train_file="$data_dir/train.32k.en.shuf",\
tgt_train_file="$data_dir/train.32k.de.shuf",\
src_dev_file="$data_dir/dev.32k.en",\
tgt_dev_file="$data_dir/dev.32k.de",\
src_test_file="$data_dir/newstest2014.32k.en",\
tgt_test_file="$data_dir/newstest2014.de",\
output_dir="train"
```
where `l0_norm_reg_scalar` is the `\lambda`, and `0.2` is a nice hyperparameter in our experiments.

### Evaluation

The evaluation follows the same procedure as the baseline Transformer.

### Citation

Please consider cite our paper as follows:
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