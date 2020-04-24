## Improving Deep Transformer with Depth-Scaled Initialization and Merged Attention, EMNLP2019

[paper link](https://www.aclweb.org/anthology/D19-1083/)

This paper focus on improving Deep Transformer. 
Our empirical observation suggests that simply stacking more Transformer layers makes training divergent.
Rather than resorting to the pre-norm structure which shifts the layer normalization before modeling blocks,
we analyze the reason why a vanilla deep Transformer suffers from poor convergence.

Our evidence shows that it's because of *gradient vanishing* caused by the interaction between residual connection
and layer normalization. We solve this problem by proposing depth-scaled initialization (DS-Init), which decreases 
parameter variance at the initialization stage. DS-Init reduces output variance of residual connections so as to
ease gradient back-propagation through normalization layers. In practice, DS-Init often produces slightly better
translation quality than the pre-norm structure.

We also care about the computational overhead raised by deep models. To settle this issue, we propose the merged
attention network which combines a simplified average attention model and the encoder-decoder attention model on 
the target side. Merged attention model enables the deep Transformer matching the decoding speed of its baseline
with a clear higher BLEU score.


### Model Training
train 12-layer Transformer model with the following settings:
>The model class is: `transformer_fuse`, the merged attention is enabled by giving `fuse_mask` into `dot_attention` function.
```
data_dir=the preprocessed data directory
python run.py --mode train --parameters=hidden_size=512,embed_size=512,filter_size=2048,\
dropout=0.1,label_smooth=0.1,attention_dropout=0.1,\
max_len=256,eval_max_len=256,eval_batch_size=240,\
token_size=6125,batch_or_token='token',\
initializer="uniform_unit_scaling",initializer_gain=1.,\
model_name="transformer_fuse",scope_name="transformer_fuse",buffer_size=600000,\
deep_transformer_init=True,\
clip_grad_norm=0.0,\
num_heads=8,\
lrate=1.0,\
process_num=3,\
estop_patience=100,\
num_encoder_layer=12,\
num_decoder_layer=12,\
warmup_steps=4000,\
lrate_strategy="noam",\
epoches=5000,\
update_cycle=4,\
gpus=[0],\
disp_freq=1,\
eval_freq=5000,\
sample_freq=1000,\
checkpoints=5,\
max_training_steps=200000,\
nthreads=8,\
beta1=0.9,\
beta2=0.98,\
epsilon=1e-8,\
random_seed=1234,\
src_vocab_file="$data_dir/vocab.en",\
tgt_vocab_file="$data_dir/vocab.de",\
src_train_file="$data_dir/train.32k.en.shuf",\
tgt_train_file="$data_dir/train.32k.de.shuf",\
src_dev_file="$data_dir/dev.32k.en",\
tgt_dev_file="$data_dir/dev.32k.de",\
src_test_file="$data_dir/newstest2014.32k.en",\
tgt_test_file="$data_dir/newstest2014.de",\
output_dir="train",\
test_output=""
```
>If you dislike this long command line, you can also write the parameters into a separate config.py file using
command like --config config.py. Data in config.py is a simple `dict` object.

More details can be found [here](../usage/README.md).

Pretrained models can be found [here](http://data.statmt.org/bzhang/emnlp19_deep_transformer/).

### Citation

Please consider cite our paper as follows:
```
@inproceedings{zhang-etal-2019-improving-deep,
    title = "Improving Deep Transformer with Depth-Scaled Initialization and Merged Attention",
    author = "Zhang, Biao  and
      Titov, Ivan  and
      Sennrich, Rico",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1083",
    doi = "10.18653/v1/D19-1083",
    pages = "898--909",
    abstract = "The general trend in NLP is towards increasing model capacity and performance via deeper neural networks. However, simply stacking more layers of the popular Transformer architecture for machine translation results in poor convergence and high computational overhead. Our empirical analysis suggests that convergence is poor due to gradient vanishing caused by the interaction between residual connection and layer normalization. We propose depth-scaled initialization (DS-Init), which decreases parameter variance at the initialization stage, and reduces output variance of residual connections so as to ease gradient back-propagation through normalization layers. To address computational cost, we propose a merged attention sublayer (MAtt) which combines a simplified average-based self-attention sublayer and the encoder-decoder attention sublayer on the decoder side. Results on WMT and IWSLT translation tasks with five translation directions show that deep Transformers with DS-Init and MAtt can substantially outperform their base counterpart in terms of BLEU (+1.1 BLEU on average for 12-layer models), while matching the decoding speed of the baseline model thanks to the efficiency improvements of MAtt. Source code for reproduction will be released soon.",
}
```
