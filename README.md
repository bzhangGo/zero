# Zero 
A neural machine translation system implemented by python (2.7) + tensorflow.

A major update will be available soon!

> The project is still under development, and not fully tested. So there may
be unknown bugs.


## Motivation
- General Framework that can deal with NLP-related tasks.
- Make system as clean and simple as possible, remove other unnecessary modules

## Features
Current implementation contains the following features:
1. Data Reading

    Why it matters? Because when your data pre-processing is very complex and 
your model is relatively small and fast, you will find that your GPU is waiting
for your CPU. This is bad, as it can drop your GPU utilization, and just 
waste your time.

    Hopefully, we can solve this problem with muti-threading programming, where
when your model is running on GPU, another thread can continually fetch data
into the CPU. As a result, there would be always available data batches for 
your GPU.

    Unfortunately, python is notorious for its multi-thread programming due to
the GIL mechanism. `In fact, I can not solve this problem` currently. I just
used the multi-thread pool, but it does not work as I expect.

2. Multi-GPU

    Splitting dataset and model into different GPU cards identified by users.

3. Multi-Cycle
    
    When only one GPU card is available, and you want to train with large batch,
a solution would be cyclically train your model sequentially on different
data batches, and collect the gradient then update your model after it.

4. Online decoding with multi-gpu

    Implement the batched-based beam search algorithm, and dealing with multi-GPU
decoding.

5. Supported Models

    * RNNSearch: support LSTM, GRU, SRU and [ATR](https://github.com/bzhangGo/ATR) models.
    * CAEncoder: the context-aware recurrent encoder, see [the paper](https://ieeexplore.ieee.org/document/8031316)
        and the original [source code](https://github.com/DeepLearnXMU/CAEncoder-NMT) (in Theano).


## How to use it?

Below is a rough procedure for WMT14 En-De translation tasks.

1. Prepare your training, development and test data. 

    For example, you can download the preprocessed WMT14 En-De dataset from [Stanford NMT](https://nlp.stanford.edu/projects/nmt/)
    * The training file: [train.en](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en),
                         [train.de](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de)
    * The development file: [newstest12.en](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en),
                            [newstest12.de](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de),
                            [newstest13.en](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en),
                            [newstest13.de](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de)
    
      Then, concate the `newstest12.en` and `newstest13.en` into `dev.en` using command like `cat newstest12.en newstest13.en > dev.en`.
      The same is for German language: `cat newstest12.de newstest13.de > dev.de`
    * The test file: [newstest14.en](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en),
                     [newstest14.de](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de)

2. Preprocess your dataset. 
    
    Generally, you can process your dataset with a standard pipeline as given in the WMT official site. 
    Full content can be found at the preprocessed [datasets](http://data.statmt.org/wmt17/translation-task/preprocessed/).
    See the `prepare.sh` for more details.
       
    In our case, for WMT14 En-De translation, however, the dataset has already been pre-processeed. 
    So, you can do nothing at this stage.
    
    *For some languages, such as Chinese, you need to perform word segmentation (Chinese-version Tokenize) first.
    You can find more information about segmentation [here](https://nlp.stanford.edu/software/segmenter.shtml)*

3. Optional but strongly suggested, Perform BPE decoding

    - download the [subword project](https://github.com/rsennrich/subword-nmt)
    - learn the subword model:
        ```
        python subword-nmt/learn_joint_bpe_and_vocab.py --input train.en train.de -s 32000 -o bpe32k --write-vocabulary vocab.en vocab.de
        ```
      Notice that the 32000 indicates 32k pieces, or you can simply understand it as your vocabulary size.
    - Apply the subword model to all your datasets.
      - To training data
        ```
        python subword-nmt/apply_bpe.py --vocabulary vocab.en --vocabulary-threshold 50 -c bpe32k < train.en > train.32k.en
        python subword-nmt/apply_bpe.py --vocabulary vocab.de --vocabulary-threshold 50 -c bpe32k < train.de > train.32k.de
        ```
      - To dev data
        ```
        python subword-nmt/apply_bpe.py --vocabulary vocab.en --vocabulary-threshold 50 -c bpe32k < dev.en > dev.32k.en
        python subword-nmt/apply_bpe.py --vocabulary vocab.de --vocabulary-threshold 50 -c bpe32k < dev.de > dev.32k.de
        ```
        Notice that you do not have to apply bpe to the `dev.de`, but we use it in our model.
      - To test data
        ```        
        python subword-nmt/apply_bpe.py --vocabulary vocab.en --vocabulary-threshold 50 -c bpe32k < newstest14.en > newstest14.32k.en
        ```

4. Extract Vocabulary

    You still need to prepare the vocabulary using our code, because there are some special symbols in our vocabulary.
    - download our project:
    ```git clone https://github.com/bzhangGo/zero.git```
    - Run the code as follows:
    ```
    python zero/vocab.py train.en vocab.en
    python zero/vocab.py train.de vocab.de
    ```
    Roughly, the vocabulary size would be 32000, more or less.

5. Training your model.

    train your model with the following settings:
    ```
    data_dir=the preprocessed data directory
    python zero/config.py --mode train --parameters=hidden_size=1024,embed_size=512,\
    dropout=0.1,label_smooth=0.1,\
    max_len=80,batch_size=80,eval_batch_size=240,\
    token_size=3000,batch_or_token='token',\
    model_name="rnnsearch",buffer_size=3200,\
    clip_grad_norm=5.0,\
    lrate=5e-4,\
    epoches=10,\
    update_cycle=1,\
    gpus=[3],\
    disp_freq=100,\
    eval_freq=10000,\
    sample_freq=1000,\
    checkpoints=5,\
    caencoder=True,\
    cell='atr',\
    max_training_steps=100000000,\
    nthreads=8,\
    swap_memory=True,\
    layer_norm=True,\
    max_queue_size=100,\
    random_seed=1234,\
    src_vocab_file="$data_dir/vocab.en",\
    tgt_vocab_file="$data_dir/vocab.de",\
    src_train_file="$data_dir/train.32k.en.shuf",\
    tgt_train_file="$data_dir/train.32k.de.shuf",\
    src_dev_file="$data_dir/dev.32k.en",\
    tgt_dev_file="$data_dir/dev.32k.de",\
    src_test_file="",\
    tgt_test_file="",\
    output_dir="train",\
    test_output=""
    ```
    Model would be saved into directory `train`

6. Testing your model

    - Average your checkpoints which can give you better results.
    ```
    python zero/scripts/checkpoint_averaging.py --checkpoints 5 --output avg --path ../train --gpu 0
    ```
    - The test your model with the following code
    ```
    data_dir=the preprocessed data directory
    python zero/config.py --mode test --parameters=hidden_size=1024,embed_size=512,\
    dropout=0.1,label_smooth=0.1,\
    max_len=80,batch_size=80,eval_batch_size=240,\
    token_size=3000,batch_or_token='token',\
    model_name="rnnsearch",buffer_size=3200,\
    clip_grad_norm=5.0,\
    lrate=5e-4,\
    epoches=10,\
    update_cycle=1,\
    gpus=[3],\
    disp_freq=100,\
    eval_freq=10000,\
    sample_freq=1000,\
    checkpoints=5,\
    caencoder=True,\
    cell='atr',\
    max_training_steps=100000000,\
    nthreads=8,\
    swap_memory=True,\
    layer_norm=True,\
    max_queue_size=100,\
    random_seed=1234,\
    src_vocab_file="$data_dir/vocab.en",\
    tgt_vocab_file="$data_dir/vocab.de",\
    src_train_file="$data_dir/train.32k.en.shuf",\
    tgt_train_file="$data_dir/train.32k.de.shuf",\
    src_dev_file="$data_dir/dev.32k.en",\
    tgt_dev_file="$data_dir/dev.32k.de",\
    src_test_file="$data_dir/newstest14.32k.en",\
    tgt_test_file="$data_dir/newstest14.de",\
    output_dir="avg",\
    test_output="newstest14.trans.bpe"
    ``` 
    The final translation will be dumped into `newstest14.trans.bpe`. 
    
    You need remove the BPE splitter as follows: `sed -r 's/(@@ )|(@@ ?$)//g' < newstest14.trans.bpe > newstest14.trans.txt`
    
    Then evaluate the BLEU score using [multi-bleu.perl](https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl):
    ```perl multi-bleu.perl $data_dir/newstest14.de < newstest14.trans.txt```
    
    > Notice that the official evaluation has stated clearly that researchers should not use the multi-bleu.perl anymore, because it
    heavily relies on the tokenization schema. In fact, tokenization could have a strong influence to the
    final BLEU score, particularly when the aggressive mode is used. However, in current stage, multi-bleu.perl is still
    the most-widely used evaluation script ~~

And That's it!
## Citation

If you use the source code, please consider citing the follow paper:
```
@InProceedings{D18-1459,
  author = 	"Zhang, Biao
		and Xiong, Deyi
		and su, jinsong
		and Lin, Qian
		and Zhang, Huiji",
  title = 	"Simplifying Neural Machine Translation with Addition-Subtraction Twin-Gated Recurrent Networks",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"4273--4283",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1459"
}
```

If you are interested in the CAEncoder model, please consider citing our TASLP paper:
```
@article{Zhang:2017:CRE:3180104.3180106,
 author = {Zhang, Biao and Xiong, Deyi and Su, Jinsong and Duan, Hong},
 title = {A Context-Aware Recurrent Encoder for Neural Machine Translation},
 journal = {IEEE/ACM Trans. Audio, Speech and Lang. Proc.},
 issue_date = {December 2017},
 volume = {25},
 number = {12},
 month = dec,
 year = {2017},
 issn = {2329-9290},
 pages = {2424--2432},
 numpages = {9},
 url = {https://doi.org/10.1109/TASLP.2017.2751420},
 doi = {10.1109/TASLP.2017.2751420},
 acmid = {3180106},
 publisher = {IEEE Press},
 address = {Piscataway, NJ, USA},
}
```

## TODO:
* Multi-threading Data Processing
* Clean and reformat the code

## Reference
When developing this repository, I referred to the following projects:

* [Nematus](https://github.com/EdinburghNLP/nematus)
* [THUMT](https://github.com/thumt/THUMT)
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
* [Keras](https://github.com/keras-team/keras)

## Contact
For any questions or suggestions, please feel free to contact [Biao Zhang](mailto:B.Zhang@ed.ac.uk)
