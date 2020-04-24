## Questions
1. What's the effective batch size, training and decoding
    
    * When using `token-based` training (batch_or_token=token), the effective token number equals `number_gpus * update_cycles * token_siz`.
    * When using `batch-based` training (batch_or_token=batch), the effective batch size equals `number_gpus * update_cycles * batch_size`
    * At decoding phrase, we only use batch-based decoding with size of `eval_batch_size`.

2. What's the difference between `model_name` and `scope_name`
  
    The `model_name` means which model you want to train. The model name should be a registered model, which is
    under the folder `models`. The `scope_name` denotes the scope name in tensorflow for each model weights or variables.
    
    For example, when you want to train a Transformer model, you should set `model_name=transformer`. But you can use
    any valid scope name as you want, such as transformer, nmtmodel, transformer_exp1, .etc.

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

    BPE algorithm is the most popular and currently standard way to handle rare words, or OOVs. It iteratively
    merges the most frequent patterns until the maximum merging number is reached. It splits rare words into
    `sub-words`, such as `Bloom` => `Blo@@ om`. Another benefit of BPE is that you can control the size of vocabulary.

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
    python zero/run.py --mode train --parameters=hidden_size=1024,embed_size=512,\
    dropout=0.1,label_smooth=0.1,\
    max_len=80,batch_size=80,eval_batch_size=240,\
    token_size=3000,batch_or_token='token',\
    model_name="rnnsearch",scope_name="rnnsearch",buffer_size=3200,\
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
    - Then test your model with the following code
    ```
    data_dir=the preprocessed data directory
    python zero/run.py --mode test --parameters=hidden_size=1024,embed_size=512,\
    dropout=0.1,label_smooth=0.1,\
    max_len=80,batch_size=80,eval_batch_size=240,\
    token_size=3000,batch_or_token='token',\
    model_name="rnnsearch",scope_name="rnnsearch",buffer_size=3200,\
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
   
7. Command line or Seperate configuration file

    In case you dislike the long command line style, you can convert the parameters into a 
    separate `config.py`. For the training example, you can convert the running comment into follows:
    ```
    python zero/run.py --mode train --config config.py
    ```
    where the `config.py` has the following structure:
    ```
    dict(
        hidden_size=1024,
        embed_size=512,
        dropout=0.1,
        label_smooth=0.1,
        max_len=80,
        batch_size=80,
        eval_batch_size=240,
        token_size=3000,
        batch_or_token='token',
        model_name="rnnsearch",
        scope_name="rnnsearch",
        buffer_size=3200,
        clip_grad_norm=5.0,
        lrate=5e-4,
        epoches=10,
        update_cycle=1,
        gpus=[3],
        disp_freq=100,
        eval_freq=10000,
        sample_freq=1000,
        checkpoints=5,
        caencoder=True,
        cell='atr',
        max_training_steps=100000000,
        nthreads=8,
        swap_memory=True,
        layer_norm=True,
        max_queue_size=100,
        random_seed=1234,
        src_vocab_file="$data_dir/vocab.en",
        tgt_vocab_file="$data_dir/vocab.de",
        src_train_file="$data_dir/train.32k.en.shuf",
        tgt_train_file="$data_dir/train.32k.de.shuf",
        src_dev_file="$data_dir/dev.32k.en",
        tgt_dev_file="$data_dir/dev.32k.de",
        src_test_file="",
        tgt_test_file="",
        output_dir="train",
        test_output="",
    )
    ```


And That's it!
