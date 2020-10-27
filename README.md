# Fast Interleaved Bidirectional Sequence Generation

This code base implements the interleaved bidirectional decoder (IBDecoder) introduced in our WMT2020 paper.

We integrate bidirectional generation and semi-autoregressive decoder, achieving comparable translation performance on 
various sequence-to-sequence tasks (machine translation and document summarization) with a decoding speedup of \~2x (very
marginal training time loss). By 
allowing the decoder to produce multiple target tokens per step (>2), our model achieves speedups to 4x\~11x across 
different tasks at the cost of <1 BLEU or <0.5ROUGE (on average).


The [usage](https://github.com/bzhangGo/zero/tree/master/docs/usage) of this codebase is consistent with the master 
version of [zero](https://github.com/bzhangGo/zero), without any
specific requirement for data preprocessing. Below offers an example for training on WMT14 EnDe (more information about
the basic usage is given [here](https://github.com/bzhangGo/zero/tree/master/docs/usage)):

* Training example script
  ```
    #!/bin/bash
    #!
    
    data=path-to-preprocessed-datadir/wmt14-ende/
    
    python zero/run.py --mode train --parameters=hidden_size=512,embed_size=512,filter_size=2048,\
    dropout=0.1,label_smooth=0.1,attention_dropout=0.1,\
    max_len=256,batch_size=80,eval_batch_size=32,\
    token_size=5000,batch_or_token='token',\
    initializer="uniform_unit_scaling",initializer_gain=1.,\
    model_name="transformer_ibdecoder",scope_name="transformer",buffer_size=600000,\
    ibdecoder_factor=1,\
    clip_grad_norm=0.0,\
    num_heads=8,\
    lrate=1.0,\
    process_num=3,\
    estop_patience=100,\
    num_encoder_layer=6,\
    num_decoder_layer=6,\
    warmup_steps=4000,\
    lrate_strategy="noam",\
    epoches=5000,\
    update_cycle=5,\
    gpus=[0],\
    disp_freq=1,\
    eval_freq=5000,\
    sample_freq=1000,\
    checkpoints=5,\
    max_training_steps=300000,\
    nthreads=8,\
    beta1=0.9,\
    beta2=0.98,\
    epsilon=1e-8,\
    swap_memory=True,\
    layer_norm=True,\
    random_seed=1234,\
    src_vocab_file="$data/vocab.zero.en",\
    tgt_vocab_file="$data/vocab.zero.de",\
    src_train_file="$data/train.32k.en.shuf",\
    tgt_train_file="$data/train.32k.de.shuf",\
    src_dev_file="$data/dev.32k.en",\
    tgt_dev_file="$data/dev.32k.de",\
    src_test_file="$data/newstest2014.32k.en",\
    tgt_test_file="$data/newstest2014.de",\
    output_dir="train",\
    test_output="trans.txt",\
    default_dtype="float32",\
    dtype_epsilon=1e-8,\
    dtype_inf=1e8,\
    loss_scale=1.0,\  
  ```
* Decoding example script (average the last 5 checkpoints into avg directory)
  ```
    #!/bin/bash
    #!
    
    data=path-to-preprocessed-datadir/wmt14-ende/
    
    python zero/run.py --mode test --parameters=hidden_size=512,embed_size=512,filter_size=2048,\
    dropout=0.1,label_smooth=0.1,attention_dropout=0.1,\
    max_len=256,batch_size=80,eval_batch_size=32,\
    token_size=5000,batch_or_token='token',\
    initializer="uniform_unit_scaling",initializer_gain=1.,\
    model_name="transformer_ibdecoder",scope_name="transformer",buffer_size=600000,\
    ibdecoder_factor=1,\
    clip_grad_norm=0.0,\
    num_heads=8,\
    lrate=1.0,\
    process_num=3,\
    estop_patience=100,\
    num_encoder_layer=6,\
    num_decoder_layer=6,\
    warmup_steps=4000,\
    lrate_strategy="noam",\
    epoches=5000,\
    update_cycle=5,\
    gpus=[0],\
    disp_freq=1,\
    eval_freq=5000,\
    sample_freq=1000,\
    checkpoints=5,\
    max_training_steps=300000,\
    nthreads=8,\
    beta1=0.9,\
    beta2=0.98,\
    epsilon=1e-8,\
    swap_memory=True,\
    layer_norm=True,\
    random_seed=1234,\
    src_vocab_file="$data/vocab.zero.en",\
    tgt_vocab_file="$data/vocab.zero.de",\
    src_train_file="$data/train.32k.en.shuf",\
    tgt_train_file="$data/train.32k.de.shuf",\
    src_dev_file="$data/dev.32k.en",\
    tgt_dev_file="$data/dev.32k.de",\
    src_test_file="$data/newstest2014.32k.en",\
    tgt_test_file="$data/newstest2014.de",\
    output_dir="avg",\
    test_output="trans.txt",\
    default_dtype="float32",\
    dtype_epsilon=1e-8,\
    dtype_inf=1e8,\
    loss_scale=1.0,\  
  ```
Two important hyperparameters: `model_name="transformer_ibdecoder"` and `ibdecoder_factor=1`. The `ibdecoder_factor` 
relates with the number of target tokens produced per decoding step. More concretely, each step generates 
`2 * ibdecoder_factor` tokens, where the factor `2` comes from the bidirectional generation.

* Please go to [here](https://github.com/bzhangGo/zero/blob/master/docs/interleaved_bidirectional_transformer) for more experimental results, including trained models, translations and preprocessed corpus.

For any questions or suggestions, please feel free to contact [Biao Zhang](mailto:B.Zhang@ed.ac.uk)