# Zero 
A neural machine translation system implemented by python + tensorflow.


## Motivation
- General Framework than can deal with NLP-related tasks
- Make system as clean and simple as possible, remove other unnecessary modules

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
For any questions or suggestions, feel free to contact [Biao Zhang](mailto:B.Zhang@ed.ac.uk)