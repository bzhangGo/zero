# Zero 
A neural machine translation system implemented by python2 + tensorflow.

## Features
1. Multi-Process Data Loading/Processing (*Problems Exist*)
2. Multi-GPU Training/Decoding
3. Gradient Aggregation 

## Papers

We associate each paper below with a readme file link. Please click the paper link you are interested for more details.

* [On Sparsifying Encoder Outputs in Sequence-to-Sequence Models](docs/l0drop)
* [Fast Interleaved Bidirectional Sequence Generation, WMT2020](docs/interleaved_bidirectional_transformer)
* [Adaptive Feature Selection for End-to-End Speech Translation, EMNLP2020 Findings](docs/afs_speech_translation)
* [Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation, ACL2020](docs/multilingual_laln_lalt)
* [Improving Deep Transformer with Depth-Scaled Initialization and Merged Attention, EMNLP2019](docs/depth_scale_init_and_merged_attention)

## Supported Models
* RNNSearch: support LSTM, GRU, SRU, [ATR, EMNLP2018](https://github.com/bzhangGo/ATR), and [LRN, ACL2019](https://github.com/bzhangGo/lrn) 
models.
* Deep attention: [Neural Machine Translation with Deep Attention, TPAMI](https://ieeexplore.ieee.org/document/8493282)
* CAEncoder: the context-aware recurrent encoder, see [the paper, TASLP](https://ieeexplore.ieee.org/document/8031316)
    and the original [source code](https://github.com/DeepLearnXMU/CAEncoder-NMT) (in Theano).
* Transformer: [attention is all you need](https://arxiv.org/abs/1706.03762)
* AAN: the [average attention model, ACL2018](https://github.com/bzhangGo/transformer-aan) that accelerates the decoding!
* Fixup: [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321)
* Relative position representation: [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)

## Requirements
* python2.7
* tensorflow <= 1.13.2

## Usage
[How to use this toolkit for machine translation?](docs/usage)

## TODO:
1. organize the parameters and interpretations in config.
2. reformat and fulfill code comments
3. simplify and remove unecessary coding
4. improve rnn models 

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

## Reference
When developing this repository, I referred to the following projects:

* [Nematus](https://github.com/EdinburghNLP/nematus)
* [THUMT](https://github.com/thumt/THUMT)
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
* [Keras](https://github.com/keras-team/keras)

## Contact
For any questions or suggestions, please feel free to contact [Biao Zhang](mailto:B.Zhang@ed.ac.uk)