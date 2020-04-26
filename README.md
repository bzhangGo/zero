
This codebase is designed for our multilingual paper: including language-aware layer normalization and
language-aware linear mapping. Improving language-specific modeling help enhance the model capacity on
different translation directions so as to deliver better translation performance.

Deepening neural models also help a lot. We applied the depth-scaled initialization to alleviate gradient
vanishing and got decent performance gains.

- About training, please see [readme](https://github.com/bzhangGo/zero/blob/master/docs/multilingual_laln_lalt/README.md).

- About our dataset (OPUS-100),please see [opus-100](https://github.com/EdinburghNLP/opus-100-corpus).

The source code and scripts might contain bugs. For any issues, please contact [Biao Zhang](B.Zhang@ed.ac.uk).