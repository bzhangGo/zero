# Adaptive Feature Selection for End-to-End Speech Translation

This code base is an adaption of text-to-text `zero` to speech-to-text tasks, 
particularly for end-to-end speech translation, along with our EMNLP findings paper:
adaptive feature selection for end-to-end speech translation.

Unlike text-based machine translation, speech translation takes audio as its source
input, which is often of higher diversity across different speakers, in terms of gender
, style, accent and so on. Also, audio is often noisy and lengthy compared to its transcript.
How to reduce such noises and extract more transcript-relevant speech signals/features is 
a long-standing challenge for speech translation.

We propose [`adaptive feature selection`](https://arxiv.org/abs/2010.08518) that adopts [L0Drop](https://arxiv.org/abs/2004.11854)
to automatically filter out speech features contributing little to speech recognition. Results
on several benchmarks, including diverse language pairs, show that our method achieves substantial
performance improvement compared to the vanilla pretrained baselines with only ~15% retrained 
speech features.

In this codebase, we provide scripts and instructions for MuST-C and LibriSpeech translation tasks. 
* Please go to the [Example](./example) for more details.
* Please go to [readme](https://github.com/bzhangGo/zero/blob/master/docs/afs_speech_translation)
for more method description.

The source code and scripts are post-edited based on our initial implementation, which might contain
 bugs. For any issues, please contact [Biao Zhang](B.Zhang@ed.ac.uk).
