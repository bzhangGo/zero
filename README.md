# Beyond Sentence-Level End-to-End Speech Translation: Context Helps

This code base contains the source code for our ACL2021 paper. We study the impact of context
on end-to-end speech translation (ST). By `context`, here we mean surrounding segments from the same document.

Main findings include that context improves general translation quality (+0.18-2.61 BLEU), benefits
pronoun and homophone translation, enhances the robustness to (artificial) audio segmentation errors,
and reduces latency and flicker to deliver higher quality for simultaneous translation.


### Model Training

We adopt concatenation-based context-aware ST model for experiments. 

Incorporating context into ST means that we have more audio segments to be handled by the encoder. This
further deteriorates computational inefficiency in ST, as audio often contains much longer feature frames
compared to its transcript counterpart. 

To solve this problem, we base our study on our previously proposed [adaptive feature selection, AFS](https://aclanthology.org/2020.findings-emnlp.230/)
approach. AFS automatically filters out uninformative speech encodings (about ~84%), which greatly narrows the length gap 
between speech and its transcript. 

As a consequence, we adopt a three-step training framework:
- **Step 1** Pretrain a sentence-level ASR Encoder-Decoder model with MLE loss and CTC loss.
- **Step 2** Add AFS into ASR, resulting in ASR Encoder-AFS-Decoder model, and finetune this ASR model with MLE alone.
- **Step 3** Extract the retained speech encodings offered by AFS, and treat this shortened sequence as speech input. On
top of this, we prform context-aware end-to-end ST.


In this codebase, we provide scripts and instructions for MuST-C and LibriSpeech translation tasks. 
* Please go to the [Example](./example) for more details.
* Please go to [readme](https://github.com/bzhangGo/zero/blob/master/docs/afs_speech_translation)
for more method description.


### Potential Mismatch

The source code and scripts are post-edited based on our original implementation, thus mismatch and bugs might exist.
For any issues, please contact [Biao Zhang](B.Zhang@ed.ac.uk).
