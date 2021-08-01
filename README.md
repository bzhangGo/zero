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

As a consequence, we adopt a four-step training framework:

* AFS pretraining
    - [**Step 1**](./example/afs_step_1_asr_pretrain.sh) Pretrain a sentence-level ASR Encoder-Decoder model with MLE loss and CTC loss.
    - [**Step 2**](./example/afs_step_2_afs_pretrain.sh) Add AFS into ASR, resulting in ASR Encoder-AFS-Decoder model, and finetune this ASR model with MLE alone.
* Sentence-level ST pretraining
    - [**Step 3**](./example/afs_step_3_afs_st_train.sh) Drop ASR Decoder and freeze ASR Encoder-AFS structure and treat it as a dynamic feature extractor. On top of 
it, pretrain a sentence-level end-to-end ST Encoder-Decoder model.
* Document-level ST finetuning
    - **Step 4** Use ASR Encoder-AFS model to extract speech features, and finetune the sentence-level ST model on concatenated 
speech feature sequences to grow it to the document-level ST.

Note we observe that sentence-level ST pretraining improves the context-aware ST in our experiments.


### Model Evaluation

We adopt different strategies for inference, including chunk-based decoding (cbd), sliding-window based decoding without target prefix
constraint (swbd), sliding-window based decoding with target prefix constrant (swbd-cons) and in-model ensemble decoding (imed).

These decoding methods mainly differ in how to handle target-side context. IMED is the one we mainly used in our paper, and CBD is the 
simplest one.

We implement all these decoding methods in this code base. 


### Potential Mismatch

The source code and scripts are post-edited based on our original implementation, thus mismatch and bugs might exist.
For any issues, please contact [Biao Zhang](B.Zhang@ed.ac.uk).
