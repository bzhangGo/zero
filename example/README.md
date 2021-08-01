# Simple Example

This file shows the rough procedure training an end-to-end context-aware SLT Transformer model based on MUST_C dataset.

### Step 1. Download MUST_C Dataset

Take the En->Ge as an example

* You can go to [the official website](https://ict.fbk.eu/must-c/) to download the dataset. 

* You can use the following 
[Google Drive Address](https://drive.google.com/open?id=1Mf2il_VelDIJMSio0bq7I8M9fSs-X4Ie) for downloading.


Untar the dataset.

### Step 2. Download this code base

```
git clone --branch context_aware_speech_translation https://github.com/bzhangGo/zero.git
```
Suppose the downloaded code path is `zero` so we refer to the code base as `${zero}`

### Step 3. Preprocess the speech dataset

There are two things to do in this step: 1) preprocessing the English and German text file, and 2) 
preprocessing the audio file

1) Preprocessing the text files
```
en_de=/path/to/untared/en-de/
ln -s ${en_de} en-de
ln -s en-de/data/dev/txt/dev.en .
ln -s en-de/data/dev/txt/dev.de .
ln -s en-de/data/tst-COMMON/txt/tst-COMMON.en test.en
ln -s en-de/data/tst-COMMON/txt/tst-COMMON.de test.de
ln -s en-de/data/train/txt/train.en .
ln -s en-de/data/train/txt/train.de .

# tokenize, true-case and BPE
# you need download the mosesdecoder and subword-nmt, and re-set the path in the following script 
# git clone https://github.com/rsennrich/subword-nmt.git
# git clone https://github.com/moses-smt/mosesdecoder.git 
./prepare.sh

# prepare vocabulary
python ${zero}/vocab.py train.bpe.en vocab.zero.en
python ${zero}/vocab.py train.bpe.de vocab.zero.de
```
The resulting file is: 
    
    - (train audio, train source, train target): `train.audio.h5, train.bpe.en, train.bpe.de`
    - (dev audio, dev source, dev target): `dev.audio.h5, dev.bpe.en, dev.bpe.de`
    - (test audio, test source, test target): `test.audio.h5, test.bpe.en, test.reftok.de`

Notice the test reference file: `test.reftok.de`. It's only tokenized, without punctuation normalizing and true-casing

2) The speech preprocessing script is at `${zero}/utils/t2t_speech.py`.
```
ln -s ${zero}/utils/t2t_speech.py

./build.sh
``` 
Note this would produce a large `audo.h5` file around 65GB.
(Optional) You could also shuffle the training dataset.
```
python ${zero}/scripts/shuffle_corpus.py --corpus train.bpe.en train.bpe.de --audio train.audio.h5 --suffix shuf
```
This converts training set to `train.auido.h5.shuf.h5, train.bpe.en.shuf, train.bpe.de.shuf`.

### Step 4. Train your model

See the given running scripts [`train.sh`](./train.sh) for reference. It uses about 3~4 days (with one GPU) or shorter (with more gpus).

To train AFS models, please follow `afs_step_*.sh`

### Step 5. Decoding

See the given running scripts [`test.sh`](./test.sh) for reference.

### Evaluation

Apart from BLEU scores, we also evaluate [pronoun](./pronoun_eval.sh) and [homophone](./homophone_eval.md) translations.

