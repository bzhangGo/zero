#!/bin/bash -v

# suffix of source language files
SRC=en

# suffix of target language files
TRG=de

# number of merge operations
bpe_operations=16000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=path-to-mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=path-to-subwordnmt

# tokenize
# should use sacreBLEU for final evaluation
for prefix in train dev test
do
    cat $prefix.$SRC \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > $prefix.tok.$SRC

    test -f $prefix.$TRG || continue

    cat $prefix.$TRG \
        | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $prefix.tok.$TRG
done

# note this "*.reftok.*" file should be used as tokenized reference
# by reference, we shouldn't apply punctuation normalization, or truecasing.
for prefix in dev test
do
    cat $prefix.$TRG \
        | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $prefix.reftok.$TRG
done


# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus train.tok.$SRC -model tc.$SRC
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus train.tok.$TRG -model tc.$TRG

# apply truecaser (cleaned training corpus)
for prefix in train dev test
do
    $mosesdecoder/scripts/recaser/truecase.perl -model tc.$SRC < $prefix.tok.$SRC > $prefix.tc.$SRC
    test -f $prefix.tok.$TRG || continue
    $mosesdecoder/scripts/recaser/truecase.perl -model tc.$TRG < $prefix.tok.$TRG > $prefix.tc.$TRG
done

# train BPE
cat train.tc.$SRC train.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > $SRC$TRG.bpe

# apply BPE
for prefix in train dev test
do
    $subword_nmt/apply_bpe.py -c $SRC$TRG.bpe < $prefix.tc.$SRC > $prefix.bpe.$SRC
    test -f $prefix.tc.$TRG || continue
    $subword_nmt/apply_bpe.py -c $SRC$TRG.bpe < $prefix.tc.$TRG > $prefix.bpe.$TRG
done
