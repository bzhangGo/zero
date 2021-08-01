#! /bin/bash


# to evaluate pronoun, we adopt the APT evaluation metric


# step 1. download APT metric
# e.g. git clone https://github.com/idiap/APT.git

# step 2. we also need automatic word alignment
# e.g. git clone https://github.com/clab/fast_align.git
#   compile the code based on the offical descriptions


# step 3. learn alignment model and generate alignments for test sets
TRG=de
align=path-to-fast-align/

# use parallel trainning information to learn word alignment model
cat train.tok.en dev.tok.en test.tok.en > corpus.en
cat train.tok.$TRG dev.tok.$TRG test.tok.$TRG > corpus.$TRG

awk '{print tolower($0)}' < corpus.$TRG > corpus.lower.$TRG
awk '{print tolower($0)}' < corpus.en > corpus.lower.en

paste -d " ||| " corpus.lower.en - - - - corpus.lower.$TRG < /dev/null > corpus.lower.en-$TRG

$align/fast_align -i corpus.lower.en-$TRG -d -o -v -p fwd_params > forward.align 2> fwd_err
$align/fast_align -i corpus.lower.en-$TRG -d -o -v -r -p rev_params > reverse.align 2> rev_err

# prepare alignment evaluation
paste -d " ||| " ../test.tok.en - - - - trans.txt < /dev/null > src-trans
awk '{print tolower($0)}' < src-trans > src-trans.lower

python $align/force_align.py fwd_params fwd_err rev_params rev_err  < src-trans.lower > trans.test.align

# step 4. evaluate
# we use the apt.config (for example) to preform evaluation (APT-required).