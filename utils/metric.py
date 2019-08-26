# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import sys
import argparse
from collections import defaultdict

'''https://github.com/DeepLearnXMU/Otem-Utem'''


def _get_refs(ref):
    """Get reference files, ref indicates the path, following the multi-bleu tradition."""
    refs = []

    # return the existed reference file, and assume only one reference
    if os.path.exists(ref):
        refs.append(ref)
    else:
        # the reference does not exist, check whether the indexed file exist, usually multiple references
        if not os.path.exists(ref + "0"):
            print('Error: could not find proper reference file ', ref + "0", file=sys.stderr)
            sys.exit(1)

        # enumerate all possible references
        while True:
            cur_file = ref + "%d" % len(refs)
            if not os.path.exists(cur_file):
                break
            refs.append(cur_file)
    return refs


def _tokenize(s):
    """An interface for tokenization, currently we rely on external tokenizers
       i.e. We assume all the inputs have been well-tokenized
    """
    return s.split()


def _read(f, lc=False):
    """Reading all contents inside the file `f`, "lc" tells whether open the 'lower case' function."""
    return [_tokenize(line.strip()) if not lc else _tokenize(line.strip().lower())
            for line in open(f, 'rU').readlines()]


def _get_ngram_list(sentence, ngram=4):
    """Read all ngrams inside the sentences, default up to 4."""
    ngram_dict = defaultdict(int)
    for n in range(1, ngram + 1):
        for start in range(0, len(sentence) - (n - 1)):
            ngram_str = ' '.join(sentence[start:start + n])
            ngram_dict[ngram_str] += 1
    return ngram_dict


def _common_strategies(choices):
    """Generate some common strategies to deal with multiple references."""
    return {'min': min(choices),
            'max': max(choices),
            'avg': sum(choices) * 1. / len(choices)
            }


def _get_length_reference(ref_lengths, cand_length, strategy="best_match"):
    """When multiple references exist, return the length of a preferred references."""

    # different strategy, no one is absolutely correct
    strategies = _common_strategies(ref_lengths)

    # the best matched cases
    length, diff = 9999, 9999
    for r in ref_lengths:
        d = abs(r - cand_length)

        if d < diff:
            length, diff = r, d
        elif d == diff:
            if r < length:
                length = r
    strategies['best_match'] = length

    return strategies[strategy]


def _safe_log(d):
    """Deal with invalid inputs."""
    if d <= 0:
        print("WARNING, a non-positive number is processed by log", file=sys.stderr)
        return -9999999999

    return math.log(d)


def otem(cand, refs, bp='closest', smooth=False, n=2, weights=None):
    """Over-Translation Evaluation Metric, LOWER is BETTER"""
    len_c = 0
    len_ref = 0

    tngram_corpus, ongram_corpus = defaultdict(int), defaultdict(int)

    # scan all candidates in the corpus
    for candidate, references in zip(cand, refs):
        len_c += len(candidate)
        len_ref += _get_length_reference([len(r) for r in references], len(candidate),
                                         strategy='best_match' if bp == 'closest' else 'min')

        # get all n-grams in current candidate from n = 1...4
        cngrams = _get_ngram_list(candidate, ngram=n)

        tngram_sample, ongram_sample = defaultdict(int), defaultdict(int)

        for reference in references:
            rngrams = _get_ngram_list(reference, ngram=n)

            for ngram in cngrams:
                tngram_sample[ngram] = cngrams[ngram]

                ngram_otem = 0

                # case 1: current n-gram doesn't appear in current reference at all,
                #         but appears in current candidate more than once
                if ngram not in rngrams:
                    if cngrams[ngram] > 1:
                        ngram_otem = cngrams[ngram] - 1
                elif cngrams[ngram] > rngrams[ngram]:
                    # case 2: the n-gram occurs in both reference and candidate, but the occurrence is more in candidate
                    ngram_otem = cngrams[ngram] - rngrams[ngram]

                if ngram_otem > 0:
                    if ongram_sample[ngram] == 0:
                        ongram_sample[ngram] = ngram_otem
                    else:
                        ongram_sample[ngram] = min(ongram_sample[ngram], ngram_otem)

        for ngram in cngrams:
            nl = len(ngram.split())
            tngram_corpus[nl] += tngram_sample[ngram]
            ongram_corpus[nl] += ongram_sample[ngram]

    if len_ref == 0:
        return 0.

    lp = 1.
    multi_otem = defaultdict(int)

    for i in range(1, n + 1):
        if i in tngram_corpus:
            if smooth and i > 1:
                ongram_corpus[i] += 1
                tngram_corpus[i] += 1
            multi_otem[i] += ongram_corpus[i] * 1. / tngram_corpus[i]

    # Over-translation: candidate prefered to be longer, so penalize long translations
    if len_c >= len_ref:
        lp = math.exp(1. - len_ref * 1. / len_c)

    if weights is None:
        weights = [1. / n for _ in range(n)]
    assert len(weights) == n, 'ERROR: the length of weights ({}) should be equal to n ({})'.format(len(weights), n)

    score = lp * math.exp(sum(_safe_log(multi_otem[i+1]) * weights[i] for i in range(n)))

    return score


def utem(cand, refs, bp='closest', smooth=False, n=4, weights=None):
    """Under-Translation Evaluation Metric, LOWER is BETTER"""
    len_c = 0
    len_ref = 0

    tngram_corpus, mngram_corpus = defaultdict(int), defaultdict(int)

    # scan all candidates in the corpus
    for candidate, references in zip(cand, refs):
        len_c += len(candidate)
        len_ref += _get_length_reference([len(r) for r in references], len(candidate),
                                         strategy='best_match' if bp == 'closest' else 'min')

        # get all n-grams in current candidate from n = 1...4
        cngrams = _get_ngram_list(candidate, ngram=n)

        tngram_sample, mngram_sample = defaultdict(list), defaultdict(list)

        for reference in references:
            rngrams = _get_ngram_list(reference, ngram=n)

            tngram_ref, mngram_ref = defaultdict(int), defaultdict(int)

            # count the number of under-translation n-grams in current candidate compared with current reference
            for ngram in rngrams:
                nl = len(ngram.split())

                tngram_ref[nl] += rngrams[ngram]

                # case 1: current n-gram doesn't appear in the candidate at all
                if ngram not in cngrams:
                    mngram_ref[nl] += rngrams[ngram]
                elif rngrams[ngram] > cngrams[ngram]:
                    # case 2: the n-gram occurs in both reference and candidate, but the occurrence is more in reference
                    mngram_ref[nl] += rngrams[ngram] - cngrams[ngram]

            for i in tngram_ref:
                tngram_sample[i].append(tngram_ref[i])
                mngram_sample[i].append(mngram_ref[i])

        for i in tngram_sample:
            m = _common_strategies(mngram_sample[i])['min']
            t = _common_strategies(tngram_sample[i])['max']

            mngram_corpus[i] += m
            tngram_corpus[i] += t

    if len_ref == 0:
        return 0.

    lp = 1.
    multi_utem = defaultdict(int)
    for i in range(1, n + 1):
        if i in tngram_corpus:
            if smooth and i > 1:
                mngram_corpus[i] += 1
                tngram_corpus[i] += 1
            multi_utem[i] += mngram_corpus[i] * 1. / tngram_corpus[i]

    # Under-translation: candidates perfered to be shorter, so penalize short translations
    if len_c <= len_ref:
        lp = math.exp(1. - len_c * 1. / len_ref)

    if weights is None:
        weights = [1. / n for _ in range(n)]
    assert len(weights) == n, 'ERROR: the length of weights ({}) should be equal to n ({})'.format(len(weights), n)

    score = lp * math.exp(sum(_safe_log(multi_utem[i+1]) * weights[i] for i in range(n)))

    return score


def bleu(cand, refs, bp='closest', smooth=False, n=4, weights=None):
    """BLEU Evaluation Metric, LARGER is BETTER"""
    len_c = 0
    len_ref = 0

    tngram_corpus, bngram_corpus = defaultdict(int), defaultdict(int)

    # scan all candidates in the corpus
    for candidate, references in zip(cand, refs):
        len_c += len(candidate)
        len_ref += _get_length_reference([len(r) for r in references], len(candidate),
                                         strategy='best_match' if bp == 'closest' else 'min')

        # get all n-grams in current candidate from n = 1...4
        cngrams = _get_ngram_list(candidate, ngram=n)

        tngram_sample, bngram_sample = defaultdict(int), defaultdict(int)

        for reference in references:
            rngrams = _get_ngram_list(reference, ngram=n)

            for ngram in cngrams:
                tngram_sample[ngram] = cngrams[ngram]
                if ngram in rngrams:
                    bngram_sample[ngram] = max(bngram_sample[ngram], min(rngrams[ngram], cngrams[ngram]))

        for ngram in cngrams:
            nl = len(ngram.split())
            tngram_corpus[nl] += tngram_sample[ngram]
            bngram_corpus[nl] += bngram_sample[ngram]

    if len_ref == 0:
        return 0.

    lp = 1.
    multi_bleu = defaultdict(int)

    for i in range(1, n + 1):
        if i in tngram_corpus:
            if smooth and i > 1:
                bngram_corpus[i] += 1
                tngram_corpus[i] += 1
            multi_bleu[i] += bngram_corpus[i] * 1. / tngram_corpus[i]

    # BLEU: candidate prefered to be longer, so penalize long translations
    if len_c <= len_ref:
        lp = math.exp(1. - len_ref * 1. / len_c)

    if weights is None:
        weights = [1. / n for _ in range(n)]
    assert len(weights) == n, 'ERROR: the length of weights ({}) should be equal to n ({})'.format(len(weights), n)

    score = lp * math.exp(sum(_safe_log(multi_bleu[i+1]) * weights[i] for i in range(n)))

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Over-translation evaluation metric (OTEM), under-translation evaluation metric (UTEM), '
                    'BLEU on multiple references.')
    parser.add_argument('-lc', help='Lowercase, i.e case-insensitive setting', action='store_true')
    parser.add_argument('-bp', help='Length penalty', default='closest', choices=['shortest', 'closest'])
    parser.add_argument('candidate', help='The candidate translation generated by MT system')
    parser.add_argument('reference', help='The references like reference or reference0, reference1, ...')

    args = parser.parse_args()

    cand = args.candidate
    refs = _get_refs(args.reference)

    cand_sentences = _read(cand, args.lc)
    refs_sentences = [_read(ref, args.lc) for ref in refs]

    assert len(cand_sentences) == len(refs_sentences[0]), \
        'ERROR: the length of candidate and reference must be the same.'

    refs_sentences = list(zip(*refs_sentences))

    otem_score = otem(cand_sentences, refs_sentences, n=2)  # OTEM-2
    utem_score = utem(cand_sentences, refs_sentences, n=4)  # UTEM-4
    bleu_score = bleu(cand_sentences, refs_sentences, n=4)  # BLEU-4

    print('OTEM-2/UTEM-4/BLEU-4: {}/{}/{}'.format(otem_score, utem_score, bleu_score))
