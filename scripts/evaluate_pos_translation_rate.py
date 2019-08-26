# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import sys
from collections import Counter


def parseargs():
    msg = "Evlauate P/R/F score for particular POS Tagged Tokens"
    usage = "{} [<args>] [-h | --help]".format(sys.argv[0])
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    parser.add_argument("--trans", type=str, required=True,
                        help="model translation")
    parser.add_argument("--refs", type=str, required=True, nargs="+",
                        help="gold reference, one or more")
    parser.add_argument("--ngram", type=int, default=4,
                        help="the maximum n for n-gram")

    parser.add_argument_group("POS setting")
    parser.add_argument("--noun", type=str, default="NN",
                        help="the pos label for noun")
    parser.add_argument("--verb", type=str, default="VB",
                        help="the pos label for verb")
    parser.add_argument("--adj", type=str, default="JJ",
                        help="the pos label for adjective")
    parser.add_argument("--adv", type=str, default="RB",
                        help="the pos label for adverb")

    parser.add_argument("--spliter", type=str, default="_",
                        help="the spliter between word and pos label")

    return parser.parse_args()


# POS conversion module
def prepare_ngram(txt, pos, ngram):
    tokens = txt.strip().split()

    words = []
    for token in tokens:
        if type(pos) is not list and pos in token:
            segs = token.strip().split('_')
            word = '_'.join(segs[:-1])
            words.append(word)
        elif type(pos) is list:
            cvt = False
            for p in pos:
                if p in token:
                    cvt = True
                    break
            if cvt:
                segs = token.strip().split('_')
                word = '_'.join(segs[:-1])
                words.append(word)
        else:
            words.append('<NaN>')

    _ngram_list = []
    for ngidx in range(ngram, len(words)):
        _ngram_list.append(' '.join(words[ngidx - ngram:ngidx]))
    ngram_list = [ng for ng in _ngram_list if '<NaN>' not in ng]

    return Counter(ngram_list)


def convert_corpus(dataset, pos, ngram):
    return [prepare_ngram(data, pos, ngram) for data in dataset]


def score(trans, refs):

    def _precision_recall_fvalue(_trans, _ref):
        t_cngrams = 0.
        t_rngrams = 0.
        m_ngrams = 0.

        for cngrams, rngrams in zip(_trans, _ref):

            t_cngrams += sum(cngrams.values())
            t_rngrams += sum(rngrams.values())

            for ngram in cngrams:
                if ngram in rngrams:
                    m_ngrams += min(cngrams[ngram], rngrams[ngram])

        precision = m_ngrams / t_cngrams if t_cngrams > 0 else 0.
        recall = m_ngrams / t_rngrams if t_rngrams > 0 else 0.
        fvalue = 2 * (recall * precision) / (recall + precision + 1e-8)

        return precision, recall, fvalue

    eval_scores = [_precision_recall_fvalue(trans, ref) for ref in refs]
    eval_scores = list(zip(*eval_scores))
    return [sum(v) / len(v) for v in eval_scores]


def evaluate_the_rate_of_specific_gram(ref, trs, pos, ngram):
    # ref: reference corpus
    # trs: translation corpus
    # pos: part-of-speech tag
    # ngram: n-gram number

    references = [convert_corpus(r, pos, ngram) for r in ref]
    candidate = convert_corpus(trs, pos, ngram)

    result = score(candidate, references)

    return pos, ngram, result


if __name__ == "__main__":
    params = parseargs()

    # loading the reference corpus
    corpus = []
    for trans_txt in params.refs:
        with open(trans_txt, 'rU') as reader:
            corpus.append(reader.readlines())
    if len(corpus) > 1:
        for cidx in range(1, len(corpus)):
            assert len(corpus[cidx]) == len(corpus[cidx - 1]), 'the length of each reference text must be the same'

    # the focused translation corpus
    with open(params.trans, 'rU') as reader:
        test = reader.readlines()
    assert len(test) == len(corpus[0]), \
        'the length of translation text should be the same as that of reference text'

    poses = [params.noun,
             params.verb,
             params.adj,
             params.adv,
             [params.noun, params.verb],
             [params.noun, params.verb, params.adj]]
    ngrams = range(params.ngram)
    for pos in poses:
        for ngram in ngrams:
            pos, ngram, evals = evaluate_the_rate_of_specific_gram(corpus, test, pos, ngram)
            print('Pos: %s, Ngram: %s, Score %s' % (pos, ngram + 1, str(evals)))
