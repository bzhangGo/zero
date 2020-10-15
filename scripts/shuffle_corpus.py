# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy
import h5py


def parseargs():
    parser = argparse.ArgumentParser(description="Shuffle corpus")

    parser.add_argument("--corpus", nargs="+", required=True,
                        help="input corpora")
    parser.add_argument("--audio", type=str, default="none",
                        help="audio corpora")
    parser.add_argument("--suffix", type=str, default="shuf",
                        help="Suffix of output files")
    parser.add_argument("--seed", type=int, help="Random seed")

    return parser.parse_args()


def main(args):
    name = args.corpus
    suffix = "." + args.suffix
    stream = [open(item, "r") for item in name]
    data = [fd.readlines() for fd in stream]
    minlen = min([len(lines) for lines in data])

    if args.seed:
        numpy.random.seed(args.seed)

    indices = numpy.arange(minlen)
    numpy.random.shuffle(indices)

    newstream = [open(item + suffix, "w") for item in name]

    if args.audio != "none":
        audiostream = h5py.File(args.audio + suffix + ".h5", 'w')
        audioreader = h5py.File(args.audio, 'r')

    for h, idx in enumerate(indices.tolist()):
        lines = [item[idx] for item in data]

        for line, fd in zip(lines, newstream):
            fd.write(line)

        if args.audio != "none":
            audio = audioreader["audio_{}".format(idx)][()]
            audiostream.create_dataset("audio_{}".format(h), data=audio)

    if args.audio != "none":
        audioreader.close()
        audiostream.close()

    for fdr, fdw in zip(stream, newstream):
        fdr.close()
        fdw.close()


if __name__ == "__main__":
    parsed_args = parseargs()
    main(parsed_args)
