

The homophone evaluation follows that of APT-based pronoun evaluation
the difference is that, instead of feeding pronouns we are interested in,
  we feed homonphones
that is change `list_en` to `list_homo`

Notice that list_homo differs across different language pairs, and we generate
automatically by identifying words with the same phonemes. This can be done
with toolkit like [Forcd Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)

See [list_homo](./list_homo) for example.
