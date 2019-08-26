# coding: utf-8

from lrs import vanillalr, noamlr, scorelr, gnmtplr, epochlr, cosinelr


def get_lr(params):

    strategy = params.lrate_strategy.lower()

    if strategy == "noam":
        return noamlr.NoamDecayLr(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
            params.warmup_steps,
            params.hidden_size
        )
    elif strategy == "gnmt+":
        return gnmtplr.GNMTPDecayLr(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
            params.warmup_steps,
            params.nstable,
            params.lrdecay_start,
            params.lrdecay_end
        )
    elif strategy == "epoch":
        return epochlr.EpochDecayLr(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
            params.lrate_decay,
        )
    elif strategy == "score":
        return scorelr.ScoreDecayLr(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
            history_scores=[v[1] for v in params.recorder.valid_script_scores],
            decay=params.lrate_decay,
            patience=params.lrate_patience,
        )
    elif strategy == "vanilla":
        return vanillalr.VanillaLR(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
        )
    elif strategy == "cosine":
        return cosinelr.CosineDecayLr(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
            params.warmup_steps,
            params.lrate_decay,
            t_mult=params.cosine_factor,
            update_period=params.cosine_period
        )
    else:
        raise NotImplementedError(
            "{} is not supported".format(strategy))
