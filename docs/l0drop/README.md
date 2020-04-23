# On Sparsifying Encoder Outputs in Sequence-to-Sequence Models

[paper](#)


## Motivation

Standard encoder-decoder models for sequence-to-sequence learning feed all encoder outputs to the decoder 
for sequence generation. By contrast, we propose L0Drop which forces model to route information 
through a subset of the encoder outputs. The subset is learned by L0Drop automatically.

<img src="figures/l0drop.png"  width=500 />

The figure above shows the difference of L0Drop compared to existing attention models. 

Note that L0Drop is data-driven
and task-agnostic. We applied it to machine translation as well as document summarization tasks. 
Experimental results show that the encoder outputs can be compressed by L0Drop with a rate of ~40% at little cost
of generation performance. L0Drop shortens the encoding sequence fed to the decoder, resulting in 1.65x speedup on
document summarization tasks.


## Code

We implement the model in `models/transformer_l0drop.py` and `modules/l0norm.py`

## Model Training

It's possible to train Transformer with L0Drop from scratch by setting proper schedulers for `\lambda`, 
a hyperparameter loosely controling the sparsity rate of L0Drop. Unfortunately, the optimal scheduler is
data&task-dependent.

We suggest first pre-train a normal Transformer model, and then finetune the Transfomer+L0Drop. This could
save a lot of efforts.

* Step 1. train a normal Transformer model as described [here](docs/usage/README.md)
* Step 2. finetune L0Drop using the following command:
```
data_dir=the preprocessed data directory
python run.py --mode train --parameters=\
l0_norm_reg_scalar=0.2,\
model_name="transformer_l0drop",scope_name="transformer",\
pretrained_model="path-to-pretrained-transformer"\
```
where `l0_norm_reg_scalar` is the `\lambda`, and `0.2` is a nice hyperparameter in our experiments.

## Evaluation

The evaluation follows the same procedure as the baseline Transformer.