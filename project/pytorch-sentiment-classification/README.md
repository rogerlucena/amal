# Final project - AMAL (Advanced Machine Learning and Deep Learning)

## Paper: "Towards Hierarchical Importance Attribution: Explaining Compositional Semantics for Neural Sequence Models"

Code adapted from <https://github.com/clairett/pytorch-sentiment-classification> (LSTM and CNN sentiment analysis in PyTorch - trained on Stanford Sentiment Treebank (SST2)).

The file `importance_attribution.py` is the core of the work for this project (run it using `python3` for example). It used a model trained through the `train_batch.py` script.

As the authors of the paper suggested, we foccused on the implementation of the SOC (Sampling and OCclusion) algorithm.

The code used for the sampling part is inside the `bert_babble` folder.
