# TokenLearner in Pytorch

## Unofficial Pytorch implementation of TokenLearner by Ryoo et al. from Google AI 
* [TokenLearner: Adaptive Space-Time Tokenization for Videos](https://openreview.net/forum?id=z-l1kpDXs88)
* [Official TokenLearner code](https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py)

The following figure presents a pictorial overview of the module
([source](https://ai.googleblog.com/2021/12/improving-vision-transformer-efficiency.html)).
![TokenLearner module GIF](https://blogger.googleusercontent.com/img/a/AVvXsEiylT3_nmd9-tzTnz3g3Vb4eTn-L5sOwtGJOad6t2we7FsjXSpbLDpuPrlInAhtE5hGCA_PfYTJtrIOKfLYLYGcYXVh1Ksfh_C1ZC-C8gw6GKtvrQesKoMrEA_LU_Gd5srl5-3iZDgJc1iyCELoXtfuIXKJ2ADDHOBaUjhU8lXTVdr2E7bCVaFgVHHkmA=w640-h208)

In this repository, we implement the TokenLearner module and demonstrate its performance with a ViT tiny and the CIFAR-10 dataset.

### Installation
`pip install -r requirements.txt`

### Run

python train_classifier.py
