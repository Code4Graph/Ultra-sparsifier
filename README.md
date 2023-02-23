# Graph Ultra-sparsifier
This is the original pytorch implementation of Graph Ultra-sparsifier in the following paper: 

[SEMI-SUPERVISED GRAPH ULTRA-SPARSIFIERS USING REWEIGHTED ℓ1 OPTIMIZATION, ICASSP 2023]


## Datasets
Here using Cora as an example.

## Pretrained step
    
    python pretrain.py

Note that different datasets should correspond to different parameter settings in the pretrained step.

## Sparse training step

    python sparse_train.py

Note that you may need to change some parameter settings during the sparse training.

## Appendix
Some supplementary materials are included in the Appendix.pdf.
