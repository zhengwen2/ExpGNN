# ExpGNN

This repository is the implementation of ExpGNN. To solve the problem of explainability in GNN using SCM. The overall structure of the model is as follows：
<div align=center><img width="400" height="200" src="https://github.com/zhengwen2/ExpGNN/blob/master/model.png"/></div>
Learning exogenous variables through Disentangle GCN. The relationship between endogenous variables through NN

## Requirements

- python 3.9
- pytorch 1.10

## Usage

To train the model(s), run this command:

```train
python main.py 
```

## References
[1] Disentangled Graph Convolutional Networks

[2] Structure by Architecture: Disentangled Representations without Regularization

[3]CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models
