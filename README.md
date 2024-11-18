# PraFFL

This repository contains the implementation for the paper "PraFFL: A Preference-Aware Scheme in Fair Federated Learning" accepted by KDD 2025.

Here is the arXiv link https://arxiv.org/abs/2404.08973.

![](client.png)

# Supported Algorithms
Our implementation depends on https://github.com/yzeng58/Improving-Fairness-via-Federated-Learning/tree/main, and our code includes FedFB, LFT+Fedavg, LTF+Ensemble, Agnosticfair, FairFed, EquiFL algorithms.

# Installation Dependencies
Our provide the packages file of our environment (requirement.txt), you can using the following command to download the environment:

```bash
# Switch Path
cd ./PraFFL
# Download dependency packages
pip install -r requirements.txt
```
# Training
## Parameters
- name: algorithm names (Available are uflfb, fflfb, agnosticfair, fairfed, fedfb, praffl)



