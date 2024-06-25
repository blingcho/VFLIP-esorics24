# submission #465

In support of open science, we will release the full implementation of VFLIP including SOTA attack methods used in the experiments to an open-source upon publication.

Currently, the evaluation code and trained models for VFLIP are provided.

## Installation

Install required packages

```shell
$ pip install -r requirements.txt
```

Download zip files of pre-trained models & backdoor patterns for cifar via below link

[Download Pre-trained models and patterns](https://drive.google.com/drive/folders/1V8LAnkrlyOoELEjGvkueKF8DajvO8yxo?usp=sharing)

Move the loaded zip files into ./toy_data directory or use --load-dir to specify your files's directory.

Unzip the loaded zip files in the directory

## Inference Example

```python
# BadVFL passive attack, a single attacker, 4 party scenario 
# "./toy_data/CIFAR10_BadVFL_ALL_False_4.pth"
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type VFLIP --attack-type BadVFL
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type NONE --attack-type BadVFL

# VILLAIN passive attack, a single attacker, 4 party scenario 
# "./toy_data/CIFAR10_VILLAIN_ALL_False_4.pth"
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type VFLIP --attack-type VILLAIN
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type NONE --attack-type VILLAIN

# BadVFL active attack, a single attacker, 4 party scenario 
# "./toy_data/CIFAR10_BadVFL_ALL_True_4.pth"
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type VFLIP --attack-type BadVFL --active
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type NONE --attack-type BadVFL --active

# VILLAIN active attack, a single attacker, 4 party scenario 
#"./toy_data/CIFAR10_VILLAIN_ALL_True_4.pth"
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type VFLIP --attack-type VILLAIN --active
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type NONE --attack-type VILLAIN --active

# BadVFL active attack with DP-SGD, a single attacker, 4 party scenario 
# "./toy_data/CIFAR10_BadVFL_DPSGD_True_4.pth"
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type DPSGD --attack-type BadVFL --active

# VILLAIN active attack with DP-SGD, a single attacker, 4 party scenario 
# "./toy_data/CIFAR10_VILLAIN_DPSGD_True_4.pth"
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 4 --defense-type DPSGD --attack-type VILLAIN --active

# BadVFL active attack, 3 attackers, 8 party scenario 
# "./toy_data/CIFAR10_BadVFL_ALL_True_8.pth"
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 8 --defense-type VFLIP --attack-type BadVFL --active --bkd-adversary 3,4,6
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 8 --defense-type NONE --attack-type BadVFL --active --bkd-adversary 3,4,6

# VILLAIN active attack, 3 attackers, 8 party scenario 
# "./toy_data/CIFAR10_VILLAIN_ALL_True_8.pth"
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 8 --defense-type VFLIP --attack-type VILLAIN --active --bkd-adversary 3,4,6
$ python main.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --gpu 0 --party-num 8 --defense-type NONE --attack-type VILLAIN --active --bkd-adversary 3,4,6
```
