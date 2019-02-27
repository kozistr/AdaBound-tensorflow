# AdaBound
An optimizer that trains as fast as Adam and as good as SGD in Tensorflow

This repo is based on pytorch impl [original repo](https://github.com/Luolc/AdaBound)

## Explaination
An optimizer that trains as fast as Adam and as good as SGD, 
for developing state-of-the-art deep learning models on a wide variety of popular tasks in the field of CV, NLP, and etc.

Based on Luo et al. (2019). [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://openreview.net/forum?id=Bkg3g2R9FX). In Proc. of ICLR 2019.

## Requirement
* Python 3.x
* Tensorflow 1.x (maybe for 2.x)

## Usage

`optimizer = AdaBound(lr=1e-3, final_lr=1e-1)`

## Demos


## Citation

```
@inproceedings{Luo2019AdaBound,
  author = {Luo, Liangchen and Xiong, Yuanhao and Liu, Yan and Sun, Xu},
  title = {Adaptive Gradient Methods with Dynamic Bound of Learning Rate},
  booktitle = {Proceedings of the 7th International Conference on Learning Representations},
  month = {May},
  year = {2019},
  address = {New Orleans, Louisiana}
}
```
