## $\delta$-regularized Gradient Clipping Experiments

This repository contains experiment code used to test the $\delta$-gradient clipping ($\delta$-GClip) optimization algorithm in various neural network training scenarios.

In our paper we introduce $\\delta$-Gradient Clipping ($\\delta$-GClip) as the only known way to optimize neural networks using adaptive gradients with provable convergence guarantees on deep neural networks.

The optimization algorithm utilizes the following update step:
$$x_{t+1} = x_t - h(x_t) \cdot \nabla f (x_t), \text{    where } h(x_t) \coloneqq  \eta \cdot \min ( 1 , \max ( \delta, \frac{\gamma}{ || \nabla f (x_t) ||}   )  )$$

### ResNet and VAE
```python
cd resnet_vae
# check the jupyter notebook files
ResNet18_CIFAR10.ipynb
VAE_Fashion_MNIST.ipynb
```

### Fine-tuning on BERT
```python
cd bert_finetuning
python3 bert_main.py
```

### CIFAR classification via ViT
```python
cd vit_cifar
python3 cifar_vit.py
```