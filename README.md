# CIFAR-10-Experiment
- Train CIFAR 10- get 85 % accuracy

- ✅ Final Receptive Field = 45 (>44)
- ✅ Total stride = 2 (only downsampling at the last conv).
- ✅ Params ~ 188k (<200k).
- ✅ Depthwise separable + dilated conv included.
- ✅ GAP + FC to 10 classes included.


```
---------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
         ConvBlock-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           9,216
       BatchNorm2d-6           [-1, 32, 32, 32]              64
              ReLU-7           [-1, 32, 32, 32]               0
         ConvBlock-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]             288
      BatchNorm2d-10           [-1, 32, 32, 32]              64
             ReLU-11           [-1, 32, 32, 32]               0
           Conv2d-12           [-1, 32, 32, 32]           1,024
      BatchNorm2d-13           [-1, 32, 32, 32]              64
             ReLU-14           [-1, 32, 32, 32]               0
DepthwiseSeparable-15           [-1, 32, 32, 32]               0
           Conv2d-16           [-1, 32, 32, 32]           9,216
      BatchNorm2d-17           [-1, 32, 32, 32]              64
             ReLU-18           [-1, 32, 32, 32]               0
        ConvBlock-19           [-1, 32, 32, 32]               0
           Conv2d-20           [-1, 32, 32, 32]           9,216
      BatchNorm2d-21           [-1, 32, 32, 32]              64
             ReLU-22           [-1, 32, 32, 32]               0
        ConvBlock-23           [-1, 32, 32, 32]               0
           Conv2d-24           [-1, 32, 32, 32]           9,216
      BatchNorm2d-25           [-1, 32, 32, 32]              64
             ReLU-26           [-1, 32, 32, 32]               0
        ConvBlock-27           [-1, 32, 32, 32]               0
           Conv2d-28           [-1, 32, 32, 32]           9,216
      BatchNorm2d-29           [-1, 32, 32, 32]              64
             ReLU-30           [-1, 32, 32, 32]               0
        ConvBlock-31           [-1, 32, 32, 32]               0
           Conv2d-32           [-1, 32, 32, 32]           9,216
      BatchNorm2d-33           [-1, 32, 32, 32]              64
             ReLU-34           [-1, 32, 32, 32]               0
        ConvBlock-35           [-1, 32, 32, 32]               0
           Conv2d-36           [-1, 32, 32, 32]           9,216
      BatchNorm2d-37           [-1, 32, 32, 32]              64
             ReLU-38           [-1, 32, 32, 32]               0
        ConvBlock-39           [-1, 32, 32, 32]               0
           Conv2d-40           [-1, 32, 32, 32]           9,216
      BatchNorm2d-41           [-1, 32, 32, 32]              64
             ReLU-42           [-1, 32, 32, 32]               0
        ConvBlock-43           [-1, 32, 32, 32]               0
           Conv2d-44           [-1, 32, 32, 32]           9,216
      BatchNorm2d-45           [-1, 32, 32, 32]              64
             ReLU-46           [-1, 32, 32, 32]               0
        ConvBlock-47           [-1, 32, 32, 32]               0
           Conv2d-48           [-1, 32, 32, 32]           9,216
      BatchNorm2d-49           [-1, 32, 32, 32]              64
             ReLU-50           [-1, 32, 32, 32]               0
        ConvBlock-51           [-1, 32, 32, 32]               0
           Conv2d-52           [-1, 32, 32, 32]           9,216
      BatchNorm2d-53           [-1, 32, 32, 32]              64
             ReLU-54           [-1, 32, 32, 32]               0
        ConvBlock-55           [-1, 32, 32, 32]               0
           Conv2d-56           [-1, 32, 32, 32]           9,216
      BatchNorm2d-57           [-1, 32, 32, 32]              64
             ReLU-58           [-1, 32, 32, 32]               0
        ConvBlock-59           [-1, 32, 32, 32]               0
           Conv2d-60           [-1, 32, 32, 32]           9,216
      BatchNorm2d-61           [-1, 32, 32, 32]              64
             ReLU-62           [-1, 32, 32, 32]               0
        ConvBlock-63           [-1, 32, 32, 32]               0
           Conv2d-64           [-1, 32, 32, 32]           9,216
      BatchNorm2d-65           [-1, 32, 32, 32]              64
             ReLU-66           [-1, 32, 32, 32]               0
        ConvBlock-67           [-1, 32, 32, 32]               0
           Conv2d-68           [-1, 32, 32, 32]           9,216
      BatchNorm2d-69           [-1, 32, 32, 32]              64
             ReLU-70           [-1, 32, 32, 32]               0
        ConvBlock-71           [-1, 32, 32, 32]               0
           Conv2d-72           [-1, 32, 32, 32]           9,216
      BatchNorm2d-73           [-1, 32, 32, 32]              64
             ReLU-74           [-1, 32, 32, 32]               0
        ConvBlock-75           [-1, 32, 32, 32]               0
           Conv2d-76           [-1, 32, 32, 32]           9,216
      BatchNorm2d-77           [-1, 32, 32, 32]              64
             ReLU-78           [-1, 32, 32, 32]               0
        ConvBlock-79           [-1, 32, 32, 32]               0
           Conv2d-80           [-1, 32, 32, 32]           9,216
      BatchNorm2d-81           [-1, 32, 32, 32]              64
             ReLU-82           [-1, 32, 32, 32]               0
        ConvBlock-83           [-1, 32, 32, 32]               0
           Conv2d-84           [-1, 64, 16, 16]          18,432
      BatchNorm2d-85           [-1, 64, 16, 16]             128
             ReLU-86           [-1, 64, 16, 16]               0
        ConvBlock-87           [-1, 64, 16, 16]               0
AdaptiveAvgPool2d-88             [-1, 64, 1, 1]               0
           Linear-89                   [-1, 10]             650
================================================================
Total params: 188,618
Trainable params: 188,618
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 21.25
Params size (MB): 0.72
Estimated Total Size (MB): 21.98
----------------------------------------------------------------
```

## RF Calculations


| Layer idx | Layer type                | k | stride | dilation | eff_k | stride_total | RF_out |
| --------- | ------------------------- | - | ------ | -------- | ----- | ------------ | ------ |
| 0         | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **3**  |
| 1         | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **5**  |
| 2         | DepthwiseSep3x3           | 3 | 1      | 1        | 3     | 1            | **7**  |
| 3         | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **9**  |
| 4         | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **11** |
| 5         | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **13** |
| 6         | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **15** |
| 7         | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **17** |
| 8         | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **19** |
| 9         | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **21** |
| 10        | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **23** |
| 11        | **Dilated Conv3x3 (d=2)** | 3 | 1      | 2        | 5     | 1            | **27** |
| 12        | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **29** |
| 13        | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **31** |
| 14        | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **33** |
| 15        | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **35** |
| 16        | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **37** |
| 17        | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **39** |
| 18        | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **41** |
| 19        | Conv3x3                   | 3 | 1      | 1        | 3     | 1            | **43** |
| 20        | **Conv3x3 stride=2**      | 3 | 2      | 1        | 3     | 2            | **45** |
| GAP       | GlobalAvgPool             | – | –      | –        | –     | –            | **45** |
| FC        | Linear (64→10)            | – | –      | –        | –     | –            | –      |


## Train Logs
```log
sagemaker-user@default:~$ python3 train_cifar10.py 
Device: cuda
Model params: 188,618
Receptive field: 45, final total_stride: 2

Epoch 1 Iter 100/391 loss 2.0634 acc 17.78%
Epoch 1 Iter 200/391 loss 1.9473 acc 22.72%
Epoch 1 Iter 300/391 loss 1.8753 acc 25.89%
Epoch 1/50  train_loss 1.8220 train_acc 28.29%  val_loss 1.7927 val_acc 34.19%  time 18.0s
Saved best checkpoint at epoch 1 val_acc 34.19%
Epoch 2 Iter 100/391 loss 1.6012 acc 38.44%
Epoch 2 Iter 200/391 loss 1.5622 acc 40.11%
Epoch 2 Iter 300/391 loss 1.5317 acc 41.62%
Epoch 2/50  train_loss 1.5061 train_acc 42.81%  val_loss 1.8783 val_acc 35.89%  time 14.2s
Saved best checkpoint at epoch 2 val_acc 35.89%
Epoch 3 Iter 100/391 loss 1.3830 acc 47.96%
Epoch 3 Iter 200/391 loss 1.3577 acc 49.12%
Epoch 3 Iter 300/391 loss 1.3464 acc 49.97%
Epoch 3/50  train_loss 1.3372 train_acc 50.54%  val_loss 1.4238 val_acc 47.47%  time 14.2s
Saved best checkpoint at epoch 3 val_acc 47.47%
Epoch 4 Iter 100/391 loss 1.2756 acc 53.09%
Epoch 4 Iter 200/391 loss 1.2676 acc 53.56%
Epoch 4 Iter 300/391 loss 1.2528 acc 54.12%
Epoch 4/50  train_loss 1.2486 train_acc 54.19%  val_loss 1.3473 val_acc 52.37%  time 14.2s
Saved best checkpoint at epoch 4 val_acc 52.37%
Epoch 5 Iter 100/391 loss 1.2044 acc 55.87%
Epoch 5 Iter 200/391 loss 1.2027 acc 56.36%
Epoch 5 Iter 300/391 loss 1.1999 acc 56.26%
Epoch 5/50  train_loss 1.1901 train_acc 56.74%  val_loss 1.2987 val_acc 52.69%  time 14.2s
Saved best checkpoint at epoch 5 val_acc 52.69%
Epoch 6 Iter 100/391 loss 1.1401 acc 59.23%
Epoch 6 Iter 200/391 loss 1.1391 acc 59.14%
Epoch 6 Iter 300/391 loss 1.1311 acc 59.23%
Epoch 6/50  train_loss 1.1313 train_acc 59.28%  val_loss 1.5070 val_acc 45.41%  time 14.1s
Epoch 7 Iter 100/391 loss 1.0842 acc 61.55%
Epoch 7 Iter 200/391 loss 1.0871 acc 61.29%
Epoch 7 Iter 300/391 loss 1.0832 acc 61.45%
Epoch 7/50  train_loss 1.0760 train_acc 61.65%  val_loss 1.2237 val_acc 53.77%  time 14.2s
Saved best checkpoint at epoch 7 val_acc 53.77%
Epoch 8 Iter 100/391 loss 1.0514 acc 62.37%
Epoch 8 Iter 200/391 loss 1.0484 acc 62.53%
Epoch 8 Iter 300/391 loss 1.0429 acc 62.71%
Epoch 8/50  train_loss 1.0337 train_acc 62.96%  val_loss 1.3054 val_acc 55.84%  time 14.1s
Saved best checkpoint at epoch 8 val_acc 55.84%
Epoch 9 Iter 100/391 loss 1.0130 acc 63.95%
Epoch 9 Iter 200/391 loss 1.0153 acc 63.81%
Epoch 9 Iter 300/391 loss 1.0110 acc 63.86%
Epoch 9/50  train_loss 1.0079 train_acc 64.09%  val_loss 1.6672 val_acc 47.52%  time 14.2s
Epoch 10 Iter 100/391 loss 0.9637 acc 65.65%
Epoch 10 Iter 200/391 loss 0.9649 acc 65.73%
Epoch 10 Iter 300/391 loss 0.9709 acc 65.72%
Epoch 10/50  train_loss 0.9662 train_acc 65.83%  val_loss 1.5100 val_acc 54.92%  time 14.1s
Epoch 11 Iter 100/391 loss 0.9427 acc 67.16%
Epoch 11 Iter 200/391 loss 0.9488 acc 66.71%
Epoch 11 Iter 300/391 loss 0.9456 acc 66.81%
Epoch 11/50  train_loss 0.9460 train_acc 66.84%  val_loss 1.1310 val_acc 62.47%  time 14.1s
Saved best checkpoint at epoch 11 val_acc 62.47%
Epoch 12 Iter 100/391 loss 0.9252 acc 67.48%
Epoch 12 Iter 200/391 loss 0.9351 acc 67.19%
Epoch 12 Iter 300/391 loss 0.9253 acc 67.59%
Epoch 12/50  train_loss 0.9244 train_acc 67.69%  val_loss 1.3577 val_acc 54.10%  time 14.0s
Epoch 13 Iter 100/391 loss 0.9217 acc 67.28%
Epoch 13 Iter 200/391 loss 0.9159 acc 67.54%
Epoch 13 Iter 300/391 loss 0.9063 acc 67.96%
Epoch 13/50  train_loss 0.9048 train_acc 68.06%  val_loss 0.9780 val_acc 65.47%  time 14.1s
Saved best checkpoint at epoch 13 val_acc 65.47%
Epoch 14 Iter 100/391 loss 0.8706 acc 69.38%
Epoch 14 Iter 200/391 loss 0.8864 acc 68.80%
Epoch 14 Iter 300/391 loss 0.8839 acc 68.87%
Epoch 14/50  train_loss 0.8887 train_acc 68.69%  val_loss 1.1344 val_acc 62.02%  time 14.1s
Epoch 15 Iter 100/391 loss 0.8576 acc 70.18%
Epoch 15 Iter 200/391 loss 0.8552 acc 70.13%
Epoch 15 Iter 300/391 loss 0.8596 acc 69.89%
Epoch 15/50  train_loss 0.8665 train_acc 69.76%  val_loss 1.1405 val_acc 59.04%  time 14.0s
Epoch 16 Iter 100/391 loss 0.8391 acc 70.28%
Epoch 16 Iter 200/391 loss 0.8372 acc 70.63%
Epoch 16 Iter 300/391 loss 0.8403 acc 70.59%
Epoch 16/50  train_loss 0.8398 train_acc 70.52%  val_loss 1.1057 val_acc 64.76%  time 14.1s
Epoch 17 Iter 100/391 loss 0.8320 acc 71.18%
Epoch 17 Iter 200/391 loss 0.8255 acc 71.40%
Epoch 17 Iter 300/391 loss 0.8264 acc 71.21%
Epoch 17/50  train_loss 0.8246 train_acc 71.33%  val_loss 1.2418 val_acc 57.34%  time 14.1s
Epoch 18 Iter 100/391 loss 0.8070 acc 71.59%
Epoch 18 Iter 200/391 loss 0.8239 acc 71.23%
Epoch 18 Iter 300/391 loss 0.8166 acc 71.53%
Epoch 18/50  train_loss 0.8117 train_acc 71.71%  val_loss 0.8923 val_acc 69.74%  time 14.1s
Saved best checkpoint at epoch 18 val_acc 69.74%
Epoch 19 Iter 100/391 loss 0.7883 acc 72.45%
Epoch 19 Iter 200/391 loss 0.7853 acc 72.64%
Epoch 19 Iter 300/391 loss 0.7911 acc 72.41%
Epoch 19/50  train_loss 0.7921 train_acc 72.32%  val_loss 0.9337 val_acc 67.97%  time 14.1s
Epoch 20 Iter 100/391 loss 0.7815 acc 72.66%
Epoch 20 Iter 200/391 loss 0.7873 acc 72.55%
Epoch 20 Iter 300/391 loss 0.7835 acc 72.74%
Epoch 20/50  train_loss 0.7825 train_acc 72.89%  val_loss 0.9152 val_acc 67.16%  time 14.1s
Epoch 21 Iter 100/391 loss 0.7579 acc 73.88%
Epoch 21 Iter 200/391 loss 0.7582 acc 73.88%
Epoch 21 Iter 300/391 loss 0.7601 acc 73.72%
Epoch 21/50  train_loss 0.7686 train_acc 73.30%  val_loss 0.9088 val_acc 67.15%  time 14.1s
Epoch 22 Iter 100/391 loss 0.7406 acc 74.24%
Epoch 22 Iter 200/391 loss 0.7426 acc 74.10%
Epoch 22 Iter 300/391 loss 0.7460 acc 74.05%
Epoch 22/50  train_loss 0.7492 train_acc 73.94%  val_loss 1.0024 val_acc 66.42%  time 14.3s
Epoch 23 Iter 100/391 loss 0.7475 acc 74.08%
Epoch 23 Iter 200/391 loss 0.7415 acc 74.40%
Epoch 23 Iter 300/391 loss 0.7389 acc 74.53%
Epoch 23/50  train_loss 0.7355 train_acc 74.62%  val_loss 0.8534 val_acc 71.06%  time 14.4s
Saved best checkpoint at epoch 23 val_acc 71.06%
Epoch 24 Iter 100/391 loss 0.7164 acc 74.67%
Epoch 24 Iter 200/391 loss 0.7247 acc 74.60%
Epoch 24 Iter 300/391 loss 0.7219 acc 74.83%
Epoch 24/50  train_loss 0.7190 train_acc 75.03%  val_loss 0.8501 val_acc 69.36%  time 14.3s
Epoch 25 Iter 100/391 loss 0.6945 acc 75.91%
Epoch 25 Iter 200/391 loss 0.6943 acc 75.92%
Epoch 25 Iter 300/391 loss 0.6965 acc 75.94%
Epoch 25/50  train_loss 0.7011 train_acc 75.80%  val_loss 0.8545 val_acc 70.49%  time 14.4s
Epoch 26 Iter 100/391 loss 0.6938 acc 75.60%
Epoch 26 Iter 200/391 loss 0.6887 acc 76.01%
Epoch 26 Iter 300/391 loss 0.6903 acc 75.95%
Epoch 26/50  train_loss 0.6877 train_acc 76.04%  val_loss 0.8134 val_acc 71.70%  time 14.4s
Saved best checkpoint at epoch 26 val_acc 71.70%
Epoch 27 Iter 100/391 loss 0.6620 acc 77.27%
Epoch 27 Iter 200/391 loss 0.6721 acc 76.75%
Epoch 27 Iter 300/391 loss 0.6787 acc 76.49%
Epoch 27/50  train_loss 0.6745 train_acc 76.66%  val_loss 0.9001 val_acc 69.60%  time 14.2s
Epoch 28 Iter 100/391 loss 0.6493 acc 77.83%
Epoch 28 Iter 200/391 loss 0.6475 acc 77.86%
Epoch 28 Iter 300/391 loss 0.6498 acc 77.61%
Epoch 28/50  train_loss 0.6546 train_acc 77.43%  val_loss 0.8860 val_acc 70.24%  time 14.2s
Epoch 29 Iter 100/391 loss 0.6515 acc 77.77%
Epoch 29 Iter 200/391 loss 0.6472 acc 77.75%
Epoch 29 Iter 300/391 loss 0.6494 acc 77.63%
Epoch 29/50  train_loss 0.6501 train_acc 77.56%  val_loss 1.0338 val_acc 66.56%  time 14.2s
Epoch 30 Iter 100/391 loss 0.6219 acc 78.61%
Epoch 30 Iter 200/391 loss 0.6210 acc 78.66%
Epoch 30 Iter 300/391 loss 0.6230 acc 78.55%
Epoch 30/50  train_loss 0.6230 train_acc 78.50%  val_loss 0.7222 val_acc 75.29%  time 14.1s
Saved best checkpoint at epoch 30 val_acc 75.29%
Epoch 31 Iter 100/391 loss 0.5954 acc 79.87%
Epoch 31 Iter 200/391 loss 0.6026 acc 79.48%
Epoch 31 Iter 300/391 loss 0.6101 acc 79.09%
Epoch 31/50  train_loss 0.6148 train_acc 78.95%  val_loss 0.9657 val_acc 68.57%  time 14.1s
Epoch 32 Iter 100/391 loss 0.5875 acc 79.72%
Epoch 32 Iter 200/391 loss 0.5962 acc 79.49%
Epoch 32 Iter 300/391 loss 0.5964 acc 79.44%
Epoch 32/50  train_loss 0.5919 train_acc 79.65%  val_loss 0.7158 val_acc 76.20%  time 14.1s
Saved best checkpoint at epoch 32 val_acc 76.20%
Epoch 33 Iter 100/391 loss 0.5842 acc 80.14%
Epoch 33 Iter 200/391 loss 0.5723 acc 80.47%
Epoch 33 Iter 300/391 loss 0.5711 acc 80.46%
Epoch 33/50  train_loss 0.5742 train_acc 80.35%  val_loss 0.6269 val_acc 78.93%  time 14.1s
Saved best checkpoint at epoch 33 val_acc 78.93%
Epoch 34 Iter 100/391 loss 0.5596 acc 80.86%
Epoch 34 Iter 200/391 loss 0.5555 acc 80.80%
Epoch 34 Iter 300/391 loss 0.5622 acc 80.75%
Epoch 34/50  train_loss 0.5626 train_acc 80.76%  val_loss 0.8451 val_acc 72.36%  time 14.1s
Epoch 35 Iter 100/391 loss 0.5344 acc 81.77%
Epoch 35 Iter 200/391 loss 0.5319 acc 81.82%
Epoch 35 Iter 300/391 loss 0.5377 acc 81.63%
Epoch 35/50  train_loss 0.5396 train_acc 81.42%  val_loss 0.7306 val_acc 75.36%  time 14.2s
Epoch 36 Iter 100/391 loss 0.5238 acc 82.17%
Epoch 36 Iter 200/391 loss 0.5276 acc 81.80%
Epoch 36 Iter 300/391 loss 0.5305 acc 81.77%
Epoch 36/50  train_loss 0.5290 train_acc 81.88%  val_loss 0.5836 val_acc 79.90%  time 14.1s
Saved best checkpoint at epoch 36 val_acc 79.90%
Epoch 37 Iter 100/391 loss 0.4937 acc 83.42%
Epoch 37 Iter 200/391 loss 0.5046 acc 82.68%
Epoch 37 Iter 300/391 loss 0.5086 acc 82.59%
Epoch 37/50  train_loss 0.5083 train_acc 82.54%  val_loss 0.6029 val_acc 79.90%  time 14.2s
Epoch 38 Iter 100/391 loss 0.4964 acc 82.80%
Epoch 38 Iter 200/391 loss 0.4904 acc 82.99%
Epoch 38 Iter 300/391 loss 0.4933 acc 82.88%
Epoch 38/50  train_loss 0.4903 train_acc 82.98%  val_loss 0.6156 val_acc 79.99%  time 14.1s
Saved best checkpoint at epoch 38 val_acc 79.99%
Epoch 39 Iter 100/391 loss 0.4835 acc 83.55%
Epoch 39 Iter 200/391 loss 0.4784 acc 83.69%
Epoch 39 Iter 300/391 loss 0.4759 acc 83.74%
Epoch 39/50  train_loss 0.4775 train_acc 83.67%  val_loss 0.5072 val_acc 83.15%  time 14.2s
Saved best checkpoint at epoch 39 val_acc 83.15%
Epoch 40 Iter 100/391 loss 0.4556 acc 84.26%
Epoch 40 Iter 200/391 loss 0.4469 acc 84.70%
Epoch 40 Iter 300/391 loss 0.4526 acc 84.46%
Epoch 40/50  train_loss 0.4547 train_acc 84.41%  val_loss 0.6055 val_acc 79.81%  time 14.2s
Epoch 41 Iter 100/391 loss 0.4311 acc 84.99%
Epoch 41 Iter 200/391 loss 0.4387 acc 84.83%
Epoch 41 Iter 300/391 loss 0.4396 acc 84.86%
Epoch 41/50  train_loss 0.4378 train_acc 85.00%  val_loss 0.4946 val_acc 83.20%  time 14.2s
Saved best checkpoint at epoch 41 val_acc 83.20%
Epoch 42 Iter 100/391 loss 0.4234 acc 85.36%
Epoch 42 Iter 200/391 loss 0.4212 acc 85.45%
Epoch 42 Iter 300/391 loss 0.4197 acc 85.47%
Epoch 42/50  train_loss 0.4217 train_acc 85.34%  val_loss 0.4458 val_acc 84.76%  time 14.3s
Saved best checkpoint at epoch 42 val_acc 84.76%
Epoch 43 Iter 100/391 loss 0.4115 acc 85.80%
Epoch 43 Iter 200/391 loss 0.4009 acc 86.35%
Epoch 43 Iter 300/391 loss 0.4021 acc 86.28%
Epoch 43/50  train_loss 0.4036 train_acc 86.27%  val_loss 0.4490 val_acc 84.75%  time 14.3s
Epoch 44 Iter 100/391 loss 0.4005 acc 86.05%
Epoch 44 Iter 200/391 loss 0.3942 acc 86.44%
Epoch 44 Iter 300/391 loss 0.3928 acc 86.50%
Epoch 44/50  train_loss 0.3921 train_acc 86.53%  val_loss 0.4680 val_acc 84.31%  time 14.3s
Epoch 45 Iter 100/391 loss 0.3777 acc 87.38%
Epoch 45 Iter 200/391 loss 0.3742 acc 87.54%
Epoch 45 Iter 300/391 loss 0.3731 acc 87.46%
Epoch 45/50  train_loss 0.3745 train_acc 87.34%  val_loss 0.3958 val_acc 86.85%  time 14.4s
Saved best checkpoint at epoch 45 val_acc 86.85%
Epoch 46 Iter 100/391 loss 0.3697 acc 87.15%
Epoch 46 Iter 200/391 loss 0.3672 acc 87.40%
Epoch 46 Iter 300/391 loss 0.3655 acc 87.46%
Epoch 46/50  train_loss 0.3635 train_acc 87.57%  val_loss 0.3841 val_acc 87.10%  time 14.4s
Saved best checkpoint at epoch 46 val_acc 87.10%
Epoch 47 Iter 100/391 loss 0.3439 acc 88.40%
Epoch 47 Iter 200/391 loss 0.3451 acc 88.26%
Epoch 47 Iter 300/391 loss 0.3495 acc 88.13%
Epoch 47/50  train_loss 0.3513 train_acc 88.03%  val_loss 0.3706 val_acc 87.77%  time 14.3s
Saved best checkpoint at epoch 47 val_acc 87.77%
Epoch 48 Iter 100/391 loss 0.3446 acc 88.20%
Epoch 48 Iter 200/391 loss 0.3482 acc 88.02%
Epoch 48 Iter 300/391 loss 0.3466 acc 88.14%
Epoch 48/50  train_loss 0.3442 train_acc 88.25%  val_loss 0.3664 val_acc 87.94%  time 14.3s
Saved best checkpoint at epoch 48 val_acc 87.94%
Epoch 49 Iter 100/391 loss 0.3451 acc 88.26%
Epoch 49 Iter 200/391 loss 0.3379 acc 88.52%
Epoch 49 Iter 300/391 loss 0.3374 acc 88.50%
Epoch 49/50  train_loss 0.3377 train_acc 88.51%  val_loss 0.3700 val_acc 87.73%  time 14.2s
Epoch 50 Iter 100/391 loss 0.3449 acc 88.28%
Epoch 50 Iter 200/391 loss 0.3353 acc 88.62%
Epoch 50 Iter 300/391 loss 0.3353 acc 88.69%
Epoch 50/50  train_loss 0.3338 train_acc 88.69%  val_loss 0.3623 val_acc 87.96%  time 14.2s
Saved best checkpoint at epoch 50 val_acc 87.96%
Training complete. Best val acc: 87.96
```
