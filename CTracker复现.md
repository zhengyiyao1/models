## 数据集

在数据集根目录下，运行`generate_csv.py`，得到`train_annots.csv`文件，然后移动到数据集目录下。

创建`train_labels.csv`文件，只需要填一行`bee,0`

![image-20220124195830191](C:/Users/Admistr/AppData/Roaming/Typora/typora-user-images/image-20220124195830191.png)

## 性能

### 1. 官方数据集

训练结果

- 精度

- 速度

- 占用资源

  | Parameters           | datasize(nums) [1080p] | GPU memory(GB) | time/epoch(min) |
  | -------------------- | ---------------------- | -------------- | --------------- |
  | resnet18+batchsize2  | 5316                   | 3.85           | 11.15           |
  | resnet34+batchsize16 | 5316                   | 5.49           | 14.41           |
  | resnet50+batchsize2  | 5316                   | 6.54           | 15.78           |
  | resnet101+batchsize2 | 5316                   | 9.54           | 20.21           |
  | resnet152+batchsize2 | 5316                   | 12             | 24.94           |
  | resnet18+batchsize4  | 5316                   | 3.86           | 9.32            |
  | resnet18+batchsize8  | 5316                   | 3.84           | 8.50            |
  | resnet18+batchsize16 | 5316                   | 3.86           | 8.35            |
  | resnet18+batchsize32 | 5316                   | -              | -               |

  

测试结果

- 精度
- 速度：0.46sec/frame (10 density)
- 占用资源
  - 1631MB GPU

### 2. 私有数据集

#### CTracker: resnet50-epoch200-batch_size8

| 指标        | 训练集 | 测试集-bee0009 | 测试集-bee0010 |
| ----------- | ------ | -------------- | -------------- |
| frame       | 750    | 250            | 250            |
| labels      | 4624   | 1973           | 2482           |
| label/frame | 6.2    | 7.9            | 9.9            |

GPU memory：10.24GB；CPU memory：12GB


| Sequence | Hz    | IDF1$\uparrow$ | IDP$\uparrow$ | IDR$\uparrow$ | Rcll$\uparrow$ | Prcn$\uparrow$ | GT   | MT$\uparrow$ | PT   | ML   | FP$\downarrow$ | FN$\downarrow$ | IDs$\downarrow$ | FM$\downarrow$ | MOTA$\uparrow$ | MOTP$\uparrow$ | IDt  | IDa  | IDm  |
| -------- | ----- | -------------- | ------------- | ------------- | -------------- | -------------- | ---- | ------------ | ---- | ---- | -------------- | -------------- | --------------- | -------------- | -------------- | -------------- | ---- | ---- | ---- |
| bee0009  | 13.15 | 48.70%         | 69.70%        | 37.50%        | 53.30%         | 99.20%         | 19   | 9            | 7    | 3    | 8              | 921            | 67              | 66             | 49.50%         | 83.20%         | 2    | 64   | 0    |
| bee0010  | 8.06  | 43.70%         | 59.60%        | 34.50%        | 55.50%         | 95.90%         | 34   | 9            | 20   | 5    | 59             | 1105           | 108             | 106            | 48.80%         | 78.70%         | 4    | 103  | 2    |

| Sequence | Hz    | IDF1   | IDP    | IDR    | Rcll   | Prcn   | GT   | MT   | PT   | ML   | FP   | FN   | IDs  | FM   | MOTA   | MOTP   | IDt  | IDa  | IDm  |
| -------- | ----- | ------ | ------ | ------ | ------ | ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ | ---- | ---- | ---- |
| bee0009  | 13.15 | 48.30% | 69.60% | 37.00% | 53.00% | 99.60% | 19   | 9    | 7    | 3    | 4    | 928  | 71   | 66   | 49.20% | 83.20% | 2    | 69   | 0    |

##### 问题分析

1. IDF1和MOTA都很低，说明检测的效果很差
   - $MOTA=1-\frac{\sum(FP+FN+IDs)}{\sum(GT)}=1-\frac{8+921+67}{1973}=0.495$
     - FN占了一半，也就是说，漏检了50%左右，**综合评判，是漏检的问题**
       - 原因1（数据集不足）：用于检测器训练的数据不足：只有750frame, 4624 labels用于整个CTracker；而YOLOv5+deepsort在此基础上加了**26834 labels**用于检测器训练
         - 增加bee0010作为训练集，测试bee0009的效果，发现还下降了一点...
       - 原因2（训练问题）：anchor多样性：原文的anchor固定只有1个，是用于行人跟踪的，可能比较合适；但我们这里蜜蜂会有不同姿态角度，所以应该加入不同大小anchor
2. MOTP很高，说明检测器的定位精度是高的
   - $MOTP=\frac{\sum overlap\_rate}{\sum matchs}$，即每个成功匹配的bbox与gt的重叠度



---

#### YOLOv5m+deep128x64

检测器：3545 frame, 26834 labels

跟踪器：4624 labels

| Sequence | Hz | IDF1$\uparrow$ | IDP$\uparrow$ | IDR$\uparrow$ | Rcll$\uparrow$ | Prcn$\uparrow$ | GT   | MT$\uparrow$ | PT   | ML   | FP$\downarrow$ | FN$\downarrow$ | IDs$\downarrow$ | FM$\downarrow$ | MOTA$\uparrow$ | MOTP$\uparrow$ | IDt  | IDa  | IDm  |
| ------- | ------ | ------ | ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ | ---- | ---- | ---- | ---- | ---- | ---- |
|bee0009 |15.33 |81.40% |83.30% |79.60% |93.20% |97.50% |19 |10 |3 |6 |47 |134 |15 |16 |90.10% |80.90% |14 |2 |4 |
|bee0010 |13.03 |68.40% |69.30% |67.50% |90.00% |92.50% |34 |15 |11 |8 |182 |247 |44 |20 |80.90% |80.10% |33 |8 |10 |




## 坑

1. 库要导入正确
   - numpy=1.16.0
2. 数据集要下正确
   - [MOT17Det](https://motchallenge.net/data/MOT17Det/)



## 训练自己的数据集

### 环境

创世云服务器-ubuntu-002

- RTX 3090 [2张卡]
- NVIDIA-SMI 470
- cuda 11.0
- cudnn 8.0.5
- pytorch 1.7.0
- python 3.7.11

---



### 训练参数1

**RTX 3090 [2张卡]**

| Parameters            | datasize(nums) [1080p] | GPU memory(GB) | time/epoch(min) |
| --------------------- | ---------------------- | -------------- | --------------- |
| resnet34+batchsize16  | 5316                   | 10.11          | 1.49            |
| resnet50+batchsize32  | 5316                   | 41.50          | 1.2             |
| resnet50+batchsize64  | 5316                   | -              | -               |
| resnet101+batchsize16 | 5316                   | 29.00          | 1.18            |
| resnet101+batchsize32 | 5316                   | -              | -               |
| resnet152+batchsize16 | 5316                   | 37.25          | 1.2             |

---

RTX 2080 [1张卡]

| Parameters           | datasize(nums) [1080p] | GPU memory(GB) | time/epoch(min) |
| -------------------- | ---------------------- | -------------- | --------------- |
| resnet34+batchsize16 | 5316                   | 10.11          | 1.49            |



### 测试性能

#### public detection + tracking

| Sequence | Hz$\uparrow$ | IDF1$\uparrow$ | IDP$\uparrow$ | IDR$\uparrow$ | Rcll$\uparrow$ | Prcn$\uparrow$ | GT   | MT$\uparrow$ | PT   | ML   | FP$\downarrow$ | FN$\downarrow$ | IDs$\downarrow$ | FM$\downarrow$ | MOTA$\uparrow$ | MOTP$\uparrow$ | IDt  | IDa  | IDm  |
| -------- | ------------ | -------------- | ------------- | ------------- | -------------- | -------------- | ---- | ------------ | ---- | ---- | -------------- | -------------- | --------------- | -------------- | -------------- | -------------- | ---- | ---- | ---- |
| bee0009  | 3            | 99.40%         | 99.60%        | 99.10%        | 99.50%         | 100.00%        | 19   | 18           | 0    | 1    | 0              | 10             | 3               | 1              | 99.30%         | 100.00%        | 3    | 0    | 0    |
