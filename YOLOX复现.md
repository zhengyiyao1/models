# 1 配环境

## 1.1 深度学习环境

### 1）RTX 3090

- ubuntu 18
- python 3.6
- cuda 11.1
  - [卸载cuda 11.0，安装cuda11.1](https://www.cxyzjd.com/article/u012372401/117251997)
- nvidia-smi：470.94

## 1.2 代码依赖

### 1）RTX 3090

```
conda create -n yolox
conda activate yolox
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -r requirements.txt
pip install matplotlib # 需要补充的
pip3 install -v -e .  # or  python3 setup.py develop
```

**\*注意：3090只能cuda11.1，torch1.7.1，其他都不行！**

## 1.3 安装[pycocotools](https://github.com/cocodataset/cocoapi)

```
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## 1.4 pre-trained model安装

#### Standard Models.

|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8| [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.7 | 48.0 | 11.1 |63.7 | 185.3 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |

<details>
<summary>Legacy models</summary>

|Model |size |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |39.6      |9.8     |9.0 | 26.8 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.4      |12.3     |25.3 |73.8| [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |50.0  |14.5 |54.2| 155.6 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640  |**51.2**      | 17.3 |99.1 |281.9 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.4      | 11.1 |63.7 | 185.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth) |

#### Light Models.

|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/nano.py) |416  |25.8  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) |

# 2 数据集准备【参考Bytetrack】

# 3 训练与测试

## 3.1 demo

在`<yolox_root>`下新建一个`demo.sh`，并复制如下内容

```
python tools/demo.py image \
    -n yolox-s \
    -c ./models/yolox_s.pth \
    --path assets/dog.jpg \
    --conf 0.25 \
    --nms 0.45 \
    --tsize 640 \
    --save_result \
    --device gpu \
    --fp16
```

在`<yolox_root>`下运行`sh demo.sh`

![image-20220112102948757](https://s2.loli.net/2022/01/12/BkqGVOmh9T2PXx8.png)

## 3.2 训练

### 01 修改配置文件

将`tools/train.py`移动到`yolox_root`下；并打开`train.py`，修改`make_parser`函数里的相关参数，主要为

```python
parser.add_argument("-expn", "--experiment-name", type=str, default="BEE22v1-yolox-s")
parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")
```



打开`exps/example/custom/yolox_s.py`文件，将class替换，为如下内容：

```
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/BEE22" # 数据集名称
        self.train_ann = "train.json" # 训练数据
        self.val_ann = "test.json" # 测试数据

        self.num_classes = 1 # 类别数量，改为1
        self.max_epoch = 100 # epoch数量
        self.data_num_workers = 4
        self.eval_interval = 10 # 评估的间隔
```



打开`yolox/data/datasets/coco.py`，修改`class COCODataset`的内容，

- 把`__init__()`函数里的`name`改为`"train"`



打开`yolox/exp/yolox_base.py`文件，修改`get_eval_loader`的内容，

- 把下面第4行的name修改为`test`

    ```python
    valdataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
                name="test" if not testdev else "test", # 修改这里为test
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy),
            )
    ```

### 02 开始训练

在`yolox_root`下创建一个`train.sh`，内容如下：

```
python train.py \
    -f exps/example/custom/yolox_s.py # config file
    -d 0 \ # device
    -b 2 \ # batch size
    --fp16 \
    -o \
    -c models/yolox_s.pth # pre-train model
```



## 3.3 测试





# 4 性能分析

## 4.1 训练情况

| Parameters          | datasize(nums) [1080p] | GPU memory(GB) | time/epoch(min) |
| ------------------- | ---------------------- | -------------- | --------------- |
| epoch70+batchsize2  | 5316/2                 | 6.72           | 3               |
| epoch100+batchsize8 | 5316                   | 29.1           | 2.05            |

## 4.2 测试效果

### 在预训练model测试蜜蜂数据集

显存占用：\~1800MiB

测试结果见`./YOLOX_test.xlsx`

结论

1. float

   - fp32比fp16快

2. model

   - yolox_x不能达到real-time
   - 其他model都可以FPS>30

3. img size：推测原因是fp越大model更精确，此时imgsize越大，自然更慢；fp小的时候，由于model不准，imgsize大有利于推理bbox的个数更准确，所以速度更快

   - fp16下，950x590比640x640快0.25ms
   - fp32下，950x590比640x640慢0.27ms

   

### 在蜜蜂model测试蜜蜂数据集

| Parameters                   | Sequence | Hz$\uparrow$ | IDF1$\uparrow$ | IDP$\uparrow$ | IDR$\uparrow$ | Rcll$\uparrow$ | Prcn$\uparrow$ | GT   | MT$\uparrow$ | PT   | ML   | FP$\downarrow$ | FN$\downarrow$ | IDs$\downarrow$ | FM$\downarrow$ | MOTA$\uparrow$ | MOTP$\uparrow$ | IDt  | IDa  | IDm  |
| ---------------------------- | -------- | ------------ | -------------- | ------------- | ------------- | -------------- | -------------- | ---- | ------------ | ---- | ---- | -------------- | -------------- | --------------- | -------------- | -------------- | -------------- | ---- | ---- | ---- |
| epoch70+batchsize2+halftrain | bee0009  | 12.04        | 74.90%         | 75.60%        | 74.20%        | 95.90%         | 97.70%         | 19   | 17           | 2    | 0    | 44             | 80             | 32              | 8              | 92.10%         | 82.3%          | 15   | 19   | 5    |
| epoch70+batchsize8+fulltrain | bee0009  | 14.08        | 79.00%         | 79.60%        | 78.40%        | 97.50%         | 99.10%         | 19   | 19           | 0    | 0    | 17             | 49             | 31              | 7              | 95.10%         | 82.6%          | 16   | 17   | 4    |

### 1）问题分析