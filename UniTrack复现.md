# 1 配环境

## 1.1 深度学习环境

- ubuntu 16
- python 3.8
- cuda 10.2
- nvidia-smi：465.27

## 1.2 代码依赖

下面介绍该项目环境配置的过程，需要保证用户已经安装了Git和Conda，且安装了支持CUDA10.2以上的显卡驱动。

Step1. Install unitrack.

先打开`requirements.txt`，把`opencv`和`pycocotools`注释掉，因为版本不对。

先用如下命令安装

#### RTX 3090

```
# 1. Create a conda virtual environment.
conda create -n unitrack python=3.7 -y
conda activate unitrack

# 2. Install PyTorch
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Get UniTrack
git clone https://github.com/Zhongdao/UniTrack.git
cd UniTrack

# 4. Install ohter dependency
conda install --file requirements.txt 
pip install cython_bbox
pip3 install --upgrade cython
python setup.py
```

接着补充两个包

```
pip install opencv-python
pip install pycocotools
```

# 2 数据集准备

## 2.1 准备MOT16格式数据集

1. 将自己的数据集放在`DATA_ROOT`下，如将数据集`BEE20`放在`/home/caoxiaoyan/MOT_benchmark/unitrack/`下。

2. 创建一个`images`文件夹
3. 将MOT数据集分成`train`和`test`两个文件夹，并放入`images`文件夹下

![image-20211222111510889](https://s2.loli.net/2021/12/22/nlqAJxuOkPpWswy.png)

## 2.1 数据集格式转化

1. cd到`<unitrack_home>/tools/`，拷贝一份`gen_mot16_gt.py`，并根据自己的数据集命名，如`gen_bee20_gt.py`

2. 打开`gen_bee20_gt.py`，修改前`3`条语句

   ```
   # =================================只改这部分================================
   # Modify here
   dataset_root = '/home/caoxiaoyan/MOT_benchmark/unitrack/BEE20'  # 数据集根目录
   
   # MOT16的测试集路径
   # images下是sequences
   seq_root = osp.join(dataset_root, 'images', 'test')
   
   # 创建det结果路径
   label_root = osp.join(dataset_root, 'obs', 'gt', 'test')
   # ================================================================================
   ```
   
3. 运行`gen_bee20_gt.py`，会在`DATA_ROOT`下生成一个`obs`目录，就是用于后续跟踪的det结果文件了。

4. 在`DATA_ROOT`下创建一个`seqmaps`目录，进去创建一个`BEE20-test.txt`文件，用于确定需要评估的数据集

   `BEE20-test.txt`文件里除了第一行外，每一行都是一个sequence名称，例如

   ```
   name
   bee0009
   bee0010
   ```
   
   ==注意：不能换行！==

**至此，数据集准备完毕**

# 3 模型下载

## 3.1 appearance models

可以按下表来下载

==下载后在项目的根目录下新建`weights`文件夹并将模型放入其中。==


| Pre-training Method | Architecture |Link | 
| :---: | :---: | :---: |
| ImageNet classification | ResNet-50 | torchvision |
| InsDist| ResNet-50 | [Google Drive](https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACcsSIt1_Njv7GsmsuzZ6Sta/InsDis.pth)|
| MoCo-V1| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar)|
| PCL-V1| ResNet-50 |[Google Drive](https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v1_epoch200.pth.tar)|
| PIRL| ResNet-50 | [Google Drive](https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth)|
| PCL-V2| ResNet-50 | [Google Drive](https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v2_epoch200.pth.tar)|
| SimCLR-V1| ResNet-50 |[Google Drive]()|
| MoCo-V2| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar)|
| SimCLR-V2| ResNet-50 |[Google Drive]()|
| SeLa-V2| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar)|
| InfoMin| ResNet-50 | [Google Drive](https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth)|
| BarlowTwins| ResNet-50 | [Google Drive](https://drive.google.com/file/d/1iXfAiAZP3Lrc-Hk4QHUzO-mk4M4fElQw/view?usp=sharing)|
| BYOL| ResNet-50 | [Google Drive](https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl)|
| DeepCluster-V2| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar)|
| SwAV| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar)|
| PixPro| ResNet-50 |[Google Drive](https://drive.google.com/file/d/1u172sUx-kldPvrZzZxijciBHLMiSJp46/view?usp=sharing)|
| DetCo| ResNet-50 | [Google Drive](https://drive.google.com/file/d/1ahyX8HEbLUZXS-9Jr2GIMWDEZdqWe1GV/view?usp=sharing)|
| TimeCycle| ResNet-50 |[Google Drive](https://drive.google.com/file/d/1WUYLkfowJ853RG_9OhbrKpb3r-cc-cOA/view?usp=sharing)|
| ImageNet classification | ResNet-18 |torchvision|
| Colorization + memory| ResNet-18 | [Google Drive](https://drive.google.com/file/d/1gWPRgYH70t-9uwj0EId826ZxFdosbzQv/view?usp=sharing)|
| UVC| ResNet-18 |[Google Drive](https://drive.google.com/file/d/1nl0ehS8mvE5PUBOPLQSCWtrmFmS0-dPX/view?usp=sharing)|
| CRW| ResNet-18 |[Google Drive](https://drive.google.com/file/d/1C1ujnpFRijJqVD3PV7qzyYwGSWoS9fLb/view?usp=sharing)|

## 3.2 测试自己的appearance model

需要采用resnet架构，

不然就要求model是输出8倍下采样的feature map。

可以以`models/hrnet.py`为例

# 4 训练与测试

## 4.1 Demo

### 准备config文件

参考`config`目录下的`yaml`文件，进行参数配置。

作者给了4个`yaml`文件：

1. `crw_resnet18_s3.yaml` : Self-supervised model trained with Contrastive Random Walk [1], ResNet-18 stage-3 features.
2. `imagenet_resnet18_s3.yaml`: ImageNet pre-trained model, ResNet-18 stage-3 features.
3. `crw_resnet18_s3_womotion.yaml` : Model same as 1 but motion cues are discarded in association type tasks. This way, distinctions between different representations are better highlighted and potential confounding factors are avoided.
4. `imagenet_resnet18_s3_womotion.yaml`: Model same as 2, motion cues are discared in association type tasks.

要做自己的`yaml`文件，则可以参考以上几种的方式

例如，拷贝`crw_resnet18_s3.yaml`为`crw_resnet18_s3_bee.yaml`，然后打开后者，一共`4`处要修改：

```
common:
    exp_name: crw_resnet18_s3_bee # change1: 实验名称
   
    # Model related
    model_type: crw
    remove_layers: ['layer4']
    im_mean: [0.4914, 0.4822, 0.4465]
    im_std: [0.2023, 0.1994, 0.2010]
    nopadding: False
    head_depth: -1
    resume: 'weights/crw.pth' # change2: appearance model的path
    
    # Misc
    down_factor: 8
    infer2D: True 
    workers: 4
    gpu_id: 0
    device: cuda

mot:
    obid: 'gt' # change3: det的结果路径
    mot_root: '/home/caoxiaoyan/MOT_benchmark/unitrack/BEE20' # change4: 数据集根目录
    feat_size: [4,10]
    save_videos: True
    save_images: False
    test_mot16: False
    track_buffer: 30
    min_box_area: 200
    nms_thres: 0.4
    conf_thres: 0.5
    iou_thres: 0.5
    dup_iou_thres: 0.15
    confirm_iou_thres: 0.7
    img_size: [1088, 608]
    prop_flag: False
    use_kalman: True 
    asso_with_motion: True 
    motion_lambda: 0.98
    motion_gated: True 
```

### Run

将`$UNITRACK_ROOT/test`目录下的`test_mot.py`拷贝到`$UNITRACK_ROOT`根目录下。

打开`test_mot.py`脚本，做一些修改

- 在调用main()函数前，把要测试数据集修改掉，原本是

    ```
    if not opt.test_mot16:
            seqs_str = '''MOT16-02
                          MOT16-04
                          MOT16-05
                          MOT16-09
                          MOT16-10
                          MOT16-11
                          MOT16-13
                        '''
            data_root = '{}/images/train'.format(opt.mot_root)
        else:
            seqs_str = '''MOT16-01
                         MOT16-03
                         MOT16-06
                         MOT16-07
                         MOT16-08
                         MOT16-12
                         MOT16-14'''
            data_root = '{}/images/test'.format(opt.mot_root)
        seqs = [seq.strip() for seq in seqs_str.split()]
    ```

    - 把上面的都注释掉，然后改为

    ```
    seqs_str = '''bee0009'''
    data_root = '{}/images/test'.format(opt.mot_root)
    seqs = [seq.strip() for seq in seqs_str.split()]
    ```

- 在main()函数里找到如下一行，

  ```
  dataset_config['SPLIT_TO_EVAL'] = 'train'
  ```

  - 把`train`改为`test`，修改为如下：

  ```
  dataset_config['SPLIT_TO_EVAL'] = 'test'
  ```

- 找到如下一行
  
  ```
  dataset_config['BENCHMARK'] = 'MOT16'
  ```
  
  - 把`MOT16`修改为`BEE20`
  
  ```
  dataset_config['BENCHMARK'] = 'BEE20'
  ```
  
- 并在上面的那行下面，添加一行
  
  ```
  dataset_config['SKIP_SPLIT_FOL'] = True
  ```
  

---

接下来准备开始测试了

在`$UNITRACK_ROOT`下，打开`eval.sh`脚本，

只选择`test_mot.py`作为测试命令，其他命令都注释掉，如下：

```
#!/bin/bash

EXP_NAME=$1
CFG_PATH=config/${EXP_NAME}.yaml
SMRY_ROOT=results/summary/${EXP_NAME}
mkdir -p $SMRY_ROOT

CUDA_VISIBLE_DEVICES=$2 python -u test_mot.py --config $CFG_PATH | tee results/summary/${EXP_NAME}/mot.log 2>&1
```

运行以下命令开始测试

```
# ./eval.sh $EXP_NAME $GPU_ID
./eval.sh crw_resnet18_s3_bee 0
```

如果正常运行，会在终端显示跟踪过程：

<img src="https://s2.loli.net/2021/12/21/D1xlQMuGq4a6pji.png" alt="image-20211221122940480" style="zoom:50%;" />

跟踪结果会输出到文件夹`results/summary`中。

## 4.1 训练

只有appearance model可以训练，但paper中，作者没有自己训练，只是用通用的。

## 4.2 测试

没有训练，就没有测试，可以参考上面的demo

# 5 性能分析

## 5.1 训练情况

## 5.2 测试效果

| Sequence | model              | Hz$\uparrow$ | IDF1$\uparrow$ | MOTA$\uparrow$ |
| -------- | ------------------ | ------------ | -------------- | -------------- |
| bee0009  | imagenet_resnet18  | 23           | 75.50          | 69.34          |
| bee0009  | imagenet_resnet50  | 9.9          | 76.45          | 69.64          |
| bee0009  | imagenet_resnet101 | 4.3          | 76.74          | 69.64          |



### 1）问题分析

1. IDF1和MOTA有待进一步提升
   - $MOTA=1-\frac{\sum(FP+FN+IDs)}{\sum(GT)}=1-\frac{23+445+18}{1973}=0.754$
     - FN相比CTracker下降了一倍，但相比+YOLOV5+deepsort的结果，还是高了1倍左右；还是比较高的FN，从结果视频也可以看出来
2. MOTP也是需要进一步，说明检测器的定位精度是高的
   - $MOTP=\frac{\sum overlap\_rate}{\sum matchs}$，即每个成功匹配的bbox与gt的重叠度