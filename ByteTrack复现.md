# 1 配环境

## 1.1 深度学习环境

- ubuntu 16
- python 3.6 【必须要3.6！不然det评估的时候会报错】
- cuda 10.2
- nvidia-smi：465.27

## 1.2 代码依赖

下面介绍该项目环境配置的过程，需要保证用户已经安装了Git和Conda，且安装了支持CUDA10.2以上的显卡驱动。

Step1. Install ByteTrack.

```
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install -r requirements.txt
python setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```
pip install cython; pip install pip install cython'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others

```
pip3 install cython_bbox
```

# 2 数据集准备

详解：https://blog.csdn.net/weixin_39260670/article/details/121642184

1. 将MOT格式的数据集下载好放在`DATA_ROOT=<bytetrack_home>/datasets/`下

3. 将MOT数据集自己分成`train`和`test`两个文件夹

   <img src="https://s2.loli.net/2021/12/21/tHPx8s9gFku4AJV.png" alt="image-20211221134819841" style="zoom:50%;" />

3. 到`<ByteTrack_HOME>`下，找到`tools`文件夹，运行里面的相关脚本，将MOT数据集格式改为COCO格式。例如，可参照`convert_mot20_to_coco.py`，拷贝一份，以自己数据集命名，如`convert_BEE20_to_coco.py`。然后需要对该脚本进行一个`PATH`的修改，再运行

   - 修改line 8的`DATA_PATH`为自己的`DATA_ROOT`，例如：

     ```
     DATA_PATH = '<bytetrack_home>/datasets/BEE20_tiny'
     ```

   - 修改line 27的`categories`，将name的value改为`'bee'`

     ```
     out = {'images': [], 'annotations': [], 'videos': [],
                    'categories': [{'id': 1, 'name': 'bee'}]}
     ```

   - 在`<ByteTrack_HOME>`下运行如下命令

   ```
   python3 tools/convert_BEE20_to_coco.py
   ```

   ![image-20211221135600922](https://s2.loli.net/2021/12/21/MLxY1Wi7DjwQVc9.png)

   会生成一个`annotations`文件夹，包含train/val/test的数据集信息，格式为

   - 图像信息：` {'file_name': 'MOT17-02-FRCNN/img1/000001.jpg', 'id': 1, 'frame_id': 1, 'prev_image_id': -1, 'next_image_id': 2, 'video_id': 1, 'height': 1080, 'width': 1920}`
     - id：整个训练集中的序号，从1开始。image_cnt + i + 1，其中 image_cnt 在本序列中始终保持一个值，就是一个视频序列的帧数
     - frame_id：每个视频序列中的序号，从1开始， 验证集的 frame_id 也是从 1 开始的
     - prev_image_id：整个训练集中的前一帧的序号，若是当前视频中的第一帧，则为-1
     - next_image_id：整个训练集中的前一帧的序号，若是当前视频中的最后一帧，则为-1
     - annotations： 只保留 人 的信息，其它物体的去掉。 `{'id': 3851, 'category_id': 1,'image_id': 159, 'track_id': 7, 'bbox': [1000.0, 445.0, 36.0, 99.0], 'conf': 1.0, 'iscrowd': 0, 'area': 3564.0}`
       - id：对象 ID，这个 object 在整个数据集中的序号
         category_id：根据gt.txt文件中，原本类别是非静态的行人为1，其余人是-1，其他事物过滤
       - image_id：image_cnt + frame_id，其中 image_cnt 在本序列中始终保持一个值，就是一个视频序列的帧数
       - track_id：轨迹序号，就是一个人在N帧中的同一个序号。
       - bbox：目标框 [x, y, w, h]
       - conf：置信度
       - iscrowd：0
       - area：目标的面积

4. 将数据集的`train`目录系的sequence都提出到与`train`相同层级的目录下。

5. 至此，数据集准备完毕

# 3 模型下载

## 3.1 bytectrack训练的模型

使用在 CrowdHuman、MOT17、Cityperson 和 ETHZ 上训练的模型， 下载地址如下表，表中指标在MOT17训练集上测试得到。

==下载后在项目的根目录下新建`pretrained`文件夹并将模型放入其中。==

- **Standard models**

| Model                                                        | MOTA | IDF1 | IDs  | FPS  |
| ------------------------------------------------------------ | ---- | ---- | ---- | ---- |
| bytetrack_x_mot17 [[google\]](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing), [[baidu(code:ic0i)\]](https://pan.baidu.com/s/1OJKrcQa_JP9zofC6ZtGBpw) | 90.0 | 83.3 | 422  | 29.6 |
| bytetrack_l_mot17 [[google\]](https://drive.google.com/file/d/1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz/view?usp=sharing), [[baidu(code:1cml)\]](https://pan.baidu.com/s/1242adimKM6TYdeLU2qnuRA) | 88.7 | 80.7 | 460  | 43.7 |
| bytetrack_m_mot17 [[google\]](https://drive.google.com/file/d/11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun/view?usp=sharing), [[baidu(code:u3m4)\]](https://pan.baidu.com/s/1fKemO1uZfvNSLzJfURO4TQ) | 87.0 | 80.1 | 477  | 54.1 |
| bytetrack_s_mot17 [[google\]](https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing), [[baidu(code:qflm)\]](https://pan.baidu.com/s/1PiP1kQfgxAIrnGUbFP6Wfg) | 79.2 | 74.3 | 533  | 64.5 |

- **Light models**

| Model                                                        | MOTA | IDF1 | IDs  | Params(M) | FLOPs(G) |
| ------------------------------------------------------------ | ---- | ---- | ---- | --------- | -------- |
| bytetrack_nano_mot17 [[google\]](https://drive.google.com/file/d/1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX/view?usp=sharing), [[baidu(code:1ub8)\]](https://pan.baidu.com/s/1dMxqBPP7lFNRZ3kFgDmWdw) | 69.0 | 66.3 | 531  | 0.90      | 3.99     |
| bytetrack_tiny_mot17 [[google\]](https://drive.google.com/file/d/1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju/view?usp=sharing), [[baidu(code:cr8i)\]](https://pan.baidu.com/s/1jgIqisPSDw98HJh8hqhM5w) | 77.1 | 71.5 | 519  | 5.03      | 24.45    |

## 3.2 下载YOLOX作为预训练模型

下载COCO数据集的YOLOX预训练模型下载地址[model zoo](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0)。

把下载的yolox预训练模型放入文件夹：`<ByteTrack_HOME>/pretrained`.

# 4 训练与测试

## 4.1 Demo

在`ROOT_PATH`创建一个`shell`脚本`pure_track.sh`：

```
python tools/demo_track.py \
    --demo image \
    -f exps/example/mot/yolox_x_bee.py \
    -c YOLOX_outputs/yolox_x_bee/best_ckpt.pth.tar \
    --path datasets/BEE20/test/bee0009/img1 \
    --fp16 \
    --fuse \
    --save_result
```

并运行进行测试

如果正常运行，会在终端显示跟踪过程：

<img src="https://s2.loli.net/2021/12/21/D1xlQMuGq4a6pji.png" alt="image-20211221122940480" style="zoom:50%;" />

跟踪结果会输出到以下文件夹中：

```
./YOLOX_outputs/yolox_s_mix_det/track_vis/2021_12_21_12_24_22.txt
```

## 4.1 训练

cd到`<bytetrack_home>/exps/example/mot/`下，

拷贝`yolox_x_ch.py`为`yolox_x_bee.py`，并打开后者。

- 修改`get_data_loader`函数

  - 将要测试的数据集`data_dir`参数的最后字符串改为自己的数据集名称，如下`"BEE20_tiny"`
  - 将`name`赋值`'train'`

  ```
  valdataset = MOTDataset(
              data_dir=os.path.join(get_yolox_datadir(), "BEE20_tiny"),
              json_file=self.val_ann,
              img_size=self.test_size,
              name='test',
              preproc=ValTransform(
                  rgb_means=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225),
              ),
          )
  ```

- 修改`get_eval_loader`函数，

  - 将要测试的数据集`data_dir`参数的最后字符串改为自己的数据集名称，如下`"BEE20_tiny"`
  - 将`name`赋值`'train'` 【==这里是个大坑啊，因为val也是用train里的数据跑的！==】

  ```
  valdataset = MOTDataset(
              data_dir=os.path.join(get_yolox_datadir(), "BEE20_tiny"),
              json_file=self.val_ann,
              img_size=self.test_size,
              name='train',
              preproc=ValTransform(
                  rgb_means=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225),
              ),
          )
  ```


cd到`yolox/data/datasets/`目录下，打开`mot.py`

修改`MOTDataset`类的初始化函数

```
# 原始
if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
...

# 修改后
if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "BEE20_tiny")
...
```



准备开始训练，在`<bytetrack_home>`目录下运行如下命令：

```
python3 -u tools/train.py \
    -f exps/example/mot/yolox_x_bee.py \
    -d 0 \ # device
    -b 1 \ # batchsize
    --fp16 \
    -o \
    -c pretrained/yolox_x.pth.tar # pretrained det model
```

如果正常，则会在终端打印如下信息：

![image-20211221164836947](https://s2.loli.net/2021/12/21/coYNaymPOw49dWe.png)

并会在`<bytetrack_home>YOLOX_outputs`目录下生成相应的结果模型，以及train log。如下：

![image-20211221165030635](https://s2.loli.net/2021/12/21/vRm2cpH53qPfQAw.png)

## 4.2 测试

cd到`<bytetrack_home>/exps/example/mot/`下，

拷贝`yolox_x_ch.py`为`yolox_x_bee_test.py`，并打开后者。

- 修改class `Exp`的初始化函数的一个变量`self.val_ann = "test.json"`

- 修改`get_data_loader`函数

  - 将要测试的数据集`data_dir`参数的最后字符串改为自己的数据集名称，如下`"BEE20_tiny"`
  - 将`name`赋值`'test'`

  ```
  valdataset = MOTDataset(
              data_dir=os.path.join(get_yolox_datadir(), "BEE20_tiny"),
              json_file=self.val_ann,
              img_size=self.test_size,
              name='test',
              preproc=ValTransform(
                  rgb_means=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225),
              ),
          )
  ```

- 修改`get_eval_loader`函数，

  - 将要测试的数据集`data_dir`参数的最后字符串改为自己的数据集名称，如下`"BEE20_tiny"`
  - 将`name`赋值`'test'` 

  ```
  valdataset = MOTDataset(
              data_dir=os.path.join(get_yolox_datadir(), "BEE20_tiny"),
              json_file=self.val_ann,
              img_size=self.test_size,
              name='test',
              preproc=ValTransform(
                  rgb_means=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225),
              ),
          )
  ```

- 打开`tools/track.py`，修改gt路径

  把else逻辑下的path字符串改为自己数据集的测试集所在路径`'datasets/BEE20/test'`

  ```
  if args.mot20:
  	gtfiles = glob.glob(os.path.join('datasets/MOT20/train', 		'*/gt/gt{}.txt'.format(gt_type)))
  else:
      gtfiles = glob.glob(os.path.join('datasets/BEE20/test',
      '*/gt/gt{}.txt'.format(gt_type)))
  ```

接下来正式开始测试

```
cd <ByteTrack_HOME>
python3 tools/track.py \
	-f exps/example/mot/yolox_x_bee_test.py \
	-c YOLOX_outputs/yolox_x_bee/best_ckpt.pth.tar \ # model path
	-b 1 \
    -d 1 \
    --fp16 \
    --fuse
    --save_result
```

如果正常运行，则会在终端打印如下信息：

![image-20211222002011825](https://s2.loli.net/2021/12/22/U7mgy538XhvaIkB.png)

结果的txt文件会自动保存到model相应的文件夹下。

# 5 性能分析

## 5.1 训练情况

| Parameters          | datasize(nums) [1080p] | GPU memory(GB) | time/epoch(min) |
| ------------------- | ---------------------- | -------------- | --------------- |
| epoch100+batchsize1 | 5316                   | 9.82           | 17.33           |

## 5.2 测试效果

| Sequence | Hz$\uparrow$ | IDF1$\uparrow$ | IDP$\uparrow$ | IDR$\uparrow$ | Rcll$\uparrow$ | Prcn$\uparrow$ | GT   | MT$\uparrow$ | PT   | ML   | FP$\downarrow$ | FN$\downarrow$ | IDs$\downarrow$ | FM$\downarrow$ | MOTA$\uparrow$ | MOTP$\uparrow$ | IDt  | IDa  | IDm  |
| -------- | ---- | ------ | ------ | ------ | ------ | ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ | ---- | ---- | ---- |
| bee0009  | 18.3 | 71.00% | 80.70% | 63.40% | 77.40% | 98.50% | 19   | 7    | 5    | 7    | 23   | 445  | 18   | 57   | 75.40% | 81.60% | 2    | 12   | 1   |
| bee0010  | 17.3 | 69.80% | 77.40% | 63.60% | 81.80% | 99.60% | 34   | 8    | 14   | 12   | 9    | 452  | 57   | 69   | 79.10% | 81.30% | 18   | 28   | 6    |

| Sequence | Hz$\uparrow$ | IDF1$\uparrow$ | IDP$\uparrow$ | IDR$\uparrow$ | Rcll$\uparrow$ | Prcn$\uparrow$ | GT   | MT$\uparrow$ | PT   | ML   | FP$\downarrow$ | FN$\downarrow$ | IDs$\downarrow$ | FM$\downarrow$ | MOTA$\uparrow$ | MOTP$\uparrow$ | IDt  | IDa  | IDm  |
| -------- | ---- | ------ | ------ | ------ | ------ | ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ----- | ---- | ---- | ---- |
| bee0009  | 3 | 93.10% | 95.60% | 90.60% | 93.60% | 98.70% | 19   | 10   | 4    | 5    | 24   | 127  | 6    | 14   | 92.00% | 0.206 | 0    | 5    | 0    |

| Sequence | Hz$\uparrow$ | IDF1$\uparrow$ | MOTA$\uparrow$ |
| -------- | ------------ | -------------- | -------------- |
| bee0009  | 3            | 93.10%         | 92.00%         |



### 1）问题分析

1. IDF1和MOTA有待进一步提升
   - $MOTA=1-\frac{\sum(FP+FN+IDs)}{\sum(GT)}=1-\frac{23+445+18}{1973}=0.754$
     - FN相比CTracker下降了一倍，但相比+YOLOV5+deepsort的结果，还是高了1倍左右；还是比较高的FN，从结果视频也可以看出来
2. MOTP也是需要进一步，说明检测器的定位精度是高的
   - $MOTP=\frac{\sum overlap\_rate}{\sum matchs}$，即每个成功匹配的bbox与gt的重叠度
