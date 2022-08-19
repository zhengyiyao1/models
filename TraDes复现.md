# 1 配环境

## 1.1 深度学习环境

### 1）RTX 3090

- ubuntu 18
- python 3.6
- cuda 11.1
  - [卸载cuda 11.0，安装cuda11.1](https://www.cxyzjd.com/article/u012372401/117251997)
- nvidia-smi：470.94

### 2）RTX 1080Ti

- ubuntu 18
- python 3.6
- cuda 10.2
- nvidia-smi：460

## 1.2 代码依赖

### 1）RTX 3090

```
conda create -n trades
conda activate trades
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
cd ${trades_ROOT}
pip install cython
pip install -r requirements.txt
```

**\*注意：3090只能cuda11.1，torch1.7.1，其他都不行！**

### 2）RTX 1080Ti

```
conda create -n trades
conda activate trades
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
cd ${trades_ROOT}
pip install cython
pip install -r requirements.txt
```

### 3）Geforce 2080Ti

```
conda create -n trades python=3.6
conda activate trades
conda install pytorch=1.3.1 torchvision=0.4.2 cudatoolkit=10.0.130 -c pytorch
pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
cd ${trades_ROOT}
pip install cython
pip install -r requirements.txt
cd $TraDeS_ROOT/src/lib/model/networks/DCNv2
. make.sh
```

### 4）TITAN Xp

```
conda create -n trades
conda activate trades
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
cd ${trades_ROOT}
pip install cython
pip install -r requirements.txt
cd $TraDeS_ROOT/src/lib/model/networks/DCNv2
. make.sh
```

## 1.3 其他依赖 (不针对2080Ti)

- We use [DCNv2_pytorch_1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7) in our backbone network (pytorch_1.7 branch). Previous versions can be found in [DCNv2](https://github.com/CharlesShang/DCNv2).

```
cd ./src/lib/model/networks/
mv DCNv2 DCNv2_ori # 自带的是cuda10的，所以需要替换下
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
```

然后将`DCNv2_ori/dcn_v2.py`的`DCN_TraDeS`类，整个复制到刚git下来的`DCNv2/dcn_v2.py`里

然后执行如下命令来编译：

```
cd DCNv2
./make.sh
```

​	* 这里有个坑，需要cuda版本匹配才行！

- In order to run the code for demos, you also need to install [ffmpeg](https://www.ffmpeg.org/).

# 2 数据集准备

1. 在`trades_ROOT`创建一个`data`文件夹

2. 将MOT格式的数据集下载好放入`trades_ROOT/data`，数据集结构如下：

   <img src="https://s2.loli.net/2021/12/29/HGk8mj2aL9zITh7.png" alt="image-20211229100351670" style="zoom:50%;" />

3. cd到`trades_ROOT/src/tools/`下，打开拷贝`convert_mot_to_coco.py`脚本为

   - `convert_mot_to_coco_train.py`

     ```
     import os
     import numpy as np
     import json
     import cv2
     # Use the same script for MOT16
     # DATA_PATH = '../../data/mot16/'
     DATA_PATH = '../../data/BEE20/'
     OUT_PATH = DATA_PATH + 'annotations/'
     # 'train_half', 'val_half', 'train', 'test'
     SPLITS = ['train']
     # SPLITS = ['train_half', 'val_half', 'train', 'test']
     HALF_VIDEO = True
     CREATE_SPLITTED_ANN = True
     CREATE_SPLITTED_DET = False
     
     if __name__ == '__main__':
         if not os.path.exists(OUT_PATH):
         	os.makedirs(OUT_PATH, exist_ok=True)
     
         for split in SPLITS:
             data_path = DATA_PATH + (split if not HALF_VIDEO else 'train')
             print(data_path)
             out_path = OUT_PATH + '{}.json'.format(split)
             out = {'images': [], 'annotations': [],
                    'categories': [{'id': 1, 'name': 'pedestrain'}],
                    'videos': []}
             seqs = os.listdir(data_path)
             image_cnt = 0
             ann_cnt = 0
             video_cnt = 0
             global_track_id = {}
             for seq in sorted(seqs):
                 if '.DS_Store' in seq:
                     continue
                 if 'mot17' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
                     continue
                 video_cnt += 1
                 out['videos'].append({
                     'id': video_cnt,
                     'file_name': seq})
                 seq_path = '{}/{}/'.format(data_path, seq)
                 img_path = seq_path + 'img1/'
                 ann_path = seq_path + 'gt/gt.txt'
                 images = os.listdir(img_path)
                 num_images = len([image for image in images if 'jpg' in image])
                 if HALF_VIDEO and ('half' in split):
                     image_range = [0, num_images // 2] if 'train' in split else \
                         [num_images // 2 + 1, num_images - 1]
                 else:
                     image_range = [0, num_images - 1]
                 print(num_images, image_range)
                 for i in range(num_images):
                     if (i < image_range[0] or i > image_range[1]):
                         continue
                     image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                                   'id': image_cnt + i + 1,
                                   'frame_id': i + 1 - image_range[0],
                                   'prev_image_id': image_cnt + i if i > 0 else -1,
                                   'next_image_id':
                                   image_cnt + i + 2 if i < num_images - 1 else -1,
                                   'video_id': video_cnt}
                     out['images'].append(image_info)
                 print('{}: {} images'.format(seq, num_images))
                 if split != 'test':
                     det_path = seq_path + 'det/det.txt'
                     anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                     dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                     if CREATE_SPLITTED_ANN and ('half' in split):
                         anns_out = np.array([anns[i] for i in range(anns.shape[0]) if
                                              int(anns[i][0]) - 1 >= image_range[0] and
                                              int(anns[i][0]) - 1 <= image_range[1]], np.float32)
                         anns_out[:, 0] -= image_range[0]
                         gt_out = seq_path + '/gt/gt_{}.txt'.format(split)
                         fout = open(gt_out, 'w')
                         for o in anns_out:
                             fout.write(
                                 '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                     int(o[0]), int(o[1]), int(o[2]), int(
                                         o[3]), int(o[4]), int(o[5]),
                                     int(o[6]), int(o[7]), o[8]))
                         fout.close()
                     if CREATE_SPLITTED_DET and ('half' in split):
                         dets_out = np.array([dets[i] for i in range(dets.shape[0]) if
                                              int(dets[i][0]) - 1 >= image_range[0] and
                                              int(dets[i][0]) - 1 <= image_range[1]], np.float32)
                         dets_out[:, 0] -= image_range[0]
                         det_out = seq_path + '/det/det_{}.txt'.format(split)
                         dout = open(det_out, 'w')
                         for o in dets_out:
                             dout.write(
                                 '{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                                     int(o[0]), int(o[1]), float(o[2]), float(
                                         o[3]), float(o[4]), float(o[5]),
                                     float(o[6])))
                         dout.close()
     
                     print(' {} ann images'.format(int(anns[:, 0].max())))
                     for i in range(anns.shape[0]):
                         frame_id = int(anns[i][0])
                         if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
                             continue
                         track_id = int(anns[i][1])
                         cat_id = int(anns[i][7])
                         ann_cnt += 1
                         if not ('15' in DATA_PATH):
                             if not (float(anns[i][8]) >= 0.25):
                                 continue
                             if not (int(anns[i][6]) == 1):
                                 continue
                             if (int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]):  # Non-person
                                 continue
                             if (int(anns[i][7]) in [2, 7, 8, 12]):  # Ignored person
                                 category_id = -1
                             else:
                                 category_id = 1
                         else:
                             category_id = 1
                         identity = '{}_{}'.format(video_cnt, track_id)
                         if identity not in global_track_id:
                             if global_track_id:
                                 global_track_id.update(
                                     {identity: global_track_id[list(global_track_id)[-1]] + 1})
                             else:
                                 global_track_id.update({identity: 1})
                         ann = {'id': ann_cnt,
                                'category_id': category_id,
                                'image_id': image_cnt + frame_id,
                                'track_id': track_id,
                                'bbox': anns[i][2:6].tolist(),
                                'conf': float(anns[i][6]),
                                'global_track_id': global_track_id[identity],
                                'iscrowd': 0}
                         #  ann['bbox'] x1 y2 w h
                         ann['area'] = ann['bbox'][2] * ann['bbox'][3]
                         out['annotations'].append(ann)
                 image_cnt += num_images
             print('total # identities:', len(list(global_track_id)))
             print('loaded {} for {} images and {} samples'.format(
                 split, len(out['images']), len(out['annotations'])))
             json.dump(out, open(out_path, 'w'))
     ```
   
   - `convert_mot_to_coco_test.py`
   
   ```
   import os
   import numpy as np
   import json
   import cv2
   # Use the same script for MOT16
   # DATA_PATH = '../../data/mot16/'
   DATA_PATH = '../../data/BEE20/'
   OUT_PATH = DATA_PATH + 'annotations/'
   # 'train_half', 'val_half', 'train', 'test'
   SPLITS = ['test']
   HALF_VIDEO = True
   CREATE_SPLITTED_ANN = True
   CREATE_SPLITTED_DET = False
   
   if __name__ == '__main__':
   	if not os.path.exists(OUT_PATH):
           os.makedirs(OUT_PATH, exist_ok=True)
   
       for split in SPLITS:
           data_path = DATA_PATH + (split if not HALF_VIDEO else 'test')
           print(data_path)
           out_path = OUT_PATH + '{}.json'.format(split)
           out = {'images': [], 'annotations': [],
                  'categories': [{'id': 1, 'name': 'pedestrain'}],
                  'videos': []}
           seqs = os.listdir(data_path)
           image_cnt = 0
           ann_cnt = 0
           video_cnt = 0
           global_track_id = {}
           for seq in sorted(seqs):
               if '.DS_Store' in seq:
                   continue
               if 'mot17' in DATA_PATH and (split != 'train' and not ('FRCNN' in seq)):
                   continue
               video_cnt += 1
               out['videos'].append({
                   'id': video_cnt,
                   'file_name': seq})
               seq_path = '{}/{}/'.format(data_path, seq)
               img_path = seq_path + 'img1/'
               ann_path = seq_path + 'gt/gt.txt'
               images = os.listdir(img_path)
               num_images = len([image for image in images if 'jpg' in image])
               if HALF_VIDEO and ('half' in split):
                   image_range = [0, num_images // 2] if 'train' in split else \
                       [num_images // 2 + 1, num_images - 1]
               else:
                   image_range = [0, num_images - 1]
               print(num_images, image_range)
               for i in range(num_images):
                   if (i < image_range[0] or i > image_range[1]):
                       continue
                   image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                                 'id': image_cnt + i + 1,
                                 'frame_id': i + 1 - image_range[0],
                                 'prev_image_id': image_cnt + i if i > 0 else -1,
                                 'next_image_id':
                                 image_cnt + i + 2 if i < num_images - 1 else -1,
                                 'video_id': video_cnt}
                   out['images'].append(image_info)
               print('{}: {} images'.format(seq, num_images))
               if split != 'test':
                   det_path = seq_path + 'det/det.txt'
                   anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                   dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                   if CREATE_SPLITTED_ANN and ('half' in split):
                       anns_out = np.array([anns[i] for i in range(anns.shape[0]) if
                                            int(anns[i][0]) - 1 >= image_range[0] and
                                            int(anns[i][0]) - 1 <= image_range[1]], np.float32)
                       anns_out[:, 0] -= image_range[0]
                       gt_out = seq_path + '/gt/gt_{}.txt'.format(split)
                       fout = open(gt_out, 'w')
                       for o in anns_out:
                           fout.write(
                               '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                   int(o[0]), int(o[1]), int(o[2]), int(
                                       o[3]), int(o[4]), int(o[5]),
                                   int(o[6]), int(o[7]), o[8]))
                       fout.close()
                   if CREATE_SPLITTED_DET and ('half' in split):
                       dets_out = np.array([dets[i] for i in range(dets.shape[0]) if
                                            int(dets[i][0]) - 1 >= image_range[0] and
                                            int(dets[i][0]) - 1 <= image_range[1]], np.float32)
                       dets_out[:, 0] -= image_range[0]
                       det_out = seq_path + '/det/det_{}.txt'.format(split)
                       dout = open(det_out, 'w')
                       for o in dets_out:
                           dout.write(
                               '{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                                   int(o[0]), int(o[1]), float(o[2]), float(
                                       o[3]), float(o[4]), float(o[5]),
                                   float(o[6])))
                       dout.close()
   
                   print(' {} ann images'.format(int(anns[:, 0].max())))
                   for i in range(anns.shape[0]):
                       frame_id = int(anns[i][0])
                       if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
                           continue
                       track_id = int(anns[i][1])
                       cat_id = int(anns[i][7])
                       ann_cnt += 1
                       if not ('15' in DATA_PATH):
                           if not (float(anns[i][8]) >= 0.25):
                               continue
                           if not (int(anns[i][6]) == 1):
                               continue
                           if (int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]):  # Non-person
                               continue
                           if (int(anns[i][7]) in [2, 7, 8, 12]):  # Ignored person
                               category_id = -1
                           else:
                               category_id = 1
                       else:
                           category_id = 1
                       identity = '{}_{}'.format(video_cnt, track_id)
                       if identity not in global_track_id:
                           if global_track_id:
                               global_track_id.update(
                                   {identity: global_track_id[list(global_track_id)[-1]] + 1})
                           else:
                               global_track_id.update({identity: 1})
                       ann = {'id': ann_cnt,
                              'category_id': category_id,
                              'image_id': image_cnt + frame_id,
                              'track_id': track_id,
                              'bbox': anns[i][2:6].tolist(),
                              'conf': float(anns[i][6]),
                              'global_track_id': global_track_id[identity],
                              'iscrowd': 0}
                       #  ann['bbox'] x1 y2 w h
                       ann['area'] = ann['bbox'][2] * ann['bbox'][3]
                       out['annotations'].append(ann)
               image_cnt += num_images
           print('total # identities:', len(list(global_track_id)))
           print('loaded {} for {} images and {} samples'.format(
               split, len(out['images']), len(out['annotations'])))
           json.dump(out, open(out_path, 'w'))
   
   ```
   
   - 在`trades_ROOT/data/BEE20`下，创建`annotations`文件夹
   
   分别运行`convert_mot_to_coco_train.py`脚本和`convert_mot_to_coco_test.py`
   
   - 即在`trades_ROOT/data/BEE20/annotations`下会生成json文件

至此，数据集准备完毕

# 3 训练与测试

## 3.1 demo

cd到`./models/`下，打开`readme.md`，里面有个硬盘链接，打开后有所有model。

先下载预训练的model，比如2D跟踪就是``

然后将model放在`./models/`下，

再cd到`./src`，执行：

```
python demo.py --tasktracking --dataset mot --load_model ../models/mot_half.pth --demo ../videos/mot_mini.mp4 --pre_hm --ltrb_amodal --pre_thresh 0.5 --track_thresh 0.4 --inference --clip_len 3 --trades --save_video --resize_video --input_h 544 --input_w 960
```

![image-20211229020001928](https://s2.loli.net/2021/12/29/of268YbyRdxrSsq.png)

## 3.2 训练

cd到`trades_root/src/lib/dataset/datasets/`，打开`mot.py`脚本，

在`def __init__(self, opt, split):`函数中，找到如下段落

```
data_dir = os.path.join(opt.data_dir, 'mot{}'.format(self.year))

if opt.dataset_version in ['17trainval', '17test']:
            ann_file = '{}.json'.format('train' if split == 'train' else
                                        'test')
        elif opt.dataset_version == '17halftrain':
            ann_file = '{}.json'.format('train_half')
        elif opt.dataset_version == '17halfval':
            ann_file = '{}.json'.format('val_half')
        img_dir = os.path.join(data_dir, '{}'.format(
            'test' if 'test' in self.dataset_version else 'train'))
```

替换为如下：

```
data_dir = os.path.join(opt.data_dir, 'BEE{}'.format(self.year))

if opt.dataset_version in ['20trainval', '20test']:
            ann_file = '{}.json'.format('train' if split == 'train' else
                                        'test')
        elif opt.dataset_version == '20halftrain':
            ann_file = '{}.json'.format('train_half')
        elif opt.dataset_version == '20halfval':
            ann_file = '{}.json'.format('val_half')
        img_dir = os.path.join(data_dir, '{}'.format(
            'test' if 'test' in self.dataset_version else 'train'))
```

cd到`./experiments`下，创建一个`bee20_train.sh`脚本，内容如下

```
CUDA_VISIBLE_DEVICES=0,1 python main.py tracking \
    --exp_id bee20_fulltrain \
    --dataset mot \
    --dataset_version 20trainval \
    --pre_hm --ltrb_amodal --same_aug \
    --hm_disturb 0.05 \
    --lost_disturb 0.4 \
    --fp_disturb 0.1 \
    --gpus 0,1 \
    --num_epochs 100 \
    --save_point 60,70,80,90 \
    --load_model ../models/crowdhuman.pth \
    --clip_len 3 \
    --max_frame_dist 10  \
    --batch_size 8 \
    --trades
```

然后cd到`trades_root`下，运行

```
sh experiments/bee20_train.sh
```

如果报错：

```
subprocess.CalledProcessError: Command '['git', 'describe']' returned non-zero exit status 128
```

则在`trades_root`创建一个git，即

```
git init
```

重新运行shell命令即可

成功运行就开始训练了：

![image-20211229102751010](https://s2.loli.net/2021/12/29/NEKuBLIqJ1oXiAw.png)

并会在`trades_root`根目录下生成一个`./exp`文件夹，存放训练结果

## 3.3 测试

### model

训练完成后，会在`CODE_ROOT`下生成一个`exp/tracking/`文件夹，里面放着你相应数据集版本的训练结果，比如版本是`bee20_half`，则会有一个`bee20_half`文件夹，里面保存了训练过程中生成的model，以及训练日志。

我们只需要最后一个model：`model_last.ph`。（其他都可以删掉，不然太占空间了）

### 创建shell脚本

cd到`./experiments`下，创建一个`bee20_test.sh`脚本，内容如下

```
cd src
# test
CUDA_VISIBLE_DEVICES=0 python test.py tracking \
    --exp_id bee20_fulltrain \
    --dataset mot \
    --test_dataset mot \
    --dataset_version 20test \
    --pre_hm \
    --ltrb_amodal \
    --inference \
    --load_model ../exp/tracking/bee20_half/model_last.pth \
    --gpus 0 \
    --clip_len 3 \
    --trades
cd ..

```

打开`src/lib/dataset/datasets/mot.py`，找到`run_eval`函数，替换为如下：

```
def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    gt_type_str = '{}'.format(
    '_train_half' if '20halftrain' in self.opt.dataset_version
    else '_val_half' if '20halfval' in self.opt.dataset_version
    else '')
    gt_type_str = '_val_half' if self.year in [16, 19] else gt_type_str
    gt_type_str = '--gt_type {}'.format(gt_type_str) if gt_type_str != '' else \
    ''
    os.system('python tools/eval_motchallenge.py ' +
    '../data/BEE{}/{}/ '.format(self.year, 'test') +
    '{}/results_mot{}/ '.format(save_dir, self.dataset_version) +
    gt_type_str + ' --eval_official')
```

然后，在`trades_root`目录下，运行

```
sh experiments/bee20_test.sh
```

就开始测试了

![image-20211229135342436](https://s2.loli.net/2021/12/29/UyQHdl4Jk6Ohe5C.png)

测试完成后，会在`exp/tracking/`目录下生成相应的结果文件。



# 4 性能分析

## 4.1 训练情况

| Parameters          | datasize(nums) [1080p] | GPU memory(GB) | time/epoch(min) |
| ------------------- | ---------------------- | -------------- | --------------- |
| epoch70+batchsize2  | 5316/2                 | 6.72           | 3               |
| epoch100+batchsize8 | 5316                   | 29.1           | 2.05            |

## 4.2 测试效果

GPU: 1997MB

| Parameters                   | Sequence | Hz$\uparrow$ | IDF1$\uparrow$ | IDP$\uparrow$ | IDR$\uparrow$ | Rcll$\uparrow$ | Prcn$\uparrow$ | GT   | MT$\uparrow$ | PT   | ML   | FP$\downarrow$ | FN$\downarrow$ | IDs$\downarrow$ | FM$\downarrow$ | MOTA$\uparrow$ | MOTP$\uparrow$ | IDt  | IDa  | IDm  |
| ---------------------------- | -------- | ------------ | -------------- | ------------- | ------------- | -------------- | -------------- | ---- | ------------ | ---- | ---- | -------------- | -------------- | --------------- | -------------- | -------------- | -------------- | ---- | ---- | ---- |
| epoch70+batchsize2+halftrain | bee0009  | 12.04        | 74.90%         | 75.60%        | 74.20%        | 95.90%         | 97.70%         | 19   | 17           | 2    | 0    | 44             | 80             | 32              | 8              | 92.10%         | 82.3%          | 15   | 19   | 5    |
| epoch70+batchsize8+fulltrain | bee0009  | 14.08        | 79.00%         | 79.60%        | 78.40%        | 97.50%         | 99.10%         | 19   | 19           | 0    | 0    | 17             | 49             | 31              | 7              | 95.10%         | 82.6%          | 16   | 17   | 4    |

### 1）问题分析
