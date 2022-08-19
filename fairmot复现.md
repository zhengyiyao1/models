# 1 配环境

## 1.1 深度学习环境

- ubuntu 18
- python 3.8
- cuda 10.2
- nvidia-smi：460.32.03

## 1.2 代码依赖

### 1）RTX 3090

```
conda create -n FairMOT
conda activate FairMOT
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
cd ${FAIRMOT_ROOT}
pip install cython
pip install -r requirements.txt
```

**\*注意：3090只能cuda11.1，torch1.7.1，其他都不行！**

### 1）RTX 1080

```
conda create -n FairMOT
conda activate FairMOT
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
cd ${FAIRMOT_ROOT}
pip install cython
pip install -r requirements.txt
```

## 1.3 其他依赖

- We use [DCNv2_pytorch_1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7) in our backbone network (pytorch_1.7 branch). Previous versions can be found in [DCNv2](https://github.com/CharlesShang/DCNv2).

```
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
```

​	* 这里有个坑，需要cuda版本匹配才行！

- In order to run the code for demos, you also need to install [ffmpeg](https://www.ffmpeg.org/).

# 2 数据集准备

1. 将MOT格式的数据集下载好放在`DATA_ROOT`,例如"xxxx/MOT_benchmark/fairmot/",

2. 创建一个`images`文件夹

3. 将MOT数据集自己分成`train`和`test`两个文件，移动到`images`文件夹内

   ![image-20211208121620029](https://s2.loli.net/2021/12/08/TNXWDlmJRG8Bcqp.png)

4. 将`preprocess_data`脚本文件夹移动到`DATA_ROOT`下，里面有两个脚本

5. 在`preprocess_data`文件夹下，运行`gen_labels_with_ids.py`，运行前，先做如下修改

   - line 5：

     ```
     base_path = "../images/train"
     ```

   - line 12：
   
     ```
     if(d.startswith("bee")): # 这个bee是指前缀，可以根据数据集命名来改的，主要是为了获取文件夹下的所有序列
     ```
   
   - line 21：
   
     ```
     os.rename(os.path.join(t_path, 'img1'), os.path.join(t_path, 'images')) # 因为MOT的数据集都是img1来命名图片文件夹，这里要改成images
     ```
   
6. 运行完会在比如`images/train`下的每个序列下生成一个文件夹`labels_with_ids`，此时里面是empty的

   <img src="https://s2.loli.net/2021/12/08/UZPd5lByYrgLf4Q.png" alt="image-20211208122421541" style="zoom:50%;" />

7. cd到`CODE_ROOT/src`下，例如`xxxx/MOTmethods/fairmot/src/`

8. 运行`gen_labels_16.py`，用来在`DATA_ROOT`目录下生成label文件，运行前，先做如下修改

   - line12-13:

   ```
   seq_root = '../../../MOT_benchmark/fairmot/images/train' # 当前序列的位置
   label_root = '../../../MOT_benchmark/fairmot/labels_with_ids/train' # 待保存的label的位置
   ```

   运行完，会在`DATA_ROOT`目录下生成`labels_with_ids`文件夹；

   并且，`images/train`里的每个sequence的`labels_with_ids`里也生成相同的txt文件。

   （相当于把每个sequence下的label都搬到`DATA_ROOT`下的`labels_with_ids`里了）

   ![image-20211208123021959](https://s2.loli.net/2021/12/08/mwQoKyfiqY3xauM.png)

9. cd到`src/data/`下，运行`generate_imgpath.py`，用于生成训练/测试图片所在的绝对路径。运行前，先做如下修改

   line4-7：

   ```
   # 设定是要训练还是测试，以及sequence的根目录
   data_type = "test"
   BasePath = os.path.join(
       "/home/caoxiaoyan/MOT_benchmark/fairmot/images/%s" % data_type)
   # 可以指定数据集名称
   target_file_name = "bee_v2.%s" % data_type
   ```

   运行完，如果是`data_type="train"`，则会在`src/data/`下生成一个`bee_v2.train`，里面就是训练图片的绝对路径

   ![image-20211208123738403](https://s2.loli.net/2021/12/08/pxQazH1P9dTk68K.png)

   

10. cd到`src/lib/cfg`目录下

11. 创建一个json文件，例如`bee_v2.json`。用于指明训练or测试的文件路径（可与步骤9连起来看）

    ```
    {
      "root": "/home/caoxiaoyan/MOT_benchmark/fairmot",
      "train": {
        "bee_v2": "./data/bee_v2.train"
      },
      "test_emb": {
        "bee_v2": "./data/bee_v2.test"
      },
      "test": {
        "bee_v2": "./data/bee_v2.test"
      }
    }
    ```

12. cd到`src/lib`目录下，

13. 打开`opts.py`文件，可以设置一些超参数。如果不设置的话，可以在到时候运行的时候再设置也行

14. 至此，数据集准备完毕

# 3 训练与测试

## 3.1 训练

cd到`CODE_ROOT/experiments`下，可以创建一个shell文件，例如:`bee_train.sh`

```
cd ../src
python train.py mot \
    --exp_id bee_v2 \
    --load_model '../models/ctdet_coco_dla_2x.pth' \
    --data_cfg './lib/cfg/bee_v2.json' \
    --gpus 0 \
    --batch_size 2 \
    --num_epochs 100 \
cd ..
```

以上的参数都可以调整，调整好后，

激活conda虚拟环境，执行`sh bee_train.sh`，就可以开始训练了！

## 3.2 测试

训练完成后，会在`CODE_ROOT`下生成一个`exp/mot/`文件夹，里面放着你相应数据集版本的训练结果，比如版本是`bee_v2`，则会有一个`bee_v2`文件夹，里面保存了训练过程中生成的model，以及训练日志。

我们只需要最后一个model：`model_last.ph`。（其他都可以删掉，不然太占空间了）



接下来还要设置一下参数

cd到`src/lib/`下，打开`opts.py`，创建一个参数，来表明要测试哪个数据集，把`default=True`即可，

```
self.parser.add_argument(
            '--val_bee', default=True, help='val bee')
```

然后再cd到`src`下，打开`track.py`

在主函数那边加一个段if语句

```
if opt.val_bee:
        # =============================
        seqs_str = '''bee0010
                      bee0009
                      '''
        # =============================
        data_root = os.path.join(opt.data_dir, 'images/test')
```

其中，seqs_str的字符串表示要测试的序列，'images/test'表示测试序列的根目录



接下来就可以准备测试了，cd到`experiments`目录下，运行`test.sh`，

```
cd ../src
python track.py mot \
    --load_model ../exp/mot/bee_v2/model_last.pth \
    --conf_thres 0.6 \
    --data_dir /home/caoxiaoyan/MOT_benchmark/fairmot \
    --data_cfg ../src/lib/cfg/bee_v2.json \
    2>&1 | tee -a ../log/test.log
cd ..

```

在指定以上超参数后，就可以开始运行了

运行完成后，会在终端print出测试结果，

![image-20211208125833677](https://s2.loli.net/2021/12/08/wVd2oRknrEMQZFt.png)

另外，数据集根目录下会生成相应的结果文件`outputs`和`results`

- outputs：跟踪的结果图片or视频
- results：跟踪的结果数据

![image-20211208125908442](https://s2.loli.net/2021/12/08/9aMWVOosFk5uZ4f.png)

# 4 性能分析

## 4.1 训练情况

| Parameters          | datasize(nums) [1080p] | GPU memory(GB) | time/epoch(min) |
| ------------------- | ---------------------- | -------------- | --------------- |
| epoch100+batchsize2 | 5316                   | 9.82           | 3.27            |

## 4.2 测试效果

| Sequence | Hz$\uparrow$ | IDF1$\uparrow$ | IDP$\uparrow$ | IDR$\uparrow$ | Rcll$\uparrow$ | Prcn$\uparrow$ | GT   | MT$\uparrow$ | PT   | ML   | FP$\downarrow$ | FN$\downarrow$ | IDs$\downarrow$ | FM$\downarrow$ | MOTA$\uparrow$ | MOTP$\uparrow$ | IDt  | IDa  | IDm  |
| -------- | ---- | ------ | ------ | ------ | ------ | ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ | ---- | ---- | ---- |
| bee0009  | 18.3 | 71.00% | 80.70% | 63.40% | 77.40% | 98.50% | 19   | 7    | 5    | 7    | 23   | 445  | 18   | 57   | 75.40% | 81.60% | 2    | 12   | 1   |
| bee0010  | 17.3 | 69.80% | 77.40% | 63.60% | 81.80% | 99.60% | 34   | 8    | 14   | 12   | 9    | 452  | 57   | 69   | 79.10% | 81.30% | 18   | 28   | 6    |

### 1）问题分析

1. IDF1和MOTA有待进一步提升
   - $MOTA=1-\frac{\sum(FP+FN+IDs)}{\sum(GT)}=1-\frac{23+445+18}{1973}=0.754$
     - FN相比CTracker下降了一倍，但相比+YOLOV5+deepsort的结果，还是高了1倍左右；还是比较高的FN，从结果视频也可以看出来
2. MOTP也是需要进一步，说明检测器的定位精度是高的
   - $MOTP=\frac{\sum overlap\_rate}{\sum matchs}$，即每个成功匹配的bbox与gt的重叠度
