# MF-ID: A Benchmark and Approach for Multi-category Finegrained Intrusion Detection

## Relevant datasets
[https://pan.baidu.com/s/1u0Pz3suqge1y4L03DVOBcA?pwd=tl1g](https://pan.baidu.com/s/1IK3sl-yXa8TVcJ9TxJajrA?pwd=u7c4)

## Train custom data


1. see [how to arrange your detection dataset with yolov5](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) , then arrange your segmentation dataset same as yolo files , see data/voc.yaml:

```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: .  # dataset root dir
train: VOC/det/images/train  # train images (relative to 'path') 118287 images
val: VOC/det/images/test  # train images (relative to 'path') 5000 images
road_seg_train: VOC/seg/images/train   # road segmentation data
road_seg_val: VOC/seg/images/val
```
   
2. change the config in trainds.py and :

```
python trainds.py 
```

3. test image folder with :

```
python detectds.py
```
