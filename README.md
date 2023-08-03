# Crowd3D: Towards Hundreds of People Reconstruction from a Single Image




### [Project Page](http://cic.tju.edu.cn/faculty/likun/projects/Crowd3D) | [Paper](http://cic.tju.edu.cn/faculty/likun/projects/Crowd3D/asserts/main_paper.pdf) 


> [Crowd3D: Towards Hundreds of People Reconstruction from a Single Image]()  
> Hao Wen, Jing Huang, Huili Cui, Haozhe Lin, Yu-Kun Lai, LU FANG, Kun Li 
> CVPR 2023

**正在施工中...**


## Install

```
```

## Model Data
- Download model_data from [baidu drive](https://pan.baidu.com/s/1AqRr-NmMzyfByAZF7yOtGg?pwd=oaor). And also download the train_data if need to retrain.
- Download [largecrowd](https://pan.baidu.com/s/1XBJPD41fPysCtl1byP_8HA?pwd=c2lw) dataset if need to run evaluation code.
- Collect SMPL model files from [official website](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Rename model files (SMPL_NEUTRAL.pkl, SMPL_FEMALE.pkl, SMPL_MALE.pkl) and put them into the .../public_params/model_data/parameters/smpl/ directory.

```
./Crowd3D
├── data
│   ├── demo_data
│   ├── train_data
│   └── largecrowd
├── Crowd3DNet
│   ├── pretrained
│   └── public_params 
```

## Running the Demo
```
# 1. Set the alphapose root in 'global_setting.py'.

# 2. process image.
conda activate alphapose
python single_image/process.py --scene_image_path /the/abs/path/of/demo/image # eg. /mnt/wh/Crowd3D/data/demo_data/panda/SEQ_02_001.jpg

# 3. inference.
conda activate crowd3d
python single_image/inference.py --scene_image_path /the/abs/path/of/demo/image 
```

## LargeCrowd Dataset
We construct LargeCrowd for crowd reconstruction in a large scene. LargeCrowd is a benchmark dataset with over 100K labeled humans (2D bounding boxes, 2D keypoints, 3D ground plane and HVIPs) in 733 gigapixel-images (19200×6480) of 9 different scenes. 
![](assets/imgs/Dataset.gif)

[Baidu drive](https://pan.baidu.com/s/1XBJPD41fPysCtl1byP_8HA?pwd=c2lw) | [Data format](assets/docs/largecrowd.md)

Note: The annotations of the test set have not yet been released, due to the undetermined verification form (web page verification or code verification).

## TODO

- Code for training and testing of Crowd3D


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{Crowd3D,
  author = {Hao Wen, Jing Huang, Huili Cui, Haozhe Lin, Yu-Kun Lai, LU FANG, Kun Li},
  title = {Crowd3D: Towards Hundreds of People Reconstruction from a Single Image},
  booktitle = {CVPR},
  year={2023},
}
```
