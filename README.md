# Crowd3D: Towards Hundreds of People Reconstruction from a Single Image




### [Project Page](http://cic.tju.edu.cn/faculty/likun/projects/Crowd3D) | [Paper](http://cic.tju.edu.cn/faculty/likun/projects/Crowd3D/asserts/main_paper.pdf) 


> [Crowd3D: Towards Hundreds of People Reconstruction from a Single Image]()  
> Hao Wen, Jing Huang, Huili Cui, Haozhe Lin, Yu-Kun Lai, LU FANG, Kun Li 
> CVPR 2023


## Install
1. Install [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) and run its demo successfully, like 
```
python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth --indir examples/demo/
```

2. Install the virtual environment of Crowd3D.
```
# create virtual environment
conda create -n crowd3d python=3.9 -y
conda activate crowd3d

# install pytorch, we use pytorch=1.8, cuda=11.1
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# pip other environments by requirement.txt
pip install -r requirements.txt

# install pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

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

## Training
```
# Need to download the train_data.

cd Crowd3DNet

bash train.sh # the training log save in 'train_log/'
```

## Test
```
# Need to download the largecrowd.

# 1. process image.
conda activate alphapose
python test/preprocess.py

# 2. conda activate crowd3d
python test/inference_eval.py --use_pre_optim # Delete '--use_pre_optim' if you need to reoptimize.

# You will get the result 'predict.json', We will not release the quantitative calculation code for now. If you need to compare, you can send us your 'predict.json'.
```

## LargeCrowd Dataset
We construct LargeCrowd for crowd reconstruction in a large scene. LargeCrowd is a benchmark dataset with over 100K labeled humans (2D bounding boxes, 2D keypoints, 3D ground plane and HVIPs) in 733 gigapixel-images (19200×6480) of 9 different scenes. 
![](assets/imgs/Dataset.gif)

[Baidu drive](https://pan.baidu.com/s/1XBJPD41fPysCtl1byP_8HA?pwd=c2lw) | [Data format](assets/docs/largecrowd.md)

Note: The annotations of the test set have not yet been released, due to the undetermined verification form (web page verification or code verification).



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


Note: The base codes of Crowd3DNet are largely borrowed from [ROMP](https://github.com/Arthur151/ROMP).
