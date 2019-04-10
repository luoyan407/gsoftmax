# Gaussian-based softmax: Improving Intra-class Compactness and Inter-class Separability of Features

This repository contains the code for G-Softmax introduced in the following paper. It is built on Facebook ResNet Torch [project](https://github.com/facebook/fb.resnet.torch).

Gaussian-based softmax: Improving Intra-class Compactness and Inter-class Separability of Features ([paper](https://arxiv.org/abs/1904.04317))

Yan Luo, [Yongkang Wong](https://sites.google.com/site/yongkangwong/), [Mohan S Kankanhalli](http://www.comp.nus.edu.sg/~mohan/), and [Qi Zhao](http://www-users.cs.umn.edu/~qzhao/).


## Citation
If you find Gaussian-based Softmax useful in your research, please consider citing:

	@article{TNNLS_2019_gsoftmax,
	  title={G-softmax: Improving Intra-class Compactness and Inter-class Separability of Features},
	  author={Luo, Yan and Wong, Yongkang and Kankanhalli, Mohan S and Zhao, Qi},
	  journal={IEEE Transactions on Neural Networks and Learning Systems}
	}


## Prerequisites
1. Install Torch and required dependencies like cuDNN. See the instructions [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md) for a step-by-step guide.
2. Install probability distributions package for Torch. See the instructions [here](http://deepmind.github.io/torch-distributions/).
3. Install the requirements of Facebook ResNet Torch [project](https://github.com/facebook/fb.resnet.torch). 
4. For evaluation of multi-label classification, install [matio](https://github.com/soumith/matio-ffi.torch) by
```bash
luarocks install matio
```


## Core Components
```src/GaussianSoftMaxCriterion.lua``` is for single-label classification while ```src/MultiCrossEntropyCriterion.lua``` and ```src/MltGaussianSoftMaxCriterion.lua``` is for multi-label classification. Their usages are the same as [nn.CrossEntropyCriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.CrossEntropyCriterion).

The three files in [src](src/) folder are core files in this project. The rest of files are from [ResNet Torch project](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained) and  are modified accordingly for evaluation purposes.

## Train on Tiny ImageNet
Step 1: Download [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/)  
The directory structure of the dataset should be re-organized as 
```
/home/yluo/project/dataset/tinyimagenet
├── images
│   ├── test
│   ├── train
│   └── val
├── wnids.txt
└── words.txt
```
Step 2: Download pretrained model ResNet-101 from [ResNet Torch homepage](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)  
Step 3: To use softmax as the loss function for training, run  
```bash
export save='logs/rsn101_tinyimagenet'
mkdir -p $save
CUDA_VISIBLE_DEVICES=1,2 th main_single.lua \
	-data '/home/yluo/project/dataset/tinyimagenet/images' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 80 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'tinyimagenet' \
	-resetClassifier true \
	-nClasses 200 \
	-LR 1e-03 \
	-imgSize 224 \
	-featSize 7 \
	-nEpochs 30 \
	-lFunc 'ce' | tee $save/log.txt
```
Set **-data** to your local Tiny ImageNet images folder and set **-retrain** to the local path of the pretrained ResNet-101 model.  
To use G-softmax as the loss function for training, run
```bash
export save='logs/rsn101_tinyimagenet_gsm'
mkdir -p $save
CUDA_VISIBLE_DEVICES=1,2 th main_single.lua \
	-data '/home/yluo/project/dataset/tinyimagenet/images' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 80 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'tinyimagenet' \
	-resetClassifier true \
	-nClasses 200 \
	-LR 1e-03 \
	-imgSize 224 \
	-featSize 7 \
	-nEpochs 30 \
	-gsm_mu 0 \
	-gsm_sigma 1 \
	-gsm_scale .1 \
	-lFunc 'gsm' | tee $save/log.txt
```
where **-gsm_mu** and **-gsm_sigma** are the initial mean and stand deviation of the distribution, respectively.

## Train on MS COCO
Step 1: Download [MS COCO](http://cocodataset.org/#home)  
The directory structure of the dataset should be re-organized as 
```
/home/yluo/project/dataset/mscoco/images
├── train2014
├── train2014.t7
├── val2014
└── val2014.t7
```
Step 2: Download pretrained model ResNet-101 from [ResNet Torch homepage](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)  
Step 3: To use softmax as the loss function for training, run  
```bash
export save='logs/rsn101_coco'
mkdir -p $save
CUDA_VISIBLE_DEVICES=1,2 th main_multi.lua \
	-data '/home/yluo/project/dataset/mscoco/images/' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 16 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'coco' \
	-resetClassifier true \
	-nClasses 160 \
	-LR 1e-05 \
	-imgSize 448 \
	-featSize 14 \
	-nEpochs 10 \
	-lFunc 'mce' | tee $save/log.txt
```
To use G-softmax as the loss function for training, run
```bash
export save='logs/rsn101_coco_gsm'
mkdir -p $save
CUDA_VISIBLE_DEVICES=1,2 th main_multi.lua \
	-data '/home/yluo/project/dataset/mscoco/images/' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 16 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'coco' \
	-resetClassifier true \
	-nClasses 160 \
	-LR 1e-05 \
	-imgSize 448 \
	-featSize 14 \
	-nEpochs 10 \
	-gsm_mu 0 \
	-gsm_sigma 1 \
	-gsm_scale 1 \
	-lFunc 'gsm' | tee $save/log.txt
```

## Train on NUS-WIDE
Step 1: Download [NUS-WIDE](https://lms.comp.nus.edu.sg/research/NUS-WIDE.htm)  
Since NUS-WIDE has invalid and untagged images, we follow the work,
Learning Spatial Regularization with Image-level Supervisions for Multi-label Image Classification, to remove these images by generating new nus_wide_test_imglist.txt, nus_wide_test_label.txt, nus_wide_train_imglist.txt, and nus_wide_train_label.txt under the clean folder. These files can be downloaded from [here](https://drive.google.com/drive/folders/1FR5gUeAB-0HqhBi_j-dEuUrNBFc9q_qc?usp=sharing).
```
/home/yluo/project/dataset/nuswide/
├── clean
│   ├── nus_wide_test_imglist.txt
│   ├── nus_wide_test_label.txt
│   ├── nus_wide_train_imglist.txt
│   └── nus_wide_train_label.txt
├── Concepts81.txt
└── images
```
There are 81 folders under images folder.  
Step 2: Download pretrained model ResNet-101 from [ResNet Torch homepage](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)  
Step 3: To use softmax as the loss function for training, run  
```bash
export save='logs/rsn101_nuswide'
mkdir -p $save
CUDA_VISIBLE_DEVICES=0,1 th main_multi.lua \
	-data '/home/yluo/project/dataset/nuswide/images/' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 16 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'nuswideclean' \
	-resetClassifier true \
	-nClasses 162 \
	-LR 1e-05 \
	-imgSize 448 \
	-featSize 14 \
	-nEpochs 10 \
	-lFunc 'mce' | tee $save/log.txt
```
To use G-softmax as the loss function for training, run
```bash
export save='logs/rsn101_nuswide_gsm'
mkdir -p $save
CUDA_VISIBLE_DEVICES=0,1 th main_multi.lua \
	-data '/home/yluo/project/dataset/nuswide/images/' \
	-retrain '/home/yluo/project/lua/saliency_torch/pretrained/resnet-101.t7' \
	-save $save \
	-batchSize 16 \
	-nGPU 2 \
	-nThreads 4 \
	-shareGradInput true \
	-dataset 'nuswideclean' \
	-resetClassifier true \
	-nClasses 162 \
	-LR 1e-05 \
	-imgSize 448 \
	-featSize 14 \
	-nEpochs 10 \
	-gsm_mu 0 \
	-gsm_sigma 1 \
	-gsm_scale .1 \
	-gsm_lr_w 100 \
	-lFunc 'gsm' | tee $save/log.txt
```

## Evaluation

**Classification**: Top 1 error. The process will print out the top 1 error on the test set at the end of every epoch.

**Multi-label Classification**: mean average precision (mAP), Class-Precision (C-P), Class-Recall (C-R), Class-F1 (C-F), Overall-Precision (O-P), Overall-Recall (O-R), and Overall-F1 (O-F). The code for evaluation over those metrics are in folder [eval](eval/), which is imported from [Spatial Regularization Network](https://github.com/zhufengx/SRN_multilabel)


## Contact
luoxx648 at umn.edu   
Any discussions, suggestions, and questions are welcome!
