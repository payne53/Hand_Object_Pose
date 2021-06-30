# Joint Hand-Object Pose Estimation with Differentiably-Learned Physical Contact Point Analysis

Codes for [**Hand-Object Pose Estimation paper**](https://arxiv.org/pdf/2004.00060.pdf) (ICMR 2021), a Contextual Graph Convolutional model for Hand-Object Pose Estimation.

The goal of Hand-Object Pose Estimation (HOPE) is to jointly estimate the poses of both the hand and a handled object. Our model can estimate the 2D and 3D hand and object poses and enhance the physical stability of hand-object system.
<div align="center">
    <img src="images/motivation.png", width="1000">
</div>

## Architecture
The model starts with ResNet as the image encoder and for predicting the initial 2D coordinates
of the joints and object vertices. The coordinates concatenated with the image features used as the features of the input graph of a 3 layered graph convolution to use the power of neighbors features to estimate the better 2D pose. Finally the 2D coordinates predicted in the previous step are passed to our context-aware graph module to find the 3D coordinates of the hand and object.
<div align="center">
    <img src="images/pipeline.png", width="1000">
</div>

A physical affinity loss is proposed to enhance the stability of grasping as well.
<div align="center">
    <img src="images/loss.png", width="1000">
</div>

## Datasets
To use the datasets used in the paper download [**First-Person Hand Action Dataset**](https://guiggh.github.io/publications/first-person-hands/) and [**HO-3D Dataset**](https://www.tugraz.at/index.php?id=40231) and update the root path in the `make_data.py` file located in each folder and run the `make_data.py` files to generate the `.npy` files.

## Test Pretrained Model
First download [**First-Person Hand Action Dataset**](https://guiggh.github.io/publications/first-person-hands/) and make the `.npy` files. Then download and extract the pretrained model with the comman below.
```
wget http://vision.soic.indiana.edu/hopenet_files/checkpoints.tar.gz
tar -xvf checkpoints.tar.gz
```
And then run the model using the pretrained weights.
```
python HOPE.py \
  --input_file ./datasets/fhad/ \
  --test \
  --batch_size 64 \
  --model_def HopeNet \
  --gpu \
  --gpu_number 0 \
  --pretrained_model ./checkpoints/fhad/model-0.pkl
```


## Citation
Please cite our paper if this code helps your research.
```bibtex
@InProceedings{Doosti_2020_CVPR,
author = {Bardia Doosti and Shujon Naha and Majid Mirbagheri and David Crandall},
title = {HOPE-Net: A Graph-based Model for Hand-Object Pose Estimation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
