# HDC-Net: Hierarchical Decoupled Convolution Network for Brain Tumor Segmentation

This repository is the work of "_HDC-Net: Hierarchical Decoupled Convolution Network for Brain Tumor Segmentation_" based on **pytorch** implementation.The multimodal brain tumor dataset (BraTS 2018) could be acquired from [here](https://www.med.upenn.edu/sbia/brats2018.html).

## HDC-Net

<center>Architecture of  HDC-Net</center>
<div  align="center">  
 <img src="https://github.com/luozhengrong/HDC-Net/blob/master/fig/HDC_Net.jpg"
     align=center/>
</div>
<div  align="center">  
 <img src="https://github.com/luozhengrong/HDC-Net/blob/master/fig/HDC_block.jpg"
     align=center/>
</div>



## Requirements

* python 3.6
* pytorch 0.4 or 1.0
* nibabel
* pickle 
* imageio
* pyyaml

## Implementation

Download the BraTS2018 dataset and change the path:

```
experiments/PATH.yaml
```
the dataset structure:
```wiki
BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1：
├── MICCAI_BraTS_2018_Data_Training
    ├── HGG
        ├── Brats18_2013_2_1
            ├── Brats18_2013_2_1_flair.nii.gz
            ├── Brats18_2013_2_1_seg.nii.gz
            ├── Brats18_2013_2_1_t1.nii.gz
            ├── Brats18_2013_2_1_t1ce.nii.gz
            ├── Brats18_2013_2_1_t2.nii.gz
        ├── Brats18_2013_2_2
            ├── Brats18_2013_2_2_flair.nii.gz
            ├── Brats18_2013_2_2_seg.nii.gz
            ├── Brats18_2013_2_2_t1.nii.gz
            ├── Brats18_2013_2_2_t1ce.nii.gz
            ├── Brats18_2013_2_2_t2.nii.gz
        ├── ...
    ├── LGG
        ├── Brats18_2013_0_1
            ├── Brats18_2013_0_1_flair.nii.gz
            ├── Brats18_2013_0_1_seg.nii.gz
            ├── Brats18_2013_0_1_t1.nii.gz
            ├── Brats18_2013_0_1_t1ce.nii.gz
            ├── Brats18_2013_0_1_t2.nii.gz
        ├── Brats18_2013_1_1
            ├── Brats18_2013_1_1_flair.nii.gz
            ├── Brats18_2013_1_1_seg.nii.gz
            ├── Brats18_2013_1_1_t1.nii.gz
            ├── Brats18_2013_1_1_t1ce.nii.gz
            ├── Brats18_2013_1_1_t2.nii.gz
    ├── ...
├── MICCAI_BraTS_2018_Data_Validation
    ├── Brats18_CBICA_AAM_1
        ├── Brats18_CBICA_AAM_1_flair.nii.gz
        ├── Brats18_CBICA_AAM_1_t1.nii.gz
        ├── Brats18_CBICA_AAM_1_t1ce.nii.gz
        ├── Brats18_CBICA_AAM_1_t2.nii.gz
    ├── Brats18_CBICA_ABT_1
        ├── Brats18_CBICA_ABT_1_flair.nii.gz
        ├── Brats18_CBICA_ABT_1_t1.nii.gz
        ├── Brats18_CBICA_ABT_1_t1ce.nii.gz
        ├── Brats18_CBICA_ABT_1_t2.nii.gz
    ├── ...
```
- shape(224, 224, 155)
- spacing(1, 1, 1)
- origin(0, -239, 0)
- direction=identity matrix

### Data preprocess

generate all.txt and test.txt
```bash
python data2txt.py -i "BraTS_2018/MICCAI_BraTS_2018_Data_Training" -o "all.txt" 
python data2txt.py -i "BraTS_2018/MICCAI_BraTS_2018_Data_Validation" -o "test.txt"
```

generate train.txt and val.txt or k-fold train_{}.txt and val_{}.txt
```bash
# train/val
python split_train_val.py -i "/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training/all.txt" -k 5 -o 1 

# k-fold
python split_train_val.py -i "/mnt/data/datasets/BraTS_2018/MICCAI_BraTS_2018_Data_Training/all.txt" -k 5 -o 0
```

Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance .  
it will generate a pkl containing image([4, 224, 224, 155]) and label([224, 224, 155] or [1] for the test dataset)  
**remember to change the path in preprocess.py**  
```
python preprocess.py
```

finally the dataset structure:
```wiki
BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_2_1：
├── MICCAI_BraTS_2018_Data_Training
    ├── all.txt
    ├── train.txt
    ├── val.txt
    ├── train_{}.txt
    ├── val_{}.txt
    ├── HGG
        ├── Brats18_2013_2_1
            ├── Brats18_2013_2_1_data_f32.pkl
            ├── Brats18_2013_2_1_flair.nii.gz
            ├── Brats18_2013_2_1_seg.nii.gz
            ├── Brats18_2013_2_1_t1.nii.gz
            ├── Brats18_2013_2_1_t1ce.nii.gz
            ├── Brats18_2013_2_1_t2.nii.gz
        ├── Brats18_2013_2_2
            ├── Brats18_2013_2_2_data_f32.pkl
            ├── Brats18_2013_2_2_flair.nii.gz
            ├── Brats18_2013_2_2_seg.nii.gz
            ├── Brats18_2013_2_2_t1.nii.gz
            ├── Brats18_2013_2_2_t1ce.nii.gz
            ├── Brats18_2013_2_2_t2.nii.gz
        ├── ...
    ├── LGG
        ├── Brats18_2013_0_1
            ├── Brats18_2013_0_1_data_f32.pkl
            ├── Brats18_2013_0_1_flair.nii.gz
            ├── Brats18_2013_0_1_seg.nii.gz
            ├── Brats18_2013_0_1_t1.nii.gz
            ├── Brats18_2013_0_1_t1ce.nii.gz
            ├── Brats18_2013_0_1_t2.nii.gz
        ├── Brats18_2013_1_1
            ├── Brats18_2013_1_1_data_f32.pkl
            ├── Brats18_2013_1_1_flair.nii.gz
            ├── Brats18_2013_1_1_seg.nii.gz
            ├── Brats18_2013_1_1_t1.nii.gz
            ├── Brats18_2013_1_1_t1ce.nii.gz
            ├── Brats18_2013_1_1_t2.nii.gz
    ├── ...
├── MICCAI_BraTS_2018_Data_Validation
    ├── test.txt
    ├── Brats18_CBICA_AAM_1
        ├── Brats18_CBICA_AAM_1_data_f32.pkl
        ├── Brats18_CBICA_AAM_1_flair.nii.gz
        ├── Brats18_CBICA_AAM_1_t1.nii.gz
        ├── Brats18_CBICA_AAM_1_t1ce.nii.gz
        ├── Brats18_CBICA_AAM_1_t2.nii.gz
    ├── Brats18_CBICA_ABT_1
        ├── Brats18_CBICA_ABT_1_data_f32.pkl
        ├── Brats18_CBICA_ABT_1_flair.nii.gz
        ├── Brats18_CBICA_ABT_1_t1.nii.gz
        ├── Brats18_CBICA_ABT_1_t1ce.nii.gz
        ├── Brats18_CBICA_ABT_1_t2.nii.gz
    ├── ...
```
### Training

Sync bacth normalization is used so that a proper batch size is important to obtain a decent performance. Multiply gpus training with batch_size=10 is recommended.The total training time is about 12 hours and the average prediction time for each volume is 2.3 seconds when using randomly cropped volumes of size 128×128×128 and batch size 10 on two parallel Nvidia Tesla K40 GPUs for 800 epochs.

```
python train_all.py --gpu="0,1" --cfg="HDC_Net" --batch_size=10
```

### Test

You could obtain the resutls as paper reported by running the following code:

```
python test.py --mode=1 --is_out=True --verbose=True --use_TTA=False --postprocess=True --snapshot=True --restore="model_last.pth" --cfg="HDC_Net" --gpu="0"
```
Then make a submission to the online evaluation server.

## Citation

If you use our code or model in your work or find it is helpful, please cite the paper:
```
@ARTICLE{9103199, 
title={HDC-Net: Hierarchical Decoupled Convolution Network for Brain Tumor Segmentation},
author={Z. {Luo} and Z. {Jia} and Z. {Yuan} and J. {Peng}},
journal={IEEE Journal of Biomedical and Health Informatics}, 
year={2020}}
```

## Acknowledge

1. [DMFNet](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)
2. [BraTS2018-tumor-segmentation](https://github.com/ieee820/BraTS2018-tumor-segmentation)
3. [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

