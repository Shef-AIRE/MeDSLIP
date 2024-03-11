# MeDSLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology

## Introduction:

The official implementation  code for "MeDSLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology".
<!-- 
[**Paper Web**](https://chaoyi-wu.github.io/MeDSLIP/) -->

[**Arxiv Version**](https://arxiv.org/abs/)

## Quick Start:
Check checkpoints dir to download our pre-trained model. It can be used for all zero-shot && finetuning tasks

* **Zero-Shot Classification:**

    We give an example on CXR14 in ```Sample_Zero-Shot_Classification_CXR14```. Modify the path, and test our model by ```python test.py```
    We give an example on RSNA in ```Sample_Zero-Shot_Classification_RSNA```. Modify the path, and test our model by ```python test.py```
    
* **Zero-Shot Grounding:**

    We give an example on RSNA_Pneumonia in ```Sample_Zero-Shot_Grounding_RSNA```. Modify the path, and test our model by ```python test.py```
* **Finetuning:**

    We give segmentation and classification finetune code on SIIM_ACR dataset in ```Sample_Finetuning_SIIMACR```. Modify the path, and finetune our model by ```python I1_classification/train_res_ft.py``` or ```python I2_segementation/train_res_ft.py```

## Pre-train:
Our pre-train code is given in ```Train_MeDSLIP```.
* Check the ```Train_MeDSLIP\data_file``` dir and download the pre-processed data files.
* Modify the path as you disire in ```PreTrain_MeDSLIP/configs/Pretrain_MeDSLIP.yaml```, and ```python PreTrain_MeDSLIP/train_MeDSLIP.py``` to pre-train.

## Reference
- Wu, Chaoyi, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. "MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology." arXiv preprint arXiv:2301.02228 (2023).
## Contact
If you have any question, please feel free to contact winslow.fan@outlook.com.
