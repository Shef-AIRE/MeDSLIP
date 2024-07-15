# MeDSLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology

## Introduction:

The official implementation  code for "MeDSLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology".

[**Arxiv Version**](https://arxiv.org/abs/2403.10635)

## Quick Start:
Check checkpoints directory to download our pre-trained model from [Hugging Face: MeDSLIP](https://huggingface.co/pykale/MeDSLIP). It can be used for all zero-shot and finetuning tasks.

* **Zero-Shot Classification:**

    We give an example on CXR14 in ```Sample_Zero-Shot_Classification_CXR14```. Change the data paths, and test our model by ```python test.py```.
    We give an example on RSNA in ```Sample_Zero-Shot_Classification_RSNA```. Change the data paths, and test our model by ```python test.py```.

* **Zero-Shot Grounding:**

    We give an example on RSNA_Pneumonia in ```Sample_Zero-Shot_Grounding_RSNA```. Change the data paths, and test our model by ```python test.py```.

* **Finetuning:**

    We give segmentation and classification finetune code on SIIM_ACR dataset in ```Sample_Finetuning_SIIMACR```. Change the data paths, and finetune our model by ```python I1_classification/train_res_ft.py``` or ```python I2_segementation/train_res_ft.py```.

## Pre-train:
### Data Preparation
All files for data preparation files can be downloaded from [Hugging Face: MeDSLIP](https://huggingface.co/pykale/MeDSLIP).
- Extracted triplets: `landmark_observation_adj_mtx.npy`
- Training list: `train.json`
- Validation list: `valid.json`
- Test list: `test.json`

### Pre-training
Our pre-train code is given in ```PreTrain_MeDSLIP```.
* Check the ```PreTrain_MeDSLIP/data_file``` dir and download the files for data preparation.
* Change the data and preparation files paths as you disire in ```PreTrain_MeDSLIP/configs/Pretrain_MeDSLIP.yaml```, and ```python PreTrain_MeDSLIP/train_MeDSLIP.py``` to pre-train.

## Reference
```
@article{fan2024medslip,
  title={MeDSLIP: Medical Dual-Stream Language-Image Pre-training for Fine-grained Alignment},
  author={Fan, Wenrui and Suvon, Mohammod Naimul Islam and Zhou, Shuo and Liu, Xianyuan and Alabed, Samer and Osmani, Venet and Swift, Andrew and Chen, Chen and Lu, Haiping},
  journal={arXiv preprint arXiv:2403.10635},
  year={2024}
}
```

## Contact
If you have any question, please feel free to contact winslow.fan@outlook.com.
