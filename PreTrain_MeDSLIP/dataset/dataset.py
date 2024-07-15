import json
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import random
from dataset.randaugment import RandomAugment


class MeDSLIP_Dataset(Dataset):
    def __init__(self, csv_path, np_path, mode="train", num_neg_samples=7):
        self.num_neg_samples = num_neg_samples
        self.ann = json.load(open(csv_path, "r"))
        self.img_path_list = list(self.ann)
        self.anaomy_list = [
            "trachea",
            "left_hilar",
            "right_hilar",
            "hilar_unspec",
            "left_pleural",
            "right_pleural",
            "pleural_unspec",
            "heart_size",
            "heart_border",
            "left_diaphragm",
            "right_diaphragm",
            "diaphragm_unspec",
            "retrocardiac",
            "lower_left_lobe",
            "upper_left_lobe",
            "lower_right_lobe",
            "middle_right_lobe",
            "upper_right_lobe",
            "left_lower_lung",
            "left_mid_lung",
            "left_upper_lung",
            "left_apical_lung",
            "left_lung_unspec",
            "right_lower_lung",
            "right_mid_lung",
            "right_upper_lung",
            "right_apical_lung",
            "right_lung_unspec",
            "lung_apices",
            "lung_bases",
            "left_costophrenic",
            "right_costophrenic",
            "costophrenic_unspec",
            "cardiophrenic_sulcus",
            "mediastinal",
            "spine",
            "clavicle",
            "rib",
            "stomach",
            "right_atrium",
            "right_ventricle",
            "aorta",
            "svc",
            "interstitium",
            "parenchymal",
            "cavoatrial_junction",
            "cardiopulmonary",
            "pulmonary",
            "lung_volumes",
            "unspecified",
            "other",
        ]
        self.obs_list = [
            "normal",
            "clear",
            "sharp",
            "sharply",
            "unremarkable",
            "intact",
            "stable",
            "free",
            "effusion",
            "opacity",
            "pneumothorax",
            "edema",
            "atelectasis",
            "tube",
            "consolidation",
            "process",
            "abnormality",
            "enlarge",
            "tip",
            "low",
            "pneumonia",
            "line",
            "congestion",
            "catheter",
            "cardiomegaly",
            "fracture",
            "air",
            "tortuous",
            "lead",
            "disease",
            "calcification",
            "prominence",
            "device",
            "engorgement",
            "picc",
            "clip",
            "elevation",
            "expand",
            "nodule",
            "wire",
            "fluid",
            "degenerative",
            "pacemaker",
            "thicken",
            "marking",
            "scar",
            "hyperinflate",
            "blunt",
            "loss",
            "widen",
            "collapse",
            "density",
            "emphysema",
            "aerate",
            "mass",
            "crowd",
            "infiltrate",
            "obscure",
            "deformity",
            "hernia",
            "drainage",
            "distention",
            "shift",
            "stent",
            "pressure",
            "lesion",
            "finding",
            "borderline",
            "hardware",
            "dilation",
            "chf",
            "redistribution",
            "aspiration",
            "tail_abnorm_obs",
            "excluded_obs",
        ]
        self.rad_graph_results = np.load(np_path)
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(0.2, 1.0), interpolation=Image.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(
                        2,
                        7,
                        isPIL=True,
                        augs=[
                            "Identity",
                            "AutoContrast",
                            "Equalize",
                            "Brightness",
                            "Sharpness",
                            "ShearX",
                            "ShearY",
                            "TranslateX",
                            "TranslateY",
                            "Rotate",
                        ],
                    ),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        if mode == "test":
            self.transform = transforms.Compose(
                [
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.rad_graph_results[
            self.ann[img_path]["labels_id"], :, :
        ] 
        labels_pathology = np.zeros(class_label.shape[-1]) - 1
        labels_anatomy = np.zeros(class_label.shape[0]) - 1
        labels_pathology, index_list_pathology = self.triplet_extraction_pathology(
            class_label
        )
        labels_anatomy, index_list_anatomy = self.triplet_extraction_anatomy(
            class_label
        )
        index_list_pathology = np.array(index_list_pathology)
        index_list_anatomy = np.array(index_list_anatomy)

        img = PIL.Image.open(img_path).convert("RGB")
        image = self.transform(img)

        return {
            "image": image,
            "label_pathology": labels_pathology,
            "index_pathology": index_list_pathology,
            "label_anatomy": labels_anatomy,
            "index_anatomy": index_list_anatomy,
            "matrix": class_label,
        }

    def triplet_extraction_pathology(self, class_label):
        """
        This is for ProtoCL. Therefore, we need to extract anatomies to use in pathology stream.
        """

        exist_labels = np.zeros(class_label.shape[-1]) - 1
        anatomy_list = []
        for i in range(class_label.shape[1]):
            temp_list = []
            ### extract the exist label for each pathology and maintain -1 if not mentioned. ###
            if 0 in class_label[:, i]:
                exist_labels[i] = 0

            if 1 in class_label[:, i]:
                exist_labels[i] = 1
                ### if the pathology exists try to get its anatomy.###
                ### Note that, the contrastive loss will only be caculated on exist pathology as it is meaningless to predict their anatomy for the non-exist entities###
                temp_list.append(-1)

                try:
                    temp_list = temp_list + random.sample(
                        np.where(class_label[:, i] != 1)[0].tolist(),
                        self.num_neg_samples,
                    )
                except:
                    print("fatal error")
            if temp_list == []:
                temp_list = temp_list + random.sample(
                    np.where(class_label[:, i] != 1)[0].tolist(),
                    self.num_neg_samples + 1,
                )
            anatomy_list.append(temp_list)

        return exist_labels, anatomy_list

    def triplet_extraction_anatomy(self, class_label):
        """
        This is for ProtoCL. Therefore, we need to extract pathological labels to use in anatomy stream.
        """
        exist_labels = np.zeros(class_label.shape[0]) - 1
        pathology_list = []
        for i in range(class_label.shape[0]):
            temp_list = []
            ### extract the exist label for each pathology and maintain -1 if not mentioned. ###
            if 0 in class_label[i, :]:
                exist_labels[i] = 0

            if 1 in class_label[i, :]:
                exist_labels[i] = 1
                ### if the pathology exists try to get its anatomy.###
                ### Note that, the contrastive loss will only be caculated on exist pathology as it is meaningless to predict their anatomy for the non-exist entities###
                temp_list.append(-1)

                try:
                    temp_list = temp_list + random.sample(
                        np.where(class_label[i, :] != 1)[0].tolist(),
                        self.num_neg_samples,
                    )
                except:
                    print("fatal error")
            if temp_list == []:
                temp_list = temp_list + random.sample(
                    np.where(class_label[i, :] != 1)[0].tolist(),
                    self.num_neg_samples + 1,
                )
            pathology_list.append(temp_list)

        return exist_labels, pathology_list

    def __len__(self):
        return len(self.ann)


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
