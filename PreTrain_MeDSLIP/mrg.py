import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import utils
from scheduler import create_scheduler
from optim import create_optimizer
from dataset.dataset2 import MedKLIP_Dataset
from models.model_MedKLIP import MedKLIP
from models.tokenization_bert import BertTokenizer

import nltk
from einops import rearrange
import re
import string
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from rouge import Rouge 

def get_tokenizer(tokenizer, target_text):

    target_tokenizer = tokenizer(
        list(target_text),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    return target_tokenizer

def clean_text(text):
    # Replace newline and tab characters with a space
    text = re.sub(r'[\n\t]+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Optional: Additional cleaning steps can be added here (e.g., lowercasing, punctuation removal)
    text = text.lower()  # Lowercase the text
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    return text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = yaml.load(open('PreTrain_MedKLIP/configs/Pretrain_MedKLIP.yaml', "r"), Loader=yaml.Loader)
json_book = json.load(open('PreTrain_MedKLIP/data_file/observation explanation.json', "r"))
disease_book = [json_book[i] for i in json_book]
ana_book = [
    "It is located at " + i
    for i in [
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
]
tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])
ana_book_tokenizer = get_tokenizer(tokenizer, ana_book).to(device)
disease_book_tokenizer = get_tokenizer(tokenizer, disease_book).to(device)
print("Creating model")
model = MedKLIP(config, ana_book_tokenizer, disease_book_tokenizer, mode="train")
model = nn.DataParallel(
    model, device_ids=[i for i in range(torch.cuda.device_count())]
)

ckpt = torch.load("/home/wenrui/Projects/MIMIC/MedKLIP/runs/dual_stream/2024-02-14_22-44-14/checkpoint_65.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

model = model.to(device)

val_datasets = MedKLIP_Dataset(
    'setting/rad_graph_metric_test_local.json', 'setting/landmark_observation_adj_mtx.npy', mode="train"
)

val_dataloader = DataLoader(
    val_datasets,
    batch_size=64,
    num_workers=30,
    pin_memory=True,
    # sampler=val_sampler,
    shuffle=True,
    collate_fn=None,
    drop_last=False,
)
model.eval()
losses = []
position_name = [
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
disease_name = [
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
    "coll_eapse",
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
bleus = []
rouges = {
    "rouge-1": {},
    "rouge-2": {},
    "rouge-l": {}
}
loop = tqdm(val_dataloader)
for j, sample in enumerate(loop):
    # print(i)
    images = sample["image"].to(device)
    labels_e = sample["label_e"].to(device)
    labels_p = sample["label_p"].to(device)
    index_e = sample["index_e"].to(device)
    index_p = sample["index_p"].to(device)
    matrix = sample["matrix"].to(device)
    text = sample["txt"]
    # print(text)
    loss, x_e, ws_e, x_p, ws_p, output_logits = model(
        images,
        labels_e=labels_e,
        labels_p=labels_p,
        matrix=matrix,
        sample_index_e=index_e,
        sample_index_p=index_p,
        is_train=True,
        text_gen=True,
        no_cl=config["no_cl"],
        exclude_class=config["exclude_class"],
    )
    b, n, c = output_logits.shape
    output_logits = rearrange(output_logits, 'b n c -> (b n c)')
    output_logits = nn.functional.sigmoid(output_logits)
    output_logits = rearrange(output_logits, '(b n c) -> b n c', b=b, n=n, c=c)
    img_index, pos_index, ent_index = torch.where(output_logits > 0.1)
    report_dic = {}
    for i in range(len(img_index)):
        if img_index[i] not in report_dic.keys():
            txt = text[img_index[i]].split("FINDINGS:")[-1]
            txt = clean_text(txt.replace("IMPRESSION:", ""))
            report_dic[img_index[i]] = {
                'pred': f"The patient has {disease_name[ent_index[i]]} at {position_name[pos_index[i]]}.",
                'gt': txt
            }
            # report_dic[i] = f"The patient has {disease_name[ent_index[i]]} at {position_name[pos_index[i]]}."
        else:
            report_dic[img_index[i]]['pred'] += f"The patient has {disease_name[ent_index[i]]} at {position_name[pos_index[i]]}."

    idx = list(report_dic.keys())[0]
    rou = Rouge().get_scores(report_dic[idx]['pred'], report_dic[idx]['gt'])[0]
    
    reference = [report_dic[idx]['gt'].split()]
    candidate = report_dic[idx]['pred'].split()
    bleus.append(sentence_bleu(reference, candidate))
    # break
#     losses.append(loss.item())
print(np.mean(bleus))
# print(np.array(bleus).sum()/len(val_dataloader))