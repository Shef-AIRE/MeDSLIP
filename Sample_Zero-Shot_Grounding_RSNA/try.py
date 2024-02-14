import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset_RSNA import RSNA2018_Dataset
from models.model_MedKLIP import MedKLIP
from models.tokenization_bert import BertTokenizer
from PIL import Image
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from skimage import exposure
import pydicom


original_class = [
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


def get_tokenizer(tokenizer, target_text):

    target_tokenizer = tokenizer(
        list(target_text),
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    return target_tokenizer

def read_dcm(dcm_path):
    dcm_data = pydicom.read_file(dcm_path)
    img = dcm_data.pixel_array.astype(float) / 255.0
    img = exposure.equalize_hist(img)

    img = (255 * img).astype(np.uint8)
    img = Image.fromarray(img).convert("RGB")
    return img

def main(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type("torch.FloatTensor")

    #### Dataset ####
    print("Creating dataset")
    # test_dataset = RSNA2018_Dataset(config["test_file"])
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=config["test_batch_size"],
    #     num_workers=2,
    #     pin_memory=True,
    #     sampler=None,
    #     shuffle=False,
    #     collate_fn=None,
    #     drop_last=False,
    # )
    data_info = pd.read_csv('/home/wenrui/Projects/MIMIC/MedKLIP/Sample_Zero-Shot_Grounding_RSNA/data_sample/test.csv')
    img_path_list = np.asarray(data_info.iloc[:, 1])
    bbox_list = np.asarray(data_info.iloc[:, 2])
    class_list = np.asarray(data_info.iloc[:, 3])
    json_book = json.load(open(config["disease_book"], "r"))
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
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=False)
    print("load checkpoint from %s" % args.checkpoint)

    print("Start testing")
    model.eval()

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    for i, sample in enumerate(img_path_list):

        images_raw = read_dcm(sample)
        images = transform(images_raw).to(device).unsqueeze(0)
        bbox = bbox_list[i]
        class_label = np.array([class_list[i]])
        seg_map = np.zeros((1024, 1024))
        if class_label == 1:
            boxes = bbox.split("|")
            for box in boxes:
                cc = box.split(";")
                seg_map[
                    int(float(cc[1])) : (int(float(cc[1])) + int(float(cc[3]))),
                    int(float(cc[0])) : (int(float(cc[0])) + int(float(cc[2]))),
                ] = 1

        

        # image_path = sample["image_path"]
        batch_size = images.shape[0]
        # labels = sample["label"].to(device)
        # seg_map = sample["seg_map"][:, 0, :, :].to(device)  # B C H W

        with torch.no_grad():
            _, _, ws_e, ws_p = model(
                images, None, is_train=False
            )  # batch_size,batch_size,image_patch,text_patch
            ws_e = (ws_e[-4] + ws_e[-3] + ws_e[-2] + ws_e[-1]) / 4
            ws_p = (ws_p[-4] + ws_p[-3] + ws_p[-2] + ws_p[-1]) / 4

            ws_e = ws_e[:, original_class.index("pneumonia"), :]

            ws_e = ws_e.reshape(batch_size, 14, 14).squeeze().cpu().numpy()
            ws_p = ws_p.reshape(ws_p.shape[1], 14, 14).squeeze().cpu().numpy()


            ws_e_resize = Image.fromarray(ws_e).resize(images_raw.size, Image.BILINEAR)
            ws_p_resize = Image.fromarray(ws_p).resize(images_raw.size, Image.BILINEAR)
            ws_e_normalized = (ws_e_resize - np.min(ws_e_resize)) / (np.max(ws_e_resize) - np.min(ws_e_resize))
            ws_p_normalized = (ws_p_resize - np.min(ws_p_resize)) / (np.max(ws_p_resize) - np.min(ws_p_resize))
            ws_e_colormap = plt.get_cmap("viridis")(ws_e_normalized)[:, :, :3]
            ws_p_colormap = plt.get_cmap("viridis")(ws_p_normalized)[:, :, :3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Sample_Zero-Shot_Grounding_RSNA/configs/MedKLIP_config.yaml")
    parser.add_argument("--checkpoint", default="/home/wenrui/Projects/MIMIC/MedKLIP/runs/dual_stream/2024-02-09_04-26-52/checkpoint_state.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--use_ws_p", type=bool, default=True, help="use ws_p")

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu != "-1":
        torch.cuda.current_device()
        torch.cuda._initialized = True

    main(args, config)