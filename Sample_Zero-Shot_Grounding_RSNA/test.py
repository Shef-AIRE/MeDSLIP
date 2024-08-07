import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset_RSNA import RSNA2018_Dataset
from models.model_MeDSLIP import MeDSLIP
from models.tokenization_bert import BertTokenizer
from tqdm import tqdm

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


def score_cal(labels, seg_map, pred_map, threshold=0.005):
    """
    labels B * 1
    seg_map B *H * W
    pred_map B * H * W
    """
    device = labels.device
    total_num = torch.sum(labels)
    mask = (labels == 1).squeeze()
    seg_map = seg_map[mask, :, :].reshape(total_num, -1)
    pred_map = pred_map[mask, :, :].reshape(total_num, -1)
    one_hot_map = pred_map > threshold
    dot_product = (seg_map * one_hot_map).reshape(total_num, -1)

    max_number = torch.max(pred_map, dim=-1)[0]
    point_score = 0
    for i, number in enumerate(max_number):
        temp_pred = (pred_map[i] == number).type(torch.int)
        flag = int((torch.sum(temp_pred * seg_map[i])) > 0)
        point_score = point_score + flag
    mass_score = torch.sum(dot_product, dim=-1) / (
        (torch.sum(seg_map, dim=-1) + torch.sum(one_hot_map, dim=-1))
        - torch.sum(dot_product, dim=-1)
    )
    dice_score = (
        2
        * (torch.sum(dot_product, dim=-1))
        / (torch.sum(seg_map, dim=-1) + torch.sum(one_hot_map, dim=-1))
    )
    return total_num, point_score, mass_score.to(device), dice_score.to(device)


def main(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type("torch.FloatTensor")

    #### Dataset ####
    print("Creating dataset")
    test_dataset = RSNA2018_Dataset(config["test_file"])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        num_workers=30,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )
    json_book = json.load(open(config["disease_book"], "r"))
    disease_book = [json_book[i] for i in json_book]
    ana_list = [
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
    ana_book = []
    for i in ana_list:
        ana_book.append("It is located at " + i + ". ")
    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])
    ana_book_tokenizer = get_tokenizer(tokenizer, ana_book).to(device)
    disease_book_tokenizer = get_tokenizer(tokenizer, disease_book).to(device)

    print("Creating model")
    model = MeDSLIP(config, ana_book_tokenizer, disease_book_tokenizer, mode="train")
    if args.ddp:
        model = nn.DataParallel(
            model, device_ids=[i for i in range(torch.cuda.device_count())]
        )
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=False)
    print("load checkpoint from %s" % args.checkpoint)

    print("Start testing")
    model.eval()

    dice_score_A = torch.FloatTensor()
    dice_score_A = dice_score_A.to(device)
    mass_score_A = torch.FloatTensor()
    mass_score_A = mass_score_A.to(device)
    total_num_A = 0
    point_num_A = 0
    loop = tqdm(test_dataloader)
    for i, sample in enumerate(loop):
        loop.set_description(f"Testing: {i+1}/{len(test_dataloader)}")
        images = sample["image"].to(device)
        image_path = sample["image_path"]
        batch_size = images.shape[0]
        labels = sample["label"].to(device)
        seg_map = sample["seg_map"][:, 0, :, :].to(device)  # B C H W

        with torch.no_grad():
            _, _, ws_e, ws_p, features_e, features_p = model(
                images, labels, is_train=False
            )
            features_e = features_e.transpose(0, 1)
            features_p = features_p.transpose(0, 1)
            ws_e = (ws_e[-4] + ws_e[-3] + ws_e[-2] + ws_e[-1]) / 4
            ws_p = (ws_p[-4] + ws_p[-3] + ws_p[-2] + ws_p[-1]) / 4
            pred_map = ws_e[:, original_class.index("pneumonia"), :]

            threshold = 0
            if args.use_ws_p:
                pred_map = pred_map.unsqueeze(1)
                pred_map = pred_map.repeat(1, ws_p.shape[1], 1)
                pred_map = (pred_map * ws_p).mean(axis=1)
                threshold = 0.01

            pred_map = pred_map / torch.max(pred_map)

            pred_map = pred_map.reshape(batch_size, 14, 14).detach().cpu().numpy()

            pred_map = torch.from_numpy(
                pred_map.repeat(16, axis=1).repeat(16, axis=2)
            ).to(
                device
            )  # Final Grounding Heatmap

            total_num, point_num, mass_score, dice_score = score_cal(
                labels, seg_map, pred_map, threshold=threshold
            )
            total_num_A = total_num_A + total_num
            point_num_A = point_num_A + point_num
            dice_score_A = torch.cat((dice_score_A, dice_score), dim=0)
            mass_score_A = torch.cat((mass_score_A, mass_score), dim=0)

    dice_score_avg = torch.mean(dice_score_A)
    mass_score_avg = torch.mean(mass_score_A)
    print(
        "The average dice_score is {dice_score_avg:.5f}".format(
            dice_score_avg=dice_score_avg
        )
    )
    print(
        "The average iou_score is {mass_score_avg:.5f}".format(
            mass_score_avg=mass_score_avg
        )
    )
    point_score = point_num_A / total_num_A
    print(
        "The average point_score is {point_score:.5f}".format(point_score=point_score)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="Sample_Zero-Shot_Grounding_RSNA/configs/MeDSLIP_config.yaml",
    )
    parser.add_argument("--checkpoint", default="MeDSLIP_resnet50.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--ddp", action="store_true", help="whether to use ddp")

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu != "-1":
        torch.cuda.current_device()
        torch.cuda._initialized = True

    main(args, config)
