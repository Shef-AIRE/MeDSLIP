import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset.dataset_RSNA import RSNA2018_Dataset
from models.model_MedKLIP import MedKLIP
from models.tokenization_bert import BertTokenizer
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score


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


def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


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
        num_workers=2,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )
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
    model = MedKLIP(config, disease_book_tokenizer)
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

    
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    for i, sample in enumerate(test_dataloader):
        images = sample["image"].to(device)
        batch_size = images.shape[0]
        labels = sample["label"].to(device)
        gt = torch.cat((gt, labels), 0)

        with torch.no_grad():
            pred_class = model(images)
            pred_class = pred_class[:, original_class.index("pneumonia"), :]
            pred_class = 1 - F.softmax(pred_class)
            pred = torch.cat((pred, pred_class), 0)

    AUROC = compute_AUCs(gt, pred, 1)
    print("The AUROC of {} is {}".format("pneumonia", AUROC[0]))
    max_f1s = []
    accs = []
    gt_np = gt[:, 0].cpu().numpy()
    pred_np = pred[:, 0].cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(
        numerator, denom, out=np.zeros_like(denom), where=(denom != 0)
    )
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    max_f1s.append(max_f1)
    accs.append(accuracy_score(gt_np, pred_np > max_f1_thresh))
    f1_avg = np.array(max_f1s).mean()
    acc_avg = np.array(accs).mean()
    print("The average f1 is {F1_avg:.4f}".format(F1_avg=f1_avg))
    print("The average ACC is {ACC_avg:.4f}".format(ACC_avg=acc_avg))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Sample_Zero-Shot_Grounding_RSNA/configs/MedKLIP_config.yaml")
    parser.add_argument("--checkpoint", default="/home/wenrui/Projects/MIMIC/MedKLIP/runs/dual_stream/2024-02-16_16-53-27/checkpoint_28.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--use_ws_p", type=bool, default=False, help="use ws_p")

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu != "-1":
        torch.cuda.current_device()
        torch.cuda._initialized = True

    main(args, config)
