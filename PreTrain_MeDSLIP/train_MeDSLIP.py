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
from dataset.dataset import MeDSLIP_Dataset
from models.model_MeDSLIP import MeDSLIP
from models.tokenization_bert import BertTokenizer


def get_tokenizer(tokenizer, target_text):

    target_tokenizer = tokenizer(
        list(target_text),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    return target_tokenizer


def train(
    model,
    data_loader,
    optimizer,
    epoch,
    warmup_steps,
    device,
    scheduler,
    args,
    config,
    writer,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_ce_p", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_cl_p", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_ce_a", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_cl_a", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "loss_ap", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.update(loss=1.0)
    metric_logger.update(loss_ce_p=1.0)
    metric_logger.update(loss_cl_p=1.0)
    metric_logger.update(loss_ce_a=1.0)
    metric_logger.update(loss_cl_a=1.0)
    metric_logger.update(loss_ap=1.0)
    metric_logger.update(lr=scheduler._get_lr(epoch)[0])

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 1
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    scalar_step = epoch * len(data_loader)

    for i, sample in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        images = sample["image"].to(device)
        labels_pathology = sample["label_pathology"].to(device)
        labels_anatomy = sample["label_anatomy"].to(device)
        index_pathology = sample["index_pathology"].to(device)
        index_anatomy = sample["index_anatomy"].to(device)
        matrix = sample["matrix"].to(device)

        optimizer.zero_grad()

        (
            loss,
            loss_ce_pathology,
            loss_cl_pathology,
            loss_ce_anatomy,
            loss_cl_anatomy,
            loss_ap,
        ) = model(
            images,
            labels_pathology=labels_pathology,
            labels_anatomy=labels_anatomy,
            matrix=matrix,
            sample_index_pathology=index_pathology,
            sample_index_anatomy=index_anatomy,
            is_train=True,
            no_cl=config["no_cl"],
            exclude_class=config["exclude_class"],
        )
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss/loss", loss, scalar_step)
        writer.add_scalar("loss/loss_ce_pathology", loss_ce_pathology, scalar_step)
        writer.add_scalar("loss/loss_cl_pathology", loss_cl_pathology, scalar_step)
        writer.add_scalar("loss/loss_ce_anatomy", loss_ce_anatomy, scalar_step)
        writer.add_scalar("loss/loss_cl_anatomy", loss_cl_anatomy, scalar_step)
        writer.add_scalar("loss/loss_ap", loss_ap, scalar_step)
        scalar_step += 1
        metric_logger.update(loss_ce_p=loss_ce_pathology.item())
        metric_logger.update(loss_cl_p=loss_cl_pathology.item())
        metric_logger.update(loss_ce_a=loss_ce_anatomy.item())
        metric_logger.update(loss_cl_a=loss_cl_anatomy.item())
        metric_logger.update(loss_ap=loss_ap.item())
        metric_logger.update(loss=loss.item())
        # metric_logger.update(loss_cl=loss_cl.item())
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        metric_logger.update(lr=scheduler._get_lr(epoch)[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


def valid(model, data_loader, epoch, device, config, writer):
    model.eval()
    val_scalar_step = epoch * len(data_loader)
    val_loss = []
    for i, sample in enumerate(data_loader):

        images = sample["image"].to(device)
        labels_pathology = sample["label_pathology"].to(device)
        labels_anatomy = sample["label_anatomy"].to(device)
        index_pathology = sample["index_pathology"].to(device)
        index_anatomy = sample["index_anatomy"].to(device)
        matrix = sample["matrix"].to(device)

        with torch.no_grad():
            (
                loss,
                loss_ce_pathology,
                loss_cl_pathology,
                loss_ce_anatomy,
                loss_cl_anatomy,
                loss_ap,
            ) = model(
                images,
                labels_pathology=labels_pathology,
                labels_anatomy=labels_anatomy,
                matrix=matrix,
                sample_index_pathology=index_pathology,
                sample_index_anatomy=index_anatomy,
                is_train=True,
                no_cl=config["no_cl"],
                exclude_class=config["exclude_class"],
            )
            val_loss.append(loss.item())
            writer.add_scalar("val_loss/loss", loss, val_scalar_step)
            writer.add_scalar(
                "val_loss/loss_ce_pathology", loss_ce_pathology, val_scalar_step
            )
            writer.add_scalar(
                "val_loss/loss_cl_pathology", loss_cl_pathology, val_scalar_step
            )
            writer.add_scalar(
                "val_loss/loss_ce_anatomy", loss_ce_anatomy, val_scalar_step
            )
            writer.add_scalar(
                "val_loss/loss_cl_anatomy", loss_cl_anatomy, val_scalar_step
            )
            writer.add_scalar("val_loss/loss_ap", loss_ap, val_scalar_step)
            val_scalar_step += 1
    avg_val_loss = np.array(val_loss).mean()
    return avg_val_loss


def main(args, config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.computing == "parallel":
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        device = torch.device("cuda", rank)
        print("World size: ", world_size, "; Rank: ", rank)

    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type("torch.FloatTensor")
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]

    #### Dataset ####
    print("Creating dataset")
    train_datasets = MeDSLIP_Dataset(
        config["train_file"], config["label_file"], mode="train"
    )
    val_datasets = MeDSLIP_Dataset(
        config["valid_file"], config["label_file"], mode="train"
    )
    if args.computing == "parallel":
        # shuffl
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_datasets, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_datasets, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_datasets)
        val_sampler = torch.utils.data.RandomSampler(val_datasets)
    train_dataloader = DataLoader(
        train_datasets,
        batch_size=config["batch_size"],
        num_workers=30,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=None,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_datasets,
        batch_size=config["batch_size"],
        num_workers=30,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=None,
        drop_last=True,
    )

    print("Creating book")
    json_book = json.load(open(config["pathology_book"], "r"))
    pathology_book = [json_book[i] for i in json_book]
    anatomy_list = [
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
    anatomy_book = []
    for i in anatomy_list:
        anatomy_book.append("It is located at " + i + ". ")

    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])
    anatomy_book_tokenizer = get_tokenizer(tokenizer, anatomy_book).to(device)
    pathology_book_tokenizer = get_tokenizer(tokenizer, pathology_book).to(device)
    print("Creating model")
    model = MeDSLIP(
        config, anatomy_book_tokenizer, pathology_book_tokenizer, mode="train"
    )
    model = model.to(device)
    if args.computing == "parallel":
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(state_dict)
        print("load checkpoint from %s" % args.checkpoint)

    print("Start training")
    start_time = time.time()

    writer = SummaryWriter(os.path.join(args.output_dir, "log"))
    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)
        train_stats = train(
            model,
            train_dataloader,
            optimizer,
            epoch,
            warmup_steps,
            device,
            lr_scheduler,
            args,
            config,
            writer,
        )

        for k, v in train_stats.items():
            train_loss_epoch = v

        writer.add_scalar("loss/train_loss_epoch", float(train_loss_epoch), epoch)
        writer.add_scalar("loss/leaning_rate", lr_scheduler._get_lr(epoch)[0], epoch)

        val_loss = valid(model, val_dataloader, epoch, device, config, writer)
        writer.add_scalar("loss/val_loss_epoch", val_loss, epoch)

        if utils.is_main_process():
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "val_loss": val_loss.item(),
            }
            save_obj = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, "checkpoint_state.pth"))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % 1 == 0 and epoch > 15:
            save_obj = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(
                save_obj,
                os.path.join(args.output_dir, "checkpoint_" + str(epoch) + ".pth"),
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="PreTrain_MeDSLIP/configs/Pretrain_MeDSLIP.yaml"
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", default="runs/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument(
        "--computing", type=str, default="single", help="number of gpus"
    )
    args = parser.parse_args()
    import datetime

    args.output_dir = os.path.join(
        args.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    gpus = torch.cuda.device_count()
    if gpus > 1:
        args.computing = "parallel"

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    if args.computing == "parallel":
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    main(args, config)
