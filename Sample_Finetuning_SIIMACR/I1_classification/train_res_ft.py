import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from test_res_ft import test
from tensorboardX import SummaryWriter
import utils
from models.resnet import ModelRes_ft
from test_res_ft import test
from dataset.dataset_siim_acr import SIIM_ACR_Dataset
from scheduler import create_scheduler
from optim import create_optimizer
import warnings
warnings.filterwarnings("ignore")


def train(
    model,
    data_loader,
    optimizer,
    criterion,
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
    metric_logger.update(loss=1.0)
    metric_logger.update(lr=scheduler._get_lr(epoch)[0])

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    scalar_step = epoch * len(data_loader)

    for i, sample in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        image = sample["image"]
        label = sample["label"].float().to(device)  # batch_size,num_class
        input_image = image.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred_class = model(input_image)  # batch_size,num_class

        loss = criterion(pred_class, label)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss/loss", loss, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        metric_logger.update(lr=scheduler._get_lr(epoch)[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.6f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }  # ,loss_epoch.mean()


def valid(model, data_loader, criterion, epoch, device, config, writer):
    model.eval()
    val_scalar_step = epoch * len(data_loader)
    val_losses = []
    for i, sample in enumerate(data_loader):
        image = sample["image"]
        label = sample["label"].float().to(device)
        input_image = image.to(device, non_blocking=True)
        with torch.no_grad():
            pred_class = model(input_image)
            val_loss = criterion(pred_class, label)
            val_losses.append(val_loss.item())
            writer.add_scalar("val_loss/loss", val_loss, val_scalar_step)
            val_scalar_step += 1
    avg_val_loss = np.array(val_losses).mean()
    return avg_val_loss


def main(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type("torch.FloatTensor")

    start_epoch = 0
    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]

    #### Dataset ####
    print("Creating dataset")
    train_dataset = SIIM_ACR_Dataset(config["train_file"], percentage=config["percentage"])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=30,
        pin_memory=True,
        sampler=None,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
    )

    val_dataset = SIIM_ACR_Dataset(config["valid_file"], is_train=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=30,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )
    print(len(train_dataset), len(val_dataset))

    model = ModelRes_ft(res_base_model="resnet50", out_size=1, use_base=args.use_base)  
    model = nn.DataParallel(
        model, device_ids=[i for i in range(torch.cuda.device_count())]
    )
    model = model.to(device)

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    criterion = nn.BCEWithLogitsLoss()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(state_dict)
        print("load checkpoint from %s" % args.checkpoint)
    elif args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path, map_location="cpu")
        state_dict = checkpoint["model"]
        model_dict = model.state_dict()
        model_checkpoint = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(model_checkpoint)
        model.load_state_dict(model_dict)
        print("load pretrain_path from %s" % args.pretrain_path)

    print("Start training")
    start_time = time.time()

    best_test_auc = 0.0
    writer = SummaryWriter(os.path.join(args.output_dir, "log"))
    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)
        train_stats = train(
            model,
            train_dataloader,
            optimizer,
            criterion,
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

        val_loss = valid(
            model, val_dataloader, criterion, epoch, device, config, writer
        )
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
        
        test_auc = test(args, config)
        print(best_test_auc, test_auc)
        if test_auc > best_test_auc:
            save_obj = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, "best_test.pth"))
            best_test_auc = test_auc
            args.model_path = os.path.join(args.output_dir, "checkpoint_state.pth")
            
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(
                    "The average AUROC is {AUROC_avg:.4f}".format(AUROC_avg=test_auc)
                    + "\n"
                )

        if epoch % 20 == 1 and epoch > 1:
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
    parser.add_argument("--config", default="Sample_Finetuning_SIIMACR/I1_classification/configs/Res_train.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--pretrain_path", default="checkpoint_state.pth")
    parser.add_argument("--output_dir", default="Sample_Finetuning_SIIMACR/I1_classification/runs/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--use_base", type=bool, default=True)
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    args.output_dir = os.path.join(args.output_dir, str(config["percentage"]))
    from datetime import datetime
    args.output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.model_path = os.path.join(args.output_dir, "checkpoint_state.pth")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    main(args, config)
