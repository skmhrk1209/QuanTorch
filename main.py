import torch
from torch import distributed
from torch import backends
from torch import cuda
from torch import utils
from torch import optim
from torch import nn
from torchvision import transforms
from datasets import ImageDataset
from models import ConvNet
from distributed import *
from quant_utils import *
from utils import *
import numpy as np
import argparse
import json
import os


def main(args):

    init_process_group(backend='nccl')

    with open(args.config) as file:
        config = json.load(file)
        config.update(vars(args))
        config = apply_dict(Dict, config)

    backends.cudnn.benchmark = True
    backends.cudnn.fastest = True

    cuda.set_device(distributed.get_rank() % cuda.device_count())

    train_dataset = ImageDataset(
        root=config.train_root,
        meta=config.train_meta,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])
    )
    val_dataset = ImageDataset(
        root=config.val_root,
        meta=config.val_meta,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])
    )

    train_sampler = utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = utils.data.distributed.DistributedSampler(val_dataset)

    train_data_loader = utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.local_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_data_loader = utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.local_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model = ConvNet(
        conv_params=[
            Dict(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=2, bias=False),
            Dict(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=2, bias=False),
        ],
        linear_params=[
            Dict(in_channels=3136, out_channels=1024, kernel_size=1, bias=False),
            Dict(in_channels=1024, out_channels=10, kernel_size=1, bias=True),
        ]
    )

    config.global_batch_size = config.local_batch_size * distributed.get_world_size()
    config.optimizer.lr *= config.global_batch_size / config.global_batch_denom
    optimizer = optim.Adam(model.parameters(), **config.optimizer)

    epoch = 0
    global_step = 0
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint))
        model.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        epoch = checkpoint.last_epoch + 1
        global_step = checkpoint.global_step

    def train(data_loader):
        nonlocal global_step
        model.train()
        for images, labels in data_loader:
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward(retain_graph=True)
            average_gradients(model.parameters())
            optimizer.step()
            predictions = logits.topk(k=1, dim=1)[1].squeeze()
            accuracy = torch.mean((predictions == labels).float())
            average_tensors([loss, accuracy])
            global_step += 1
            dprint(f'[training] epoch: {epoch} global_step: {global_step} '
                   f'loss: {loss:.4f} accuracy: {accuracy:.4f}')

    @torch.no_grad()
    def validate(data_loader):
        model.eval()
        losses = []
        accuracies = []
        for images, labels in data_loader:
            images = images.cuda()
            labels = labels.cuda()
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            predictions = logits.topk(k=1, dim=1)[1].squeeze()
            accuracy = torch.mean((predictions == labels).float())
            average_tensors([loss, accuracy])
            losses.append(loss)
            accuracies.append(accuracy)
        loss = torch.mean(torch.stack(losses)).item()
        accuracy = torch.mean(torch.stack(accuracies)).item()
        dprint(f'[validation] epoch: {epoch} global_step: {global_step} '
               f'loss: {loss:.4f} accuracy: {accuracy:.4f}')

    @torch.no_grad()
    def feed(data_loader):
        model.eval()
        for images, _ in data_loader:
            images = images.cuda()
            logits = model(images)

    def save():
        if not distributed.get_rank():
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                last_epoch=epoch,
                global_step=global_step
            ), os.path.join('checkpoints', f'epoch_{epoch}'))

    if config.training:
        model.cuda()
        broadcast_tensors(model.state_dict().values())
        for epoch in range(epoch, config.num_training_epochs):
            train_sampler.set_epoch(epoch)
            train(train_data_loader)
            validate(val_data_loader)
            save()

    if config.validation:
        model.cuda()
        broadcast_tensors(model.state_dict().values())
        validate(val_data_loader)

    if config.quantization:
        model.cuda()
        broadcast_tensors(model.state_dict().values())
        with QuantizationEnabler(model):
            with BatchStatsUser(model):
                for epoch in range(epoch, config.num_quantization_epochs):
                    train_sampler.set_epoch(epoch)
                    train(train_data_loader)
                    validate(val_data_loader)
                    save()
            with AverageStatsUser(model):
                for epoch in range(epoch, config.num_quantization_epochs):
                    train_sampler.set_epoch(epoch)
                    train(train_data_loader)
                    validate(val_data_loader)
                    save()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='QuanTorch MNIST Example')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--quantization', action='store_true')
    args = parser.parse_args()

    main(args)
