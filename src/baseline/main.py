# coding: utf-8 -*-


import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

import criteria
import models
from utils import ACREDataset

torch.backends.cudnn.benchmark = True


def check_paths(args):
    try:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        new_log_dir = os.path.join(args.log_dir, time.ctime().replace(" ", "-"))
        args.log_dir = new_log_dir
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args, device):
    def train_epoch(epoch, steps):
        model.train()
        loss_avg = 0.0
        acc_avg = 0.0
        counter = 0
        train_loader_iter = iter(train_loader)
        for _ in trange(len(train_loader_iter)):
            steps += 1
            counter += 1
            images, targets, q_types = next(train_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(outputs, targets)
            acc_avg += acc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch {}, Train Avg Loss: {:.6f}, Train Avg Acc: {:.4f}".format(epoch, loss_avg / float(counter), acc_avg / float(counter)))

        return steps

    def validate_epoch(epoch, steps):
        model.eval()
        loss_avg = 0.0
        acc_avg = 0.0
        total = 0
        val_loader_iter = iter(val_loader)
        for _ in trange(len(val_loader_iter)):
            images, targets, q_types = next(val_loader_iter)
            batch_size = images.shape[0]
            total += batch_size
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_avg += loss.item() * batch_size * 4
            acc, _ = criteria.calculate_correct(outputs, targets)
            acc_avg += acc.item()
        total *= 4
        print("Epoch {}, Valid Avg Loss: {:.6f}, Valid Avg Acc: {:.4f}".format(epoch, loss_avg / float(total), acc_avg / float(total) * 100))
    
    def test_epoch(epoch, steps):
        model.eval()
        loss_avg = 0.0
        acc_avg = 0.0
        total = 0
        test_loader_iter = iter(test_loader)
        total_q_types = torch.zeros(4)
        acc_q_types = torch.zeros(4)
        acc_q = 0
        for _ in trange(len(test_loader_iter)):
            images, targets, q_types = next(test_loader_iter)
            batch_size = images.shape[0]
            total += batch_size
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_avg += loss.item() * batch_size * 4
            acc, correct_matrix = criteria.calculate_correct(outputs, targets)
            acc_avg += acc.item()
            total_q_types += torch.sum(q_types, dim=(0, 1))
            acc_q_types += torch.sum(correct_matrix.view(-1, 4, 1) * q_types, dim=(0, 1))
            acc_q += torch.sum(torch.sum(correct_matrix.view(-1, 4), dim=1) > 3.5).item()
        total *= 4
        percentage = acc_q_types / total_q_types
        print("Epoch {}, Test  Avg Loss: {:.6f}, Test  Avg Acc: {:.4f}, Test Q Acc: {:.4f}".format(epoch, 
                                                                                                   loss_avg / float(total), 
                                                                                                   acc_avg / float(total) * 100,
                                                                                                   acc_q / float(total / 4) * 100))
        print("Direct: {:.4f}, Indirect: {:.4f}, Screen_off: {:.4f}, Potential: {:.4f}".format(percentage[0] * 100, 
                                                                                               percentage[1] * 100, 
                                                                                               percentage[2] * 100,
                                                                                               percentage[3] * 100))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    criterion = criteria.cross_entropy_loss

    model = getattr(models, args.model)()
    if args.cuda and args.multigpu and torch.cuda.device_count() > 1:
        print("Running the model on {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DistributedDataParallel(model)
    model.to(device)
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], args.lr)

    train_set = ACREDataset(args.dataset, "train",  (args.img_width, args.img_height))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_set = ACREDataset(args.dataset, "val", (args.img_width, args.img_height))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_set = ACREDataset(args.dataset, "test", (args.img_width, args.img_height))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    total_steps = 0

    for epoch in range(args.epochs):
        total_steps = train_epoch(epoch, total_steps)
        with torch.no_grad():
            validate_epoch(epoch, total_steps)
            test_epoch(epoch, total_steps)
        
        # save checkpoint
        model.eval().cpu()
        ckpt_model_name = "epoch_{}_batch_{}_seed_{}_lr_{}.pth".format(
            epoch, 
            args.batch_size, 
            args.seed,
            args.lr)
        ckpt_file_path = os.path.join(args.checkpoint_dir, ckpt_model_name)
        torch.save(model.state_dict(), ckpt_file_path)
        model.to(device)
    
    # save final model
    model.eval().cpu()
    save_model_name = "Final_epoch_{}_batch_{}_seed_{}_lr_{}.pth".format(
        epoch, 
        args.batch_size, 
        args.seed,
        args.lr)
    save_file_path = os.path.join(args.save_dir, save_model_name)
    torch.save(model.state_dict(), save_file_path)

    print("Done. Model saved.")


def test(args, device):
    def test_epoch():
        model.eval()
        correct_avg = 0.0
        test_loader_iter = iter(test_loader)
        total = 0
        total_q_types = torch.zeros(4)
        acc_q_types = torch.zeros(4)
        acc_q = 0
        for _ in range(len(test_loader_iter)):
            images, targets, q_types = next(test_loader_iter)
            total += images.shape[0]
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            correct_num, correct_matrix = criteria.calculate_correct(outputs, targets)
            correct_avg += correct_num.item()
            total_q_types += torch.sum(q_types, dim=(0, 1))
            acc_q_types += torch.sum(correct_matrix.view(-1, 4, 1) * q_types, dim=(0, 1))
            acc_q += torch.sum(torch.sum(correct_matrix.view(-1, 4), dim=1) > 3.5).item()
        percentage = acc_q_types / total_q_types
        total *= 4
        print("Test Avg Acc: {:.4f}, Test Q Acc: {:.4f}".format(correct_avg / float(total) * 100), acc_q / float(total / 4) * 100)
        print("Direct: {:.4f}, Indirect: {:.4f}, Screen_off: {:.4f}, Potential: {:.4f}".format(percentage[0] * 100, 
                                                                                               percentage[1] * 100, 
                                                                                               percentage[2] * 100,
                                                                                               percentage[3] * 100))

    model = getattr(models, args.model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    test_set = ACREDataset(args.dataset, "test", (args.img_width, args.img_height))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    with torch.no_grad():
        test_epoch()    


def main(): 
    main_arg_parser = argparse.ArgumentParser()
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    
    train_arg_parser = subparsers.add_parser("train", help="parser for training")
    train_arg_parser.add_argument("--epochs", type=int, default=200,
                                  help="the number of training epochs")
    train_arg_parser.add_argument("--batch_size", type=int, default=32,
                                  help="size of batch")
    train_arg_parser.add_argument("--seed", type=int, default=12345,
                                  help="random number seed")
    train_arg_parser.add_argument("--device", type=int, default=0,
                                  help="device index for GPU; if GPU unavailable, leave it as default")
    train_arg_parser.add_argument("--num_workers", type=int, default=2,
                                  help="number of workers for data loader")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="dataset path")
    train_arg_parser.add_argument("--checkpoint_dir", type=str, default="./runs/ckpt/",
                                  help="checkpoint save path")
    train_arg_parser.add_argument("--save_dir", type=str, default="./runs/save/",
                                  help="final model save path")
    train_arg_parser.add_argument("--log_dir", type=str, default="./runs/log/",
                                  help="log save path")
    train_arg_parser.add_argument("--img_width", type=int, default=80,
                                  help="image size for training")                            
    train_arg_parser.add_argument("--img_height", type=int, default=60,
                                  help="image size for training")
    train_arg_parser.add_argument("--lr", type=float, default=1e-4,
                                  help="learning rate")
    train_arg_parser.add_argument("--multigpu", type=int, default=0,
                                  help="whether to use multi gpu")
    train_arg_parser.add_argument("--model", type=str, default="WReN",
                                  help="the model to train")
    
    test_arg_parser = subparsers.add_parser("test", help="parser for testing")
    test_arg_parser.add_argument("--batch_size", type=int, default=32,
                                 help="size of batch")
    test_arg_parser.add_argument("--device", type=int, default=0,
                                 help="device index for GPU; if GPU unavailable, leave it as default")
    test_arg_parser.add_argument("--num_workers", type=int, default=2,
                                 help="number of workers for data loader")
    test_arg_parser.add_argument("--dataset", type=str, required=True,
                                 help="dataset path")
    test_arg_parser.add_argument("--model_path", type=str, required=True,
                                 help="path to a trained model")
    test_arg_parser.add_argument("--img_width", type=int, default=80,
                                 help="image size for training")                            
    test_arg_parser.add_argument("--img_height", type=int, default=60,
                                 help="image size for training")
    test_arg_parser.add_argument("--model", type=str, default="WReN",
                                 help="the model to train")

    args = main_arg_parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    if args.subcommand is None:
        print("ERROR: Specify train or test")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args, device)
    elif args.subcommand == "test":
        test(args, device)
    else:
        print("ERROR: Unknown subcommand")
        sys.exit(1)


if __name__ == "__main__":
    main()
