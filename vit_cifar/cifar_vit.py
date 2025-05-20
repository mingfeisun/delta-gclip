import torch
from torch import nn
import argparse
import numpy as np

from datetime import datetime

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(1, '../optim')

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def read_cifar(path, if_autoencoder=False):
    # code from https://github.com/jeonsworld/MLP-Mixer-Pytorch/blob/main/utils/data_utils.py
    image_size = 64
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_data = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_train, target_transform=transforms.Lambda(
            lambda y: torch.zeros(10).scatter_(0, torch.tensor(y), value=1)
        )
    )
    test_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test, target_transform=transforms.Lambda(
            lambda y: torch.zeros(10).scatter_(0, torch.tensor(y), value=1)
        )
    )

    class DataSets(object):
        pass

    data_sets = DataSets()

    if if_autoencoder:
        train_data.targets = train_data.data
        test_data.targets = test_data.data

    data_sets.train = train_data
    data_sets.test = test_data

    return data_sets

def load_dataset(batch_size):
    dataset = read_cifar("data/", if_autoencoder=False)

    ## Dataset
    train_dataset = dataset.train
    test_dataset = dataset.test
    print("Number of training samples: ", len(train_dataset))
    print("Number of testing samples: ", len(test_dataset))
    print("Image shape: ", train_dataset[0][0].shape)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    aux_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, aux_loader, test_loader

def class_accuracy(predictions, labels):
    y = torch.max(predictions, 1)[1]
    y_labels = torch.max(labels, 1)[1]

    return torch.mean(y.eq(y_labels).float())

def train_vit(args):
    seed = args.seed
    if seed is None:
        seed = np.random.randint(1e6) # different seeds for each process

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, _, test_loader = load_dataset(64)

    if args.vit_type == 'tiny':
        from vision_transformer import vit_tiny
        model = vit_tiny(num_classes=10)
    elif args.vit_type == 'small':
        from vision_transformer import vit_small
        model = vit_small(num_classes=10)
    else:
        raise ValueError(f"Unknown ViT type: {args.vit_type}")

    print('Number of parameters: ', count_vars(model))

    device = torch.device(f'cuda:{args.device}') # Select best available device

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim == 'adam':
        opt = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optim == 'dgclip':
        from dGClip import dGClip
        opt = dGClip(
            model.parameters(),
            lr=args.lr,
            gamma=args.gamma,
            delta=args.delta,
            weight_decay=args.weight_decay,
        )
    elif args.optim == 'sgd':
        opt = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else: 
        raise ValueError(f"Unknown optimizer: {args.optim}")

    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=f'logs/vit-{args.vit_type}_cifar.{args.optim}.{args.lr}.{args.gamma}.{args.delta}.{args.weight_decay}.{time_now}_{seed}', comment='vit_cifar')

    import time
    st = time.time()
    eval_time = 0

    n_steps = 0
    n_test_steps = 0
    from tqdm import tqdm
    for epoch in range(1, args.epochs + 1):
        with tqdm(train_loader, unit="batch") as tepoch:
            running_loss = 0
            running_acc = 0
            clip_times = 0

            model.train()
            for n, (batch_data, batch_labels) in enumerate(tepoch, start=1):
                tepoch.set_description(f"Epoch {epoch}")

                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                opt.zero_grad()
                output = model(batch_data)
                loss = criterion(output, batch_labels)
                loss.backward()
                grad_norm = opt.step()
                if grad_norm is None:
                    grad_norm = 1.0
                clip_times += args.delta < args.gamma / grad_norm

                acc = class_accuracy(output, batch_labels)

                running_loss += loss.item()
                running_acc += acc.item()

                et = time.time()     

                n_steps += 1

            running_test_loss = 0
            running_test_acc = 0
            model.eval()
            for m, (test_batch_data, test_batch_labels) in enumerate(test_loader, start=1):
                test_batch_data, test_batch_labels = test_batch_data.to(device), test_batch_labels.to(device)

                test_output = model(test_batch_data)

                test_loss = criterion(test_output, test_batch_labels).item()
                test_acc = class_accuracy(test_output, test_batch_labels).item()

                running_test_loss += test_loss
                running_test_acc += test_acc

            running_test_loss /= m+1
            running_test_acc /= m+1

            n_test_steps += 1
            tepoch.set_postfix(acc=100 * running_acc / n, test_acc=running_test_acc * 100)
            writer.add_scalar('train_loss', running_loss / n, epoch)
            writer.add_scalar('train_acc', running_acc / n, epoch)
            writer.add_scalar('test_loss', running_test_loss, epoch)
            writer.add_scalar('test_acc', running_test_acc, epoch)
            writer.add_scalar('grad_norm', grad_norm, epoch)
            writer.add_scalar('grad_clip_ratio', clip_times * 1.0 / n, epoch)

            print(f"Epoch {epoch}, step {n}, loss: {running_loss / n:.3f}, test_loss: {running_test_loss:.3f}, acc: {running_acc / n:.3f}, test_acc: {running_test_acc:.3f}, grad_norm: {grad_norm:.3f}")
            eval_time += time.time() - et

            epoch_time = time.time() - st - eval_time
            tepoch.set_postfix(loss=running_loss / n, test_loss=running_test_loss, epoch_time=epoch_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR ViT Training')
    parser.add_argument('--vit_type', default='tiny', choices=['tiny', 'small'],)
    parser.add_argument('--optim', default='dgclip', choices=['adam', 'dgclip', 'sgd'],)
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--gamma', default=1.0, type=float, help='gamma')
    parser.add_argument('--delta', default=0.001, type=float, help='delta')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed (default: None)')
    parser.add_argument('--device', default=0, type=int, help='GPU device')
    args = parser.parse_args()

    train_vit(args)