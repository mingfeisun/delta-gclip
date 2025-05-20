import os
import torch
import time
import pickle
import random
import argparse
import numpy as np

from datetime import datetime

import sys
sys.path.insert(1, '../optim')
from dGClip import dGClip

from torch.utils.tensorboard import SummaryWriter

from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def load_data(batch_size):
    from datasets import load_dataset
    emotions = load_dataset('emotion')
    emotions.set_format('pandas')

    df_train = emotions['train'][:]
    df_val = emotions['validation'][:]
    df_test = emotions['test'][:]

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

    train_texts = df_train.text.to_numpy()
    train_y = df_train.label.to_numpy()
    val_texts = df_val.text.to_numpy()
    val_y = df_val.label.to_numpy()
    test_texts = df_test.text.to_numpy()
    test_y = df_test.label.to_numpy()

    train_x = np.array([tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in train_texts])
    val_x = np.array([tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in val_texts])
    test_x = np.array([tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in test_texts])

    train_m = train_x > 0
    val_m = val_x > 0
    test_m = test_x > 0

    train_x = torch.tensor(train_x)
    test_x = torch.tensor(test_x)
    val_x = torch.tensor(val_x)
    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)
    val_y = torch.tensor(val_y)
    train_m = torch.tensor(train_m)
    test_m = torch.tensor(test_m)
    val_m = torch.tensor(val_m)

    train_data = TensorDataset(train_x, train_m, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_x, val_m, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    num_labels = len(set(train_y.tolist()))

    return train_dataloader, val_dataloader, test_x, test_m, test_y, tokenizer, num_labels


def load_model(num_labels, device):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,output_attentions=False, output_hidden_states=False)
    model = model.to(device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters:{count_parameters(model)}')
    return model


def train_bert(args):
    device = torch.device(f'cuda:{args.device}') # Select best available device
    print(device)

    learning_rate = 1e-5
    adam_epsilon = 1e-8
    batch_size = 32

    seed = args.seed
    if args.seed is None:
        seed = np.random.randint(1e6) # different seeds for each process

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_dataloader, val_dataloader, test_x, test_m, test_y, tokenizer, num_labels = load_data(batch_size)
    model = load_model(num_labels, device)

    if args.optim == 'adamw':
        no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #  #   {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #  #   'weight_decay_rate': 0.2},
        #  # {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #    # 'weight_decay_rate': 0.0}
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon, weight_decay=args.weight_decay)

    elif args.optim == 'dgclip':
        optimizer=dGClip(model.parameters(), lr=args.lr, gamma=args.gamma, delta=args.delta, weight_decay=args.weight_decay)

    else:
        raise ValueError(f"Unknown optimizer: {args.optim}")

    num_epochs = args.epochs

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    test_data = TensorDataset(test_x, test_m)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=f'logs/bert_finetuning.{args.optim}.{args.lr}.{args.gamma}.{args.delta}.{args.weight_decay}.{time_now}_{seed}', comment='bert_finetuning')

    for n in range(num_epochs):
        start_time = time.time()

        clip_times = 0
        model.train()
        train_losses = []
        for k, (mb_x, mb_m, mb_y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            mb_y = mb_y.to(device)

            outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
            loss = outputs[0]
            loss.backward()
            grad_norm = optimizer.step()
            if grad_norm is None:
                grad_norm = 1.0
            clip_times += args.delta < (args.gamma / grad_norm)
            train_losses.append(loss.item())

        writer.add_scalar('train_loss', np.mean(train_losses), n)
        writer.add_scalar('grad_norm', grad_norm, n)
        writer.add_scalar('grad_clip_ratio', clip_times * 1.0 / (k+1), n)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for k, (mb_x, mb_m, mb_y) in enumerate(val_dataloader):
                mb_x = mb_x.to(device)
                mb_m = mb_m.to(device)
                mb_y = mb_y.to(device)
                outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
                loss = outputs[0]
                val_losses.append(loss.item())

        writer.add_scalar('val_loss', np.mean(val_losses), n)

        print (f"\nTrain loss after itaration {n}: {np.mean(train_losses)}")
        print (f"Validation loss after itaration {n}: {np.mean(val_losses)}")
        print (f"Gradient clipping times: {clip_times}")

        outputs = []
        with torch.no_grad():
            for k, (mb_x, mb_m) in enumerate(test_dataloader):
                mb_x = mb_x.to(device)
                mb_m = mb_m.to(device)

                output = model(mb_x, attention_mask=mb_m)
                outputs.append(output[0].to('cpu'))

            outputs = torch.cat(outputs)

            _, predicted_values = torch.max(outputs, 1)
            predicted_values = predicted_values.numpy()
            true_values = test_y.numpy()
            test_acc = np.sum(predicted_values == true_values) / len(true_values)
            print (f"Test accuracy: {test_acc}")
        writer.add_scalar('test_acc', test_acc, n)

        test_losses = []
        with torch.no_grad():
            for k, (mb_x, mb_m, mb_y) in enumerate(val_dataloader):
                mb_x = mb_x.to(device)
                mb_m = mb_m.to(device)
                mb_y = mb_y.to(device)
                outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
                loss = outputs[0]
                test_losses.append(loss.item())
        writer.add_scalar('test_loss', np.mean(test_losses), n)
        print(f"Test loss: {np.mean(test_losses)}")

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Time: {epoch_mins}m {epoch_secs}s')

    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch BERT Fine Tuning')
    parser.add_argument('--optim', default='dgclip', choices=['adamw', 'dgclip', 'sgd'],)
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--gamma', default=1.0, type=float, help='gamma')
    parser.add_argument('--delta', default=0.001, type=float, help='delta')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed (default: None)')
    parser.add_argument('--device', default=0, type=int, help='GPU device')
    args = parser.parse_args()

    model, tokenizer = train_bert(args)
    # save_model(model, tokenizer, train_losses, val_losses)
    # plot_loss()
    # test_model(model, test_x, test_m, test_y)