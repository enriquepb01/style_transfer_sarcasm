import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import preprocessing
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelBinarizer
import gc

#system settings
force_reload = False
use_CPU = False

#hyperparameters
lr = 3e-5
reg_val = 1e-2
batch_size = 2
n_epochs = 5
accum_steps = 4

def linear_block(n_in, n_hidden, drate):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.Dropout(p=drate)
    )

class bert_classifier(nn.Module):
    def __init__(self, n_hidden1=256, n_hidden2=64, drate=.125):
        super().__init__()

        self.bert = torch.hub.load('huggingface/pytorch-transformers', 
                                   'model', 
                                   'bert-base-uncased', 
                                   force_reload=force_reload)
        
        self.block1 = linear_block(768, n_hidden1, drate)
        self.block2 = linear_block(n_hidden1, n_hidden2, drate)
        self.out_preact = nn.Linear(n_hidden2, 2)
        self.out = nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        _, y = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        y = self.block1(y)
        y = self.block2(y)
        y = self.out_preact(y)
        y = self.out(y)
        return y

class token_loader:
    def __init__(self, data, target, max_length):
        self.data = data
        self.target = target
        self.max_length = max_length

        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 
                                        'tokenizer', 
                                        'bert-base-uncased',
                                        force_reload=force_reload)
    
    def __len__(self):
        return len(self.data)  
    def __getitem__(self, item):
        data = str(self.data[item])
        data = " ".join(data.split())
        inputs = self.tokenizer.encode_plus(
            data, 
            None,
            add_special_tokens=True,
            truncation=True,
            max_length = self.max_length,
            padding='max_length'
        )
        ids = inputs["input_ids"]
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        padding_length = self.max_length - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.long)
        }

def label_encoding(x):
    return F.one_hot(x.type(torch.int64),num_classes=2).type(torch.float)

def BCE(output, label):
    label = label_encoding(label)
    return nn.BCELoss()(output, label)

def train(model, dataloader, loss_fn, accum_steps, optimizer, device):
    model.to(device)
    model.train()
    running_loss = 0; running_acc = 0
    with tqdm(total=len(dataloader), desc=f"Train", unit="Batch") as pbar:
        for bi, d in enumerate(dataloader):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.int64)

            output = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
        
            loss = loss_fn(output, targets)
            running_loss += loss.item()
            running_acc += (output.argmax(1) == targets).float().mean().item()
            loss = loss / accum_steps
            loss.backward()
            if((bi + 1) % accum_steps == 0) or (bi + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * running_acc/(bi+1)})
            pbar.update()
            pbar.refresh()

    return running_loss / len(dataloader), running_acc / len(dataloader)            

def validate(model, dataloader, loss_fn):
    model.to(device)
    model.eval()
    running_loss = 0; running_acc = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=f"Val", unit="Batch") as pbar:
            for bi, d in enumerate(dataloader):
                ids = d['ids']
                token_type_ids = d['token_type_ids']
                mask = d['mask']
                targets = d['targets']

                ids = ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.int64)
                
                output = model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids
                )

                loss = loss_fn(output, targets)
                running_loss += loss.item()
                running_acc += (output.argmax(1) == targets).float().mean().item()
                pbar.set_postfix({'loss': loss.item(), 'acc': 100. * running_acc / (bi+1)})
                pbar.update()

    return running_loss / len(dataloader), running_acc / len(dataloader)

max_length = 21
df = preprocessing.clean_data("train-balanced-sarcasm.csv", 
                              max_samples=100000, 
                              min_length=4, 
                              max_length=max_length)
train_df, val_df, test_df = preprocessing.train_validate_test_split(df)

print('Train: {}, Validation: {}, Test: {}'.format(train_df.shape, val_df.shape, test_df.shape))

train_dataset = token_loader(data=train_df.comment.values, 
                            target=train_df.label.values, 
                            max_length=512)
test_dataset = token_loader(data=test_df.comment.values,
                           target=test_df.label.values,
                           max_length=512)
val_dataset = token_loader(data=val_df.comment.values,
                          target=val_df.label.values,
                          max_length=512)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = bert_classifier()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_val)

if torch.cuda.is_available() and not use_CPU:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

train_loss_history = []; train_acc_history = []
val_loss_history = []; val_acc_history = []
best_acc = -1

for epoch in range(n_epochs):
    print(f"\nEpoch {epoch+1} of {n_epochs}")
    train_loss, train_acc = train(model, train_loader, BCE, accum_steps, optimizer, device)
    train_loss, train_acc = validate(model, eval_loader, BCE, device)
    val_loss, val_acc = validate(model, val_loader, BCE, device)
    train_loss_history.append(train_loss); train_acc_history.append(train_acc)
    val_loss_history.append(val_loss); val_acc_history.append(val_acc)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model.bin")

print(f"Best & Final Accuracy: {best_acc}\nWe're done here.")