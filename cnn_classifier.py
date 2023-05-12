import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
import preprocessing
from tqdm import tqdm
import pickle

def fully_connected(n_in, n_hidden, drate):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.Dropout(p=drate)
    )

class BertEmbedding(nn.Module):
    def __init__(self, trainable):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = trainable

    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask, return_dict=True)
        hiddens = out.hidden_states # retrieve hidden states
        hiddens = torch.stack(hiddens, dim=0) # shape (13 layers, batch size, sequence length, 768)
        hiddens = hiddens[9:, :, :, :].mean(dim=0) # average over last four layers; shape (batch size, sequence length, 768)
        return hiddens

class BertCNN(nn.Module):
    def __init__(self, train_bert, sentence_max_size):
        super().__init__()
        self.bert = BertEmbedding(trainable=train_bert)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 1, (2, 768))
        self.conv2 = nn.Conv2d(1, 1, (3, 768))
        self.conv3 = nn.Conv2d(1, 1, (4, 768))
        self.conv4 = nn.Conv2d(1, 1, (5, 768))

        # Max pool layers
        self.maxpool1 = nn.MaxPool2d((self.sentence_max_size-2+1, 1))
        self.maxpool2 = nn.MaxPool2d((self.sentence_max_size-3+1, 1))
        self.maxpool3 = nn.MaxPool2d((self.sentence_max_size-4+1, 1))
        self.maxpool4 = nn.MaxPool2d((self.sentence_max_size-5+1, 1))

        # Classifier network
        self.out = nn.Linear(4, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask):
        y = self.bert(ids, mask=mask)
        y = torch.unsqueeze(y, 1)
        
        y1 = F.relu(self.conv1(y))
        y2 = F.relu(self.conv2(y))
        y3 = F.relu(self.conv3(y))
        y4 = F.relu(self.conv4(y))

        y1 = self.maxpool1(y1)
        y2 = self.maxpool2(y2)
        y3 = self.maxpool3(y3)
        y4 = self.maxpool4(y4)

        y = torch.cat((y1, y2, y3, y4), -1)
        y = self.out(y)
        y = self.sigmoid(y)

        return y.reshape((y.size(0),2))

class token_loader:
    def __init__(self, data, target, max_length):
        self.data = data
        self.target = target
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
        padding_length = self.max_length - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
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
    with tqdm(total=len(dataloader), desc=f"Train", unit="Batch") as prog:
        for bi, d in enumerate(dataloader):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["targets"]
        
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.int64)

            output = model( # forward pass
                ids=ids,
                mask=mask
            )
        
            loss = loss_fn(output, targets)
            running_loss += loss.item()
            running_acc += (output.argmax(1) == targets).float().mean().item()
            loss = loss / accum_steps
            loss.backward() # backward pass

            if((bi + 1) % accum_steps == 0) or (bi + 1 == len(dataloader)): # gradientt accumulation
                optimizer.step()
                optimizer.zero_grad()

            prog.set_postfix({'loss': loss.item(), 'acc': 100 * running_acc/(bi+1)})
            prog.update()
            prog.refresh()

    return running_loss / len(dataloader), running_acc / len(dataloader)            

def validate(model, dataloader, loss_fn, device, tag='Val'):
    model.to(device)
    model.eval()
    running_loss = 0; running_acc = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=tag, unit="Batch") as prog:
            for bi, d in enumerate(dataloader):
                ids = d['ids']
                mask = d['mask']
                targets = d['targets']

                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.int64)
                
                output = model(
                    ids=ids,
                    mask=mask
                )

                loss = loss_fn(output, targets)
                running_loss += loss.item()
                running_acc += (output.argmax(1) == targets).float().mean().item()
                prog.set_postfix({'loss': loss.item(), 'acc': 100. * running_acc / (bi+1)})
                prog.update()

    return running_loss / len(dataloader), running_acc / len(dataloader)

def run(max_samples, max_length, n_epochs,
        batch_size, accum_steps, train_bert,
        lr, reg_val,  
        path, use_CPU=False, halve_lr_at=0):
    
    df = preprocessing.clean_data("train-balanced-sarcasm.csv", 
                                max_samples=max_samples, 
                                min_length=4, 
                                max_length=max_length)
    
    train_df, val_df, test_df = preprocessing.train_validate_test_split(df)

    print('Train: {}, Validation: {}, Test: {}'.format(train_df.shape, val_df.shape, test_df.shape))

    train_dataset = token_loader(data=train_df.comment.values, 
                                target=train_df.label.values, 
                                max_length=max_length)
    test_dataset = token_loader(data=test_df.comment.values,
                            target=test_df.label.values,
                            max_length=max_length)
    val_dataset = token_loader(data=val_df.comment.values,
                            target=val_df.label.values,
                            max_length=max_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BertCNN(train_bert=train_bert, sentence_max_size=max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_val)

    if torch.cuda.is_available() and not use_CPU:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Using device: ", torch.cuda.get_device_name(device))

    train_loss_history = []; train_acc_history = []
    val_loss_history = []; val_acc_history = []
    best_acc = -1

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1} of {n_epochs}")
        if halve_lr_at > 0 and epoch > halve_lr_at:
            print(f"Reducing Learning rate from {lr} to {lr/4}")
            optimizer.param_groups[0]['lr'] /= 4
        train_loss, train_acc = train(model, train_loader, BCE, accum_steps, optimizer, device)
        train_loss, train_acc = validate(model, eval_loader, BCE, device, tag='Train Eval')
        val_loss, val_acc = validate(model, val_loader, BCE, device)
        train_loss_history.append(train_loss); train_acc_history.append(train_acc)
        val_loss_history.append(val_loss); val_acc_history.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), path + "/model_best.bin")

    torch.save(model.state_dict(), path + "/model_last.bin")
    history = {
            "train_loss": train_loss_history,
            "train_accuracy": train_acc_history,
            "validation_loss": val_loss_history,
            "validation_accuracy": val_acc_history     
        }
    with open(path + "/history_last.pickle", 'wb') as f:
        pickle.dump(history, f)

    print(f"\nBest & Final Accuracy: {best_acc}\nWe're done here.")

if __name__ == "__main__":

    train_kwargs = {
        'train_bert': True,
        'use_CPU': False,
        'path': "run3",
        'max_samples': 100000,
        'lr': 3e-5,
        'reg_val': 1e-2,
        'batch_size': 16,
        'accum_steps': 1,
        'halve_lr_at': 2,
        'n_epochs': 4,
        'max_length': 21
        } 
    
    run(**train_kwargs)