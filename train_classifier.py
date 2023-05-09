import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
import preprocessing
from tqdm import tqdm
import pickle

def linear_block(n_in, n_hidden, drate):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.Dropout(p=drate)
    )

class bert_embedding(nn.Module):
    def __init__(self, trainable):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = trainable

    def forward(self, ids, mask, token_type_ids):
        _, _, hidden_states = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        token_embeddings = torch.stack(hidden_states, dim=0).permute(1,2,0,3)
        processed_embeddings = token_embeddings[:,:,9:,:]
        processed_embeddings = processed_embeddings.mean(dim=2).mean(dim=1)
        return processed_embeddings

class bert_classifier(nn.Module):
    def __init__(self, n_hidden1=256, n_hidden2=64, drate=.125):
        super().__init__()
        self.bert = bert_embedding(trainable=True)
        self.block1 = linear_block(768, n_hidden1, drate)
        self.block2 = linear_block(n_hidden1, n_hidden2, drate)
        self.out_preact = nn.Linear(n_hidden2, 2)
        self.out = nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        y = self.bert(ids, mask=mask, token_type_ids=token_type_ids)
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

def validate(model, dataloader, loss_fn, device, tag='Val'):
    model.to(device)
    model.eval()
    running_loss = 0; running_acc = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=tag, unit="Batch") as pbar:
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

def run():
    #system settings
    force_reload = False
    use_CPU = False
    path = "run2"

    #hyperparameters
    max_samples = 100000
    lr = 3e-5
    reg_val = 1e-2
    batch_size = 2
    n_epochs = 5
    accum_steps = 4
    halve_lr = False

    max_length = 21
    
    df = preprocessing.clean_data("train-balanced-sarcasm.csv", 
                                max_samples=max_samples, 
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
    print("Using device: ", torch.cuda.get_device_name(device))

    train_loss_history = []; train_acc_history = []
    val_loss_history = []; val_acc_history = []
    best_acc = -1

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1} of {n_epochs}")
        if epoch > 2 and halve_lr:
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
    run()