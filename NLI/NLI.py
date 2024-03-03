from datasets import load_from_disk
import datasets
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
from loralib import LoRALayer
import math
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union
from health_multimodal.text.model.modelling_cxrbert import CXRBertModel
from torch.utils.data import DataLoader, Dataset
from health_multimodal.text.model.configuration_cxrbert import CXRBertConfig, CXRBertTokenizer 
from health_multimodal.image.model.pretrained import (
    BIOMED_VLP_BIOVIL_T,
    BIOMED_VLP_CXR_BERT_SPECIALIZED,
    BIOVIL_T_COMMIT_TAG,
    CXR_BERT_COMMIT_TAG,
)

class NliDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.data = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data['premise'][idx]
        hypothesis = self.data['hypothesis'][idx]
        label = self.data['label'][idx]

        # 토큰화 및 인코딩
        input_encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 텐서 반환
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label),
        }

class CXRBertNLI(nn.Module) :
    def __init__(self, num_labels, pretrained=True, model=None, tokenizer=None) :
        super(CXRBertNLI, self).__init__()
        self.cxrbert = CXRBertModel.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG) if pretrained else model
        self.classifier = nn.Linear(self.cxrbert.config.projection_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor,
               labels: torch.Tensor,
               output_cls_projected_embedding: Optional[bool] = None) :
        outputs = self.cxrbert(input_ids, attention_mask, output_cls_projected_embedding=output_cls_projected_embedding)
        cls_projected_embedding = outputs[2]
        logits = self.classifier(cls_projected_embedding)
        loss = self.loss_fct(logits, labels)

        return loss, logits, outputs.hidden_states, outputs.attentions

def print_gpu_memory_usage():
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        allocated_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB 단위로 변환
        cached_memory = torch.cuda.memory_reserved(device_id) / (1024 ** 3)  # GB 단위로 변환
        print(f"GPU Memory: {(allocated_memory+cached_memory):.2f} GB")
    else:
        print("CUDA is not available.")
        
def train_evaluate(model, train_dataloader, test_dataloader, optimizer, num_epochs) :
    def evaluate(model, test_dataloader) :
        model.eval()
        with torch.no_grad() :
            running_loss = 0.0
        
            y_true = []
            y_pred = []
        
            for batch in test_dataloader:
                inputs = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                
                outputs = model(inputs, attention_mask, labels, True)
                
                loss = outputs[0]
                
                logits = outputs[1]
                pred = torch.argmax(logits, dim=1)
                
                y_true.append(labels.cpu().numpy())
                y_pred.append(pred.cpu().numpy())
        
                running_loss += loss.sum()
        
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
                
            total_loss = running_loss / len(test_dataloader)
            total_acc = accuracy_score(y_true, y_pred)
        return total_loss, total_acc

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    
    for epoch in tqdm(range(num_epochs)):
        print(epoch+1, "Epoch")
        
        running_loss = 0.0
        running_correct = 0.0

        model.train()
        for batch in train_dataloader:
            inputs = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
    
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask, labels, True)
            loss = outputs[0].sum()
            loss.backward()
            optimizer.step()
    
            logits = outputs[1]
            pred = torch.argmax(logits, dim=1)
    
            correct = (pred==labels).sum().item()
            running_correct += correct
            running_loss += loss.sum()
        
        print_gpu_memory_usage()
        
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_correct / len(train_dataloader.dataset)
        
        print(f"{epoch+1} Epoch Train Loss : {epoch_loss:.4f}, Train Acc : {epoch_acc:.4f}")
        
        test_loss, test_acc = evaluate(model, test_dataloader)
        print(f"{epoch+1} Epoch Test Loss : {test_loss:.4f}, Test Acc : {test_acc:.4f}")
            
        train_loss_list.append(epoch_loss.item())
        train_acc_list.append(epoch_acc)
        test_loss_list.append(test_loss.item())
        test_acc_list.append(test_acc)

    print(train_loss_list, train_acc_list, test_loss_list, test_acc_list)
    plt.figure()
    plt.title('Loss Graph')
    plt.plot(train_loss_list, label='Train')
    plt.plot(test_loss_list, label='Test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.title('Acc Graph')
    plt.plot(train_acc_list, label='Train')
    plt.plot(test_acc_list, label='Test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    print(train_loss_list, train_acc_list, test_loss_list, test_acc_list)
    plt.figure()
    plt.title('Loss Graph')
    plt.plot(train_loss_list, label='Train')
    plt.plot(test_loss_list, label='Test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f'Loss_lr_{lr}.png')

    plt.figure()
    plt.title('Acc Graph')
    plt.plot(train_acc_list, label='Train')
    plt.plot(test_acc_list, label='Test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f'Acc_lr_{lr}.png')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('--epoch', type=int, default=100, help="Epoch")

args = parser.parse_args()
lr = args.lr
epochs = args.epoch

print("Check Hyperparameters.")
print(f"Learning rate : {lr}")
print(f"Epochs : {epochs}")

torch.manual_seed(88)
torch.cuda.manual_seed_all(88)
np.random.seed(88)
random.seed(88)

if __name__ == "__main__" :
    print("Evaluating NLI Task Using CXRBert with LoRA Fine-tuning")
    
    print("1. Load Datasets")
    Train_dataset = load_from_disk('./../MedNLI')
    Test_dataset = load_from_disk('./../RadNLI')
    
    max_length = 512
    tokenizer = CXRBertTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
    
    # 학습 데이터셋 및 데이터로더 생성
    train_dataset = NliDataset(Train_dataset, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=80, shuffle=True)
    test_dataset = NliDataset(Test_dataset, tokenizer, max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("2. Load Model and Use LoRA")
    model = CXRBertNLI(3).cuda()

    print("Check Learnable Parameters")
    
    for n, p in model.named_parameters() :
        if p.requires_grad :
            print(n)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters : {total_params}, Trainable Parameters : {trainable_params}")

    print("3. Pallelel GPU")
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("4. Train and Evaluate")
    train_evaluate(model, train_dataloader, test_dataloader, optimizer, epochs)
