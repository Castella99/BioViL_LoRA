from health_multimodal.image.model.pretrained import (
    BIOMED_VLP_BIOVIL_T,
    BIOMED_VLP_CXR_BERT_SPECIALIZED,
    BIOVIL_T_COMMIT_TAG,
    CXR_BERT_COMMIT_TAG,
)
from health_multimodal.text.model.modelling_cxrbert import CXRBertModel
from health_multimodal.text.model.configuration_cxrbert import CXRBertConfig, CXRBertTokenizer 
from health_multimodal.image.model.pretrained import _download_biovil_t_image_model_weights
from health_multimodal.image.model.pretrained import _download_biovil_image_model_weights
from health_multimodal.image.model.model import ImageModel
from health_multimodal.image import ImageEncoderType
import os
import pydicom
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import transformers
from torchvision import transforms
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np
from loralib import LoRALayer
import loralib as lora
import math
import pandas as pd
import random
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import argparse

class LoRAConv(nn.Module, LoRALayer):
    def __init__(self, conv_layer, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(LoRAConv, self).__init__()
        self.conv = conv_layer
        kernel_size = self.conv.kernel_size[0]
        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            #print((r * kernel_size, in_channels * kernel_size))
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        #self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(LoRAConv, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class LoRAConv2d(LoRAConv):
    def __init__(self, *args, **kwargs):
        super(LoRAConv2d, self).__init__(*args, **kwargs)

def replace_conv_with_lora(model, rank):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # LoRAConv2d 인스턴스 생성
            lora_module = LoRAConv2d(module, r=rank)
            # 현재 모듈을 LoRA 모듈로 대체
            setattr(model, name, lora_module)
        else:
            replace_conv_with_lora(module, rank)  # 재귀적으로 하위 모듈 탐색

class PneumoniaImageDataset(Dataset) :
    def __init__(self, root, df, transform=None, few_shot=None):
        super(PneumoniaImageDataset, self).__init__()
        self.root = root
        self.df = df
        self.transform = transform

        # sample data
        if few_shot :
            self.df = self.df.sample(frac=few_shot)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = row["patientId"]
        img_path = os.path.join(self.root, img_path+'.dcm')
        x = self.read_from_dicom(img_path)
        y = float(row["Target"])
        y = torch.tensor([y]).long().reshape(-1)
        return x, y

    def __len__(self):
        return len(self.df)

    def read_from_dicom(self, img_path):
        image = pydicom.dcmread(img_path).pixel_array
        def remap_to_uint8(array: np.ndarray, percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
            array = array.astype(float)
            if percentiles is not None:
                len_percentiles = len(percentiles)
                if len_percentiles != 2:
                    message = 'The value for percentiles should be a sequence of length 2,' f' but has length {len_percentiles}'
                    raise ValueError(message)
                a, b = percentiles
                if a >= b:
                    raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
                if a < 0 or b > 100:
                    raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
                cutoff: np.ndarray = np.percentile(array, percentiles)
                array = np.clip(array, *cutoff)
            array -= array.min()
            array /= array.max()
            array *= 255
            return array.astype(np.uint8)
            
        image = remap_to_uint8(image)
        image = Image.fromarray(image).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image

class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)
    
class VLP_RSNA_Model(nn.Module) :
    def __init__(self, image_model, text_emb) :
        super(VLP_RSNA_Model, self).__init__()
        self.image_model = image_model
        self.classifier = nn.Linear(128,2)
        self.classifier.weight = nn.Parameter(text_emb)
        self.classifier.bias = nn.Parameter(torch.zeros(2))
    def forward(self, img) :
        emb = self.image_model(img).projected_global_embedding
        logit = self.classifier(emb)
        return logit
    
def evaluate(model, test_dataloader) :
    criterion = nn.CrossEntropyLoss().cuda()
    model.eval()
    with torch.no_grad() :
        running_loss = 0.0
    
        y_true = []
        y_pred = []
    
        for batch in test_dataloader:
            img = batch[0].cuda()
            labels = batch[1].reshape(-1).cuda()
            
            outputs = model(img)
            
            loss = criterion(outputs, labels)
            
            pred = torch.argmax(outputs, dim=1)
            
            y_true.append(labels.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    
            running_loss += loss.sum()
    
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
            
        total_loss = running_loss / len(test_dataloader)
        total_acc = accuracy_score(y_true, y_pred)
        total_auc = roc_auc_score(y_true, y_pred)
        total_f1 = f1_score(y_true, y_pred)
        
        print(confusion_matrix(y_true, y_pred))
        
    return total_loss, total_acc, total_auc, total_f1

def print_gpu_memory_usage():
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        allocated_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB 단위로 변환
        cached_memory = torch.cuda.memory_reserved(device_id) / (1024 ** 3)  # GB 단위로 변환
        print(f"GPU Memory: {(allocated_memory+cached_memory):.2f} GB")
    else:
        print("CUDA is not available.")
        
def train_evaluate(model, train_loader, test_loader, optimizer, num_epochs) :
    criterion = nn.CrossEntropyLoss().cuda()

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in tqdm(range(num_epochs)):
        print(epoch+1, "Epoch")
        
        running_loss = 0.0
        running_correct = 0.0
        model.train()
        for batch in train_loader:
            img = batch[0].cuda()
            labels = batch[1].reshape(-1).cuda()
        
            optimizer.zero_grad()
            
            outputs = model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            
            correct = (preds==labels).sum().item()

            running_loss += loss
            running_correct += correct
        
        print_gpu_memory_usage()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_correct / len(train_loader.dataset)

        print(f"{epoch+1} Epoch Train Loss : {epoch_loss:.4f}, Train Acc : {epoch_acc:.4f}")

        test_loss, test_acc, test_auc, test_f1 = evaluate(model, test_loader)
        print(f"{epoch+1} Epoch Test Loss : {test_loss:.4f}, Test Acc : {test_acc:.4f}, Test AUC : {test_auc:.4f}, Test F1 : {test_f1:.4f}")

        train_loss_list.append(epoch_loss.item())
        train_acc_list.append(epoch_acc)
        test_loss_list.append(test_loss.item())
        test_acc_list.append(test_acc)

    plt.figure()
    plt.title('Loss Graph')
    plt.plot(train_loss_list, label='Train')
    plt.plot(test_loss_list, label='Test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f'Loss_ratio_{few_shot_ratio}_rank_{rank}_LoRA.png')

    plt.figure()
    plt.title('Acc Graph')
    plt.plot(train_acc_list, label='Train')
    plt.plot(test_acc_list, label='Test')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f'Acc_ratio_{few_shot_ratio}_rank_{rank}_LoRA.png')

torch.manual_seed(88)
torch.cuda.manual_seed_all(88)
np.random.seed(88)
random.seed(88)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help="Epoch")
parser.add_argument('--ratio', type=float, default=1.0, help="Ratio of Labels")
parser.add_argument('--rank', type=int, default=4, help="LoRA Rank")

args = parser.parse_args()

epochs = args.epoch
few_shot_ratio = args.ratio
rank = args.rank

print("Check Hyperparameters.")
print(f"Epochs : {epochs}")
print(f"Ratio of Labels for Few-shot Learning : {few_shot_ratio}")
print(f"LoRA Rank : {rank}")

if __name__ == "__main__" :
    print("1. Dataset Load")
    df = pd.read_csv('./../rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)

    train_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop((512, 512)),
        transforms.RandomAffine(degrees=45, shear=25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        ExpandChannels()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor(),
        ExpandChannels()
    ])

    image_path = './../rsna-pneumonia-detection-challenge/stage_2_train_images'
    # %%
    train_set = PneumoniaImageDataset(image_path, train_df, transform=train_transform)
    train_1_set = PneumoniaImageDataset(image_path, train_df, transform=train_transform, few_shot=0.01)
    train_10_set = PneumoniaImageDataset(image_path, train_df, transform=train_transform, few_shot=0.1)
    test_set = PneumoniaImageDataset(image_path, test_df, transform=test_transform)

    # %%
    train_loader = DataLoader(train_set, batch_size=64)
    train_1_loader = DataLoader(train_1_set, batch_size=64)
    train_10_loader = DataLoader(train_10_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=1)
    
    print("2. Model Load")
    resnet_checkpoint_path = _download_biovil_image_model_weights()
    JOINT_FEATURE_SIZE = 128
    
    image_model = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=resnet_checkpoint_path,
    )

    tokenizer = CXRBertTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
    text_model = CXRBertModel.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG).cuda()

    prompt = ["No evidence of pneumonia", "Findings suggesting pneumonia"]
    tokens = tokenizer.batch_encode_plus(batch_text_or_text_pairs=prompt, add_special_tokens=True, padding="longest", return_tensors='pt')
    tokens.input_ids = tokens.input_ids.cuda()
    tokens.attention_mask = tokens.attention_mask.cuda()
    prompt_emb = text_model.get_projected_text_embeddings(tokens.input_ids,
                                            tokens.attention_mask,
                                            True)

    
    print("3. Apply LoRA")
    replace_conv_with_lora(image_model, rank)

    lora.lora_state_dict(image_model)

    lora.mark_only_lora_as_trainable(image_model, bias='lora_only')

    model = VLP_RSNA_Model(image_model, prompt_emb).cuda()

    for n, p in model.named_parameters() :
            if p.requires_grad :
                print(n)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters : {total_params}, Trainable Parameters : {trainable_params}")

    train_dataloader =  train_1_loader if few_shot_ratio == 0.01 else train_10_loader if few_shot_ratio == 0.1 else train_loader
    
    print(f"Size of Train dataset : {len(train_dataloader.dataset)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    print("4. Train and Evaluate")
    train_evaluate(model, train_dataloader, test_loader, optimizer, epochs)