from health_multimodal.image.model.pretrained import (
    BIOMED_VLP_BIOVIL_T,
    BIOMED_VLP_CXR_BERT_SPECIALIZED,
    BIOVIL_T_COMMIT_TAG,
    CXR_BERT_COMMIT_TAG,
)
from health_multimodal.text.model.modelling_cxrbert import CXRBertModel
from health_multimodal.text.model.configuration_cxrbert import CXRBertConfig, CXRBertTokenizer 
from health_multimodal.image.model.pretrained import _download_biovil_image_model_weights
from health_multimodal.image.model.model import ImageModel
from health_multimodal.image import ImageEncoderType
import os
import pydicom
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.manifold import TSNE
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import pandas as pd
import random
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import argparse

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

# %%
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

torch.manual_seed(88)
torch.cuda.manual_seed_all(88)
np.random.seed(88)
random.seed(88)

if __name__ == "__main__" :
    print("Zero-shot Classification with Pretrained BioViL")
    resnet_checkpoint_path = _download_biovil_image_model_weights()
    JOINT_FEATURE_SIZE = 128

    image_model = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=resnet_checkpoint_path,
    )

    tokenizer = CXRBertTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
    text_model = CXRBertModel.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG).cuda()

    df = pd.read_csv('./../rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)

    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor(),
        ExpandChannels()
    ])

    image_path = './../rsna-pneumonia-detection-challenge/stage_2_train_images'
    test_set = PneumoniaImageDataset(image_path, test_df, transform=test_transform)

    test_loader = DataLoader(test_set, batch_size=128)

    prompt = ["No evidence of pneumonia", "Findings suggesting pneumonia"]

    tokens = tokenizer.batch_encode_plus(batch_text_or_text_pairs=prompt, add_special_tokens=True, padding="longest", return_tensors='pt')
    tokens.input_ids = tokens.input_ids.cuda()
    tokens.attention_mask = tokens.attention_mask.cuda()

    prompt_emb = text_model.get_projected_text_embeddings(tokens.input_ids,
                                            tokens.attention_mask,
                                            True)

    prompt_emb_pos = prompt_emb[1]
    prompt_emb_neg = prompt_emb[0]

    all_preds = []
    all_labels = []
    all_emb = []

    image_model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            img, labels = batch
            
            img = img.cuda()
            image_emb = image_model(img).projected_global_embedding
            image_emb = F.normalize(image_emb, dim=-1)

            pred_pos = image_emb @ prompt_emb_pos.t()
            pred_neg = image_emb @ prompt_emb_neg.t()

            preds = (pred_pos >= pred_neg).int()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_emb.extend(image_emb)

    # 평가 지표 계산 (여기서는 정확도를 사용)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy}")
    print(f"ROAUC: {auc}")
    print(f"F1 Score: {f1}")

    print(confusion_matrix(all_labels, all_preds))