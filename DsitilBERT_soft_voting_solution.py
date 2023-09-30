import numpy as np
import pandas as pd
import collections
import gc
from tqdm import tqdm
import cv2
import cudf, cuml, cupy
from cuml.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import transformers
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig


import sys
sys.path.append('../input/timm-shpee/pytorch-image-models-master')
import timm
from timm.models.layers import SelectAdaptivePool2d

import warnings
warnings.filterwarnings('ignore')
COMPUTE_CV = False
cos_threshs = np.array([0.16])
test_batch_size = 128

# image model
GEM_P = 4
image_size = 420

# TTA for image(do not use)
flip_TTAs = [False, False, False, False, False]
testing_scales = [[1.0], [1.0], [1.0], [1.0], [1.0]]

# text model
text_max_length = 84

# alpha query expansion
alpha_query_expansion = True
qe_mid_knn = True
qe_ms     = [[1, 1], [1, 1], [1, 1], [2, 1], [1, 1]]
qe_alphas = [[2, 5], [2, 7], [5, 2], [7, 2], [3, 3]]

# adaptive thresholding
USE_ADAPTIVE_THRESHOLDING = False
CONSERVATIVENESS = 1.0
BETA = np.mean([0.9, 0.8, 0.9, 0.75, 0.3])

# min num preds
force_2preds = True
force_2preds_relax = 1.2

# kNN
KNN = 52
ALPHA_QE_KNN = 8
knn_metric = 'cosine' # cosine or correlation
model_weight_paths = [
    '../input/joint-f0gem3420768-dbert-highway/Joint_F0GeM3-420-768Emb-1024XBM_DBERT-Aug0.1_XBM4096_fold0_HighwayConc768x3_ep-002_f1-0.88579_thresh-0.43448_bs-30_emb-2304.pth',
    '../input/joint-f0gem3420768-dbert-highway/Joint_F0GeM3-420-768Emb-1024XBM_DBERT-Aug0.1_XBM4096_fold1_HighwayConc768x3_ep-002_f1-0.89122_thresh-0.42069_bs-32_emb-2304.pth',
    '../input/joint-f0gem3420768-dbert-highway/Joint_F0GeM3-420-768Emb-1024XBM_DBERT-Aug0.1_XBM4096_fold2_HighwayConc768x3_ep-003_f1-0.88911_thresh-0.44828_bs-30_emb-2304.pth',
    '../input/joint-f0gem3420768-dbert-highway/Joint_F0GeM3-420-768Emb-1024XBM_DBERT-Aug0.1_XBM4096_fold3_HighwayConc768x3_ep-002_f1-0.88196_thresh-0.44828_bs-32_emb-2304.pth',
    '../input/joint-f0gem3420768-dbert-highway/Joint_F0GeM3-420-768Emb-1024XBM_DBERT-Aug0.1_XBM4096_fold4_HighwayConc768x3_ep-003_f1-0.88931_thresh-0.44828_bs-32_emb-2304.pth'
                     ]
tokenizer = DistilBertTokenizer.from_pretrained('../input/distilberttextaugmadgrad5folds/distil-bert-textaug-madgrad-5folds/tokenizer')


# general
loader_num_workers = 2
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
load data
if COMPUTE_CV:
    test = pd.read_csv('../input/shopee-product-matching/train.csv').iloc[0:300]
else:
    test = pd.read_csv('../input/shopee-product-matching/test.csv')
test = test.drop(columns='image_phash')

LEN_TEST = len(test)

BASE = '../input/shopee-product-matching/test_images/'
if COMPUTE_CV:
    BASE = '../input/shopee-product-matching/train_images/'
    
CHUNK = 1024*4
CTS = LEN_TEST//CHUNK
if LEN_TEST%CHUNK!=0:
    CTS += 1
    
if LEN_TEST==3:
    KNN = 3
    ALPHA_QE_KNN = 3
    qe_ms     = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    qe_alphas = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]

# tokenize titles
texts = list(test['title'].apply(lambda o: str(o)).values)
text_encodings = tokenizer(texts, 
                           padding=True, 
                           truncation=True, 
                           max_length=text_max_length)

test['input_ids'] = text_encodings['input_ids']
test['attention_mask'] = text_encodings['attention_mask']

del texts, text_encodings, tokenizer
_=gc.collect()
class Shopee(Dataset):
    def __init__(self, df, image_dir, augs):
        self.df = df
        self.augs = augs 
        self.image_dir = image_dir


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {'input_ids': torch.tensor(self.df['input_ids'].iloc[idx]), 'attention_mask': torch.tensor(self.df['attention_mask'].iloc[idx])}
        
        # image
        image = cv2.imread(self.image_dir + self.df.loc[idx, 'image']).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.augs(image=image)['image']

        return image, item
    
def make_aug(scale=1.0, horizontal_flip=False):
    im_size = int(round(scale*image_size))
    if horizontal_flip:
        valid_aug = A.Compose([A.LongestMaxSize(max_size=im_size, p=1.0),
                               A.PadIfNeeded(min_height=im_size, min_width=im_size, border_mode=0, p=1.0),
                               A.HorizontalFlip(p=1.0),
                               A.Normalize(p=1.0),
                               ToTensorV2(p=1.0)])
        
    else:
        valid_aug = A.Compose([A.LongestMaxSize(max_size=im_size, p=1.0),
                               A.PadIfNeeded(min_height=im_size, min_width=im_size, border_mode=0, p=1.0),
                               A.Normalize(p=1.0),
                               ToTensorV2(p=1.0)])
        
    return valid_aug

# J3 without joint embeddings(stacked embeddings)
# joint embeddings give no score(CV and LB) boost, image/text concat embeddings are sufficient
class AdaptiveGeneralizedMeanPool2d(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(AdaptiveGeneralizedMeanPool2d, self).__init__()
        self.p = p
        self.eps = eps
        self.flatten1 = nn.Flatten()

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        x = F.adaptive_avg_pool2d(input=x.clamp(min=eps).pow(p), output_size=(1, 1)).pow(1./p)
        x = self.flatten1(x)
        return x

class NFNetF0_GeM_L2(nn.Module):
    def __init__(self, num_embeddings, pretrained=True):
        super(NFNetF0_GeM_L2, self).__init__()
        self.model = timm.create_model('dm_nfnet_f0', pretrained=pretrained)
        self.model.head.global_pool = AdaptiveGeneralizedMeanPool2d(p=GEM_P)
        num_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(num_features, num_embeddings)

    def forward(self, x):
        x = self.model(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        
        return x

class DistilBERT_L2(nn.Module):
    def __init__(self, bert_model):
        super(DistilBERT_L2, self).__init__()
        self.bert_model = bert_model
    
    def forward(self, batch):
        output = self.bert_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_hidden_state = output.last_hidden_state
        CLS_token_state = last_hidden_state[:, 0, :]
        CLS_token_state = F.normalize(CLS_token_state, p=2, dim=1, eps=1e-12)
        
        return CLS_token_state

class JointModel(nn.Module):
    def __init__(self, text_model, image_model):
        super(JointModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        
        # define fc1(embeddings stacking), this won't be used for forward
        self.fc1 = nn.Linear(768 + 768, 768)

    
    def forward(self, image_input, batch):
        # image embeddings
        image_emb = self.image_model(image_input)

        # CLS_token as text embeddings
        text_emb = self.text_model(batch)
        
        x = torch.cat((image_emb, text_emb), dim=1)

        return x
      
image_model = NFNetF0_GeM_L2(768, False)
bert_config = DistilBertConfig(activation='gelu',
                               attention_dropout=0.1,
                               dim=768,
                               dropout=0.1,
                               hidden_dim=3072,
                               initializer_range=0.02,
                               max_position_embeddings=512,
                               model_type='distilbert',
                               n_heads=12,
                               n_layers=6,
                               output_hidden_states=True,
                               pad_token_id=0,
                               qa_dropout=0.1,
                               seq_classif_dropout=0.2,
                               sinusoidal_pos_embds=True,
                               tie_weights_=True,
                               vocab_size=32000)

bert_model = DistilBertModel(bert_config)
text_model = DistilBERT_L2(bert_model)
model = JointModel(text_model, image_model)
del bert_model, text_model, image_model

_=model.to(device)
_=model.eval()

def joint_embedder(df, model, scale=1.0, flip=False):
    embeds = []
    CHUNK = 1024*4
    for i,j in enumerate(range(CTS)):
        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b,len(df))
        
        test_data = Shopee(df.iloc[a:b].reset_index(drop=True),
                           BASE,
                           augs=make_aug(scale=scale, horizontal_flip=flip))
        test_loader = DataLoader(test_data,
                                 shuffle=False,
                                 num_workers=loader_num_workers,
                                 pin_memory=False,# False:faster
                                 batch_size=test_batch_size)
        with torch.no_grad():
            for inputs, batch in tqdm(test_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                inputs = inputs.to(device)
                embedding = model(inputs, batch).detach().cpu().numpy()
                embeds.append(embedding)
        
    return np.concatenate(embeds)


def distance_to_matching_probability(x):
    # fit distance to matching probability by poly-lines
    p1 = [0.2, 0.850*CONSERVATIVENESS]
    p2 = [0.3, 0.600*CONSERVATIVENESS]
    p3 = [0.4, 0.225*CONSERVATIVENESS]
    p4 = [0.5, 0.050*CONSERVATIVENESS]
    if x < 0.0:
        y = 1.0
    elif x < p1[0]:
        y = x*(p1[1] - 1.0)/(p1[0] - 0.0) + 1.0
    elif x < p2[0]:
        y = (x - p2[0])*(p2[1] - p1[1])/(p2[0] - p1[0]) + p2[1]
    elif x < p3[0]:
        y = (x - p3[0])*(p3[1] - p2[1])/(p3[0] - p2[0]) + p3[1]
    elif x < p4[0]:
        y = (x - p4[0])*(p4[1] - p3[1])/(p4[0] - p3[0]) + p4[1]
    elif x < 0.6:
        y = (x - 0.6)*(0.0 - p4[1])/(0.6 - p4[0])
    else:
        y = 0
    return y

def adaptive_thresholding(dists=None, global_thresh=None, beta=None):
    probs = np.frompyfunc(distance_to_matching_probability, 1, 1)(dists)
    
    # expected number of positives
    ex_num_pos = np.sum(probs > 0.5)
    
    # expected F1 change when one more prediction is added
    # sign of F1 change matters
    for num_pred in range(0, KNN):
        #denom = (num_pred + 1 + ex_num_pos)*(num_pred + ex_num_pos)
        #term1 = 2.0/denom
        term2 = (num_pred + ex_num_pos)*probs[num_pred] - np.sum(probs[:num_pred])
        #dF = term1*term2
        dF = term2
        
        if dF < 0:
            break
    
    best_thresh = dists[num_pred]*1.00001
    best_thresh = 0.5*(best_thresh-0.3)+0.1875
    adaptive_thresh = beta*best_thresh + (1.0 - beta)*global_thresh
    
    #print(f'{global_thresh}-->{adaptive_thresh}')
    
    return adaptive_thresh


def knn_matching(knn_model, embeddings, thresh):
    preds = []
    CHUNK = 1024*4
    
    CTS = len(embeddings)//CHUNK
    if len(embeddings)%CHUNK!=0: CTS += 1
        
    for j in range(CTS):
        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b,len(embeddings))
        distances, indices = knn_model.kneighbors(embeddings[a:b,])
              
        for k in range(b-a):
            dists = distances[k,]
            if USE_ADAPTIVE_THRESHOLDING:
                adaptive_thresh = adaptive_thresholding(dists=dists, global_thresh=thresh, beta=BETA)
            else:
                adaptive_thresh = thresh
                
            IDX = np.where(dists < adaptive_thresh)[0]
            IDS = indices[k,IDX]
            
            # force min_num_preds to be 2
            if force_2preds:
                if len(IDS) < 2:
                    # relax matching threshold
                    IDX = np.where(dists < thresh*force_2preds_relax)[0]
                    IDS = indices[k,IDX]
                        
            o = test.iloc[IDS].posting_id.values
            preds.append(o)
            
    return preds


def alpha_query_expansion(knn_model, embeddings, qe_alpha, qe_m):
    expanded_embeddings = []
    CHUNK = 1024*4
    
    CTS = len(embeddings)//CHUNK
    if len(embeddings)%CHUNK!=0: CTS += 1
    for j in range( CTS ):
        
        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b,len(embeddings))
        distances, indices = knn_model.kneighbors(embeddings[a:b,])
        for i in range(b-a):
            weights = ((1-distances[i, 0:qe_m+1])**qe_alpha).reshape(qe_m+1, 1)
            expanded_embedding = np.sum((embeddings[indices[i, 0:qe_m+1]]*weights), axis=0)/np.sum(weights)
            expanded_embedding = expanded_embedding / np.linalg.norm(expanded_embedding)
            expanded_embeddings.append(expanded_embedding)
            
    return np.array(expanded_embeddings)
for i_model, model_weight_path in enumerate(model_weight_paths):
    model.load_state_dict(torch.load(model_weight_path))
    
    # compute embeddings
    n_test = 0
    for testing_scale in testing_scales[i_model]:
        if n_test ==0:
            joint_embeddings = joint_embedder(test, model, scale=testing_scale, flip=False)
        else:
            joint_embeddings += joint_embedder(test, model, scale=testing_scale, flip=False)
        n_test += 1
    
        if flip_TTAs[i_model]:
            joint_embeddings += joint_embedder(test, model, scale=testing_scale, flip=True)
            n_test += 1
            
    joint_embeddings = joint_embeddings/n_test
    

    # alpha query expansion
    if alpha_query_expansion:
        knn_model = NearestNeighbors(n_neighbors=ALPHA_QE_KNN, metric=knn_metric)
        knn_model.fit(joint_embeddings)
        for qe_alpha, qe_m in zip(qe_alphas[i_model], qe_ms[i_model]):
            joint_embeddings = alpha_query_expansion(knn_model, joint_embeddings, qe_alpha, qe_m)
            
            if qe_mid_knn:
                knn_model = NearestNeighbors(n_neighbors=ALPHA_QE_KNN, metric=knn_metric)
                knn_model.fit(joint_embeddings)
        del knn_model
        _=gc.collect()
    
    # concat fold embeddings
    if i_model == 0:
        long_embeddings = joint_embeddings
    else:
        long_embeddings = np.hstack([long_embeddings, joint_embeddings])
        
print(long_embeddings.shape)

del joint_embeddings
_=gc.collect()


# prediction
knn_model = NearestNeighbors(n_neighbors=KNN, metric=knn_metric)
knn_model.fit(long_embeddings)
test['matches'] = knn_matching(knn_model, long_embeddings, np.mean(cos_threshs))

del long_embeddings, knn_model, model
_=gc.collect()

test['matches'] = test['matches'].map(lambda x : ' '.join(np.array(x)[0:KNN]))
test[['posting_id','matches']].to_csv('submission.csv',index=False)
sub = pd.read_csv('submission.csv')
sub.head()
 
# https://www.kaggle.com/code/a2015003713/shopee-nfnetf0-dsitilbert-soft-voting
