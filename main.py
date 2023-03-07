import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from wgan import FE, Discriminator, Classifier, Wasserstein_Loss, Grad_Loss
from tqdm import tqdm
import os
from scipy import io
import numpy as np
from train_val_model import printsave, train_val, make_dataloader

import os
os.getcwd()

log = open("log.txt", "w")
print("DRDA_Wasserstein_Loss\n", file=log)
log.close()

# path
path = r'../data_preprocessed_matlab/'  # 경로는 저장 파일 경로
file_list = os.listdir(path)

printsave("data path check")
for i in file_list:    # 확인
    printsave(i, end=' ')


for i in tqdm(file_list, desc="read data"): 
    mat_file = io.loadmat(path+i)
    data = mat_file['data']
    labels = np.array(mat_file['labels'])
    val = labels.T[0].round().astype(np.int8)
    aro = labels.T[1].round().astype(np.int8)
    
    if(i=="s05.mat"): 
        Data = data
        VAL = val
        ARO = aro
        continue
        
    Data = np.concatenate((Data ,data),axis=0)   # 밑으로 쌓아서 하나로 만듬
    VAL = np.concatenate((VAL ,val),axis=0)
    ARO = np.concatenate((ARO ,aro),axis=0)

# eeg preprocessing

eeg_data = []
peripheral_data = []

for i in tqdm(range(len(Data)), desc="preprocess channel"):
    for j in range (40): 
        if(j < 32): # get channels 1 to 32
            eeg_data.append(Data[i][j])
        else:
            peripheral_data.append(Data[i][j])

# set data type, shape
eeg_data = np.reshape(eeg_data, (len(Data),1,32, 8064))
eeg_data = eeg_data.astype('float32')
eeg_data32 = torch.from_numpy(eeg_data)
VAL = (torch.from_numpy(VAL)).type(torch.long)
ARO = (torch.from_numpy(ARO)).type(torch.long)

source_VAL, target_VAL, val_VAL = make_dataloader(eeg_data32, VAL)
source_ARO, target_ARO, val_ARO = make_dataloader(eeg_data32, ARO)

fe_VAL, dis_VAL, cls_VAL = train_val(source_VAL, target_VAL, val_VAL, 'VALENCE', 1000, 10, 10, 10, 100)

fe_ARO, dis_ARO, cls_ARO = train_val(source_ARO, target_ARO, val_ARO, 'AROUSAL', 1000, 10, 10, 10, 100)
