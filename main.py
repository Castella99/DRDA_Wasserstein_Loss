import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from wgan import FE, Discriminator, Classifier, Wasserstein_Loss, Grad_Loss
from tqdm import tqdm
import os
from scipy import io
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
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


#data 40 x 40 x 8064 video/trial x channel x data
#labels 40 x 4 video/trial x label (valence, arousal, dominance, liking)
#32명 -> 12 / 12 / 8

# data split
printsave("data split")
train_data, val_data,train_label, val_label = train_test_split(eeg_data32, VAL, test_size=0.25)
x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.5)

# make data loader
printsave("make data loader")
target_dataset = TensorDataset(x_train, y_train)
source_dataset = TensorDataset(x_test, y_test)
val_dataset = TensorDataset(val_data, val_label)
target_dataloader = DataLoader(target_dataset, 64, shuffle=True)
source_dataloader = DataLoader(source_dataset, 64, shuffle=True)
val_dataloader = DataLoader(val_dataset, 64, shuffle=True)

# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
printsave("device: ", device)

#model
dis = Discriminator(15960).to(device)
fe = FE(32).to(device)
classifier = Classifier().to(device)

#optim
optimizer_dis = optim.Adam(dis.parameters(),lr=0.0001,betas=(0,0.9))
optimizer_fe = optim.Adam(fe.parameters(),lr=0.0001, betas=(0,0.0))
optimizer_cls = optim.Adam(classifier.parameters(),lr=0.0001, betas=(0,0.9))

#cls_loss
criterion = nn.CrossEntropyLoss().to(device)

# train WGAN
accuracy_s = []
accuracy_t = []
accuracy_val = []
val_loss_list = []

best_loss = 10000000
limit_epoch = 100
limit_check = 0
val_loss = 0
nb_epochs = 1000
lambda_hyper = 10
mu_hyper = 10
n = 5

torch.autograd.set_detect_anomaly(True)

epochs = 0
printsave()
# while parameter converge
for epoch in range(nb_epochs):
    temp_accuracy_t = 0
    temp_accuracy_s = 0
    temp_accuracy_val = 0
    temp_gloss = 0
    temp_wdloss = 0
    temp_gradloss = 0
    temp_clsloss = 0

    printsave(epoch+1, ": epoch")

    temp = 0.0 #batch count
    fe.train()
    dis.train()
    classifier.train()
    # batch
    for i, (target, source) in enumerate(zip(target_dataloader, source_dataloader)):
        temp += 1.0

        x_target = target[0].to(device)
        y_target = target[1].to(device)
        x_source = source[0].to(device)
        y_source = source[1].to(device)

        # update discriminator
        for p in fe.parameters() :
            p.requires_grad = False
        for p in dis.parameters() :
            p.requires_grad = True
        for p in classifier.parameters() :
            p.requires_grad = False
        
        for k in range(n) :
            optimizer_dis.zero_grad()
            wd_grad_loss = 0
            feat_t = fe(x_target)
            feat_s = fe(x_source)
            pred_t = classifier(feat_t)
            pred_s = classifier(feat_s)
            for j in range(feat_s.size(0)) :
                epsil = torch.rand(1).item()
                feat = epsil*feat_s[j,:]+(1-epsil)*feat_t[j,:]
                dc_t = dis(feat_t)
                dc_s = dis(feat_s)
                wd_loss = Wasserstein_Loss(dc_s, dc_t)
                grad_loss = Grad_Loss(feat, dis, device)
                wd_grad_loss = wd_grad_loss - (wd_loss-lambda_hyper*grad_loss)
            wd_grad_loss = wd_grad_loss / feat_s.size(0)
            wd_grad_loss.backward()
            optimizer_dis.step()

        # update classifier
        for p in fe.parameters() :
            p.requires_grad = False
        for p in dis.parameters() :
            p.requires_grad = False
        for p in classifier.parameters() :
            p.requires_grad = True
        
        optimizer_cls.zero_grad()
        feat_s = fe(x_source)
        pred_s = classifier(feat_s)
        cls_loss_source = criterion(pred_s, y_source-1)
        cls_loss_source.backward()
        optimizer_cls.step()
        
        # update Feature Extractor
        for p in fe.parameters() :
            p.requires_grad = True
        for p in dis.parameters() :
            p.requires_grad = False
        for p in classifier.parameters() :
            p.requires_grad = False
        
        optimizer_fe.zero_grad()
        feat_t = fe(x_target)
        feat_s = fe(x_source)
        pred_s = classifier(feat_s)
        dc_t = dis(feat_t)
        dc_s = dis(feat_s)
        wd_loss = Wasserstein_Loss(dc_s, dc_t)
        cls_loss_source = criterion(pred_s, y_source-1)
        fe_loss = cls_loss_source + mu_hyper*wd_loss
        fe_loss.backward()
        optimizer_fe.step()
        
        # Temp_Loss
        wd_loss = Wasserstein_Loss(dc_s, dc_t)
        cls_loss_source = criterion(pred_s, y_source-1)
        g_loss = cls_loss_source + mu_hyper*(wd_loss - lambda_hyper*grad_loss)

        feat_t = fe(x_target)
        feat_s = fe(x_source)
        pred_t = classifier(feat_t)
        pred_s = classifier(feat_s)
        
        temp_wdloss = temp_wdloss + wd_loss
        temp_clsloss = temp_clsloss + cls_loss_source
        temp_gloss = temp_gloss + g_loss

        temp_accuracy_t += ((torch.argmax(pred_t,1)+1)== y_target).to(torch.float).mean()
        temp_accuracy_s += ((torch.argmax(pred_s,1)+1)== y_source).to(torch.float).mean()
    
    printsave("\ngloss", temp_gloss.item()/temp)
    printsave("wd_loss", temp_wdloss.item()/temp)
    printsave("cls_loss", temp_clsloss.item()/temp)
    printsave("acc_t", temp_accuracy_t.item()/temp)
    printsave("acc_s", temp_accuracy_s.item()/temp)
    
    accuracy_t.append(temp_accuracy_t/temp)
    accuracy_s.append(temp_accuracy_s/temp)
    
    fe.eval()
    dis.eval()
    classifier.eval()
    val_loss = 0
    temp = 0
    for x_val, y_val in val_dataloader:
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        pred_val = classifier(fe(x_val))
        temp_accuracy_val += ((torch.argmax(pred_val,1)+1)== y_val).to(torch.float).mean()
        loss = criterion(pred_val, y_val-1)
        val_loss += loss.item() * x_val.size(0)
        temp += 1
    val_total_loss = val_loss / len(val_dataloader.dataset)
    val_loss_list.append(val_total_loss)
    printsave("val_loss :", val_total_loss)
    printsave("acc_val :", temp_accuracy_val.item()/temp)
    accuracy_val.append(temp_accuracy_val.item()/temp)
    epochs = epochs + 1
    if val_total_loss > best_loss:
        limit_check += 1
        if(limit_check >= limit_epoch):
            break
    else:
        best_loss = val_total_loss
        limit_check = 0
    printsave()

printsave("\naccuracy_t ", sum(accuracy_t)/len(accuracy_t))
printsave("accuracy_s ", sum(accuracy_s)/len(accuracy_s))
printsave("accuracy_val", sum(accuracy_val)/len(accuracy_val))
printsave("best_val_loss ", best_loss)

plt.title('val_loss')
plt.plot(np.arange(1, epochs+1, 1), val_loss_list, label='val')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.savefig('save_val_loss.png')
