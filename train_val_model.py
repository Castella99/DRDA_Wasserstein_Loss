import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from wgan import FE, Discriminator, Classifier, Wasserstein_Loss, Grad_Loss
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import copy

def printsave(*a, end='\n') :
    print(*a, end=end)
    log = open("log.txt", "a")
    print(*a, file=log, end=end)
    log.close()

def make_dataloader(input, output) :
    print("data split")
    train_data, val_data,train_label, val_label = train_test_split(input, output, test_size=0.25)
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.5)

    # make data loader
    print("make data loader")
    target_dataset = TensorDataset(x_train, y_train)
    source_dataset = TensorDataset(x_test, y_test)
    val_dataset = TensorDataset(val_data, val_label)
    target_dataloader = DataLoader(target_dataset, 64, shuffle=True)
    source_dataloader = DataLoader(source_dataset, 64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, 64, shuffle=True)
    return source_dataloader, target_dataloader, val_dataloader

def train_val(source_dataloader, target_dataloader, val_dataloader, label, nb_epochs, hyper_lambda, hyper_mu, hyper_n, patience) :
    printsave(f"\n\n{label} label train and valiation")
    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    printsave("device: ", device)

    #model
    dis = Discriminator().to(device)
    fe = FE().to(device)
    classifier = Classifier().to(device)

    #optim
    optimizer_dis = optim.Adam(dis.parameters(),lr=0.0001,betas=(0,0.9))
    optimizer_fe = optim.Adam(fe.parameters(),lr=0.0001, betas=(0,0.0))
    optimizer_cls = optim.Adam(classifier.parameters(),lr=0.0001, betas=(0,0.9))

    #cls_loss
    cls_loss = nn.CrossEntropyLoss().to(device)

    # train WGAN
    accuracy_s = []
    accuracy_t = []
    accuracy_val = []
    val_loss_list = []

    best_loss = 10000000
    limit_check = 0
    val_loss = 0
    best_epoch = 1

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
            
            for k in range(hyper_n) :
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
                    wd_grad_loss = wd_grad_loss - (wd_loss-hyper_lambda*grad_loss)
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
            cls_loss_source = cls_loss(pred_s, y_source-1)
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
            cls_loss_source = cls_loss(pred_s, y_source-1)
            fe_loss = cls_loss_source + hyper_mu*wd_loss
            fe_loss.backward()
            optimizer_fe.step()
            
            # Temp_Loss
            wd_loss = Wasserstein_Loss(dc_s, dc_t)
            cls_loss_source = cls_loss(pred_s, y_source-1)
            g_loss = cls_loss_source + hyper_mu*(wd_loss - hyper_lambda*grad_loss)

            feat_t = fe(x_target)
            feat_s = fe(x_source)
            pred_t = classifier(feat_t)
            pred_s = classifier(feat_s)
            
            temp_wdloss = temp_wdloss + wd_loss
            temp_clsloss = temp_clsloss + cls_loss_source
            temp_gloss = temp_gloss + g_loss

            temp_accuracy_t += ((torch.argmax(pred_t,1)+1)== y_target).to(torch.float).mean()
            temp_accuracy_s += ((torch.argmax(pred_s,1)+1)== y_source).to(torch.float).mean()
        
        printsave("\ngloss :", temp_gloss.item()/temp)
        printsave("wd_loss :", temp_wdloss.item()/temp)
        printsave("cls_loss :", temp_clsloss.item()/temp)
        printsave("acc_t :", temp_accuracy_t.item()/temp)
        printsave("acc_s :", temp_accuracy_s.item()/temp)
        
        accuracy_t.append(temp_accuracy_t.item()/temp)
        accuracy_s.append(temp_accuracy_s.item()/temp)
        
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
            loss = cls_loss(pred_val, y_val-1)
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
            if(limit_check >= patience):
                break
        else:
            best_loss = val_total_loss
            best_fe_wts = copy.deepcopy(fe.state_dict())
            best_dis_wts = copy.deepcopy(dis.state_dict())
            best_cls_wts = copy.deepcopy(classifier.state_dict())
            best_epoch = epoch+1
            limit_check = 0
        printsave(f"best_val_loss : {best_loss}, epoch : {best_epoch}")
        printsave()

    printsave("\naccuracy_t :", sum(accuracy_t)/len(accuracy_t))
    printsave("accuracy_s :", sum(accuracy_s)/len(accuracy_s))
    printsave("accuracy_val :", sum(accuracy_val)/len(accuracy_val))
    printsave(f"best_val_loss : {best_loss}, epoch : {best_epoch}")

    plt.title('val_loss')
    plt.plot(np.arange(1, epochs+1, 1), val_loss_list, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(f'{label} val_loss.png')
    
    fe.load_state_dict(best_fe_wts)
    dis.load_state_dict(best_dis_wts)
    classifier.load_state_dict(best_cls_wts)

    return fe, dis, classifier