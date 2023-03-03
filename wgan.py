import torch 
import torch.nn as nn

class FE(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,30,kernel_size=(1,25)),
            nn.Conv2d(30,30, kernel_size=(channel_size, 1)),
            nn.AvgPool2d(kernel_size=(1, 75), stride=15),
            nn.Flatten()
        )
        
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module) :
    def __init__(self, input_size=15960, cls_num=9) :
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, cls_num)
        )

    def forward(self, x) :
        return self.fc(x)

def Wasserstein_Loss(dc_s, dc_t) :
    return torch.mean(dc_s) - torch.mean(dc_t)

def Grad_Loss(feat, dis, device) :
    feat_ = feat.clone().detach().to(device).requires_grad_(True)
    output = dis(feat_)
    grad = torch.autograd.grad(output, feat_, torch.ones(1).to(device))[0]
    return torch.square(grad.norm()-1)

if __name__ == "__main__":
    print(__name__)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    input_s = torch.randn(64, 1, 32, 8064).to(device)
    input_t = torch.randn(64, 1, 32, 8064).to(device)
    dis = Discriminator(15960).to(device)
    fe = FE(32).to(device)
    classifier = Classifier().to(device)
    feat_s = fe(input_s)
    feat_t = fe(input_t)
    pred_s = classifier(feat_s)
    pred_t = classifier(feat_t)
    feat = torch.cat((feat_s,feat_t), dim=0)
    print("loss", Grad_Loss(feat, dis, device))

    print("-"*50)
    test_a = torch.randn(128, 15960).to(device)
    print(torch.norm(test_a, dim=1).shape) # norm에 대한 평균
    print(torch.mean(test_a, dim=0).shape) # 평균값에 대한 norm
    print(torch.norm(test_a, dim=1).mean().item())
    print(torch.mean(test_a, dim=0).norm().item())