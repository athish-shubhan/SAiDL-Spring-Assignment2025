import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        return self.classifier(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss  
        return focal_loss.mean()


class NormalizedCrossEntropy(nn.Module):
    def forward(self,x,y):
        log_probs=F.log_softmax(x,-1)
        num=-log_probs[range(len(y)),y]
        denom=-log_probs.sum(dim=-1)
        return (num/denom).mean()

class NormalizedFocalLoss(nn.Module):
    def __init__(self,gamma=2):
        super().__init__()
        self.gamma=gamma

    def forward(self,x,y):
        probs=F.softmax(x,-1)
        pt=probs[range(len(y)),y]
        num=((1-pt)**self.gamma)*(-torch.log(pt+1e-10))
        denom=((1-probs)**self.gamma*(-torch.log(probs+1e-10))).sum(dim=-1)
        return (num/denom).mean()

class MeanAbsoluteError(nn.Module):
    def forward(self,x,y):
        probs=F.softmax(x,-1)
        y_onehot=F.one_hot(y,num_classes=x.size(-1)).float()
        return F.l1_loss(probs,y_onehot)

class ReverseCrossEntropy(nn.Module):
    def __init__(self,A=-4):
        super().__init__()
        self.A=A

    def forward(self,x,y):
        probs=F.softmax(x,-1)
        pt=probs[range(len(y)),y]
        return -self.A*(1-pt).mean()

class ActivePassiveLoss(nn.Module):
    def __init__(self,a,p,a_wt=1,p_wt=1):
        super().__init__()
        self.a,self.p,self.a_wt,self.p_wt=a,p,a_wt,p_wt

    def forward(self,x,y):
        return self.a_wt*self.a(x,y)+self.p_wt*self.p(x,y)

def create_symmetric_noise(labels, rate):
    noisy_labels = labels.copy()
    indices = np.random.permutation(len(labels))
    noise_indices = indices[:int(len(labels) * rate)]
    
    for idx in noise_indices:
        noisy_label = np.random.randint(0, 10)
        while noisy_label == labels[idx]:
            noisy_label = np.random.randint(0, 10)
        noisy_labels[idx] = noisy_label
    
    return noisy_labels


def calculate_accuracy(loader,m,dvc):
    m.eval()
    correct,total=0,0
    
    with torch.no_grad():
      for x,y in loader:
          x,y=x.to(dvc),y.to(dvc)
          pred=m(x).argmax(dim=-1)
          correct+=(pred==y).sum().item()
          total+=y.size(0)
    
    return correct/total*100

def plot_accuracy(train_acc_history, test_acc_history, title, save_path):
    plt.figure(figsize=(8,4))
    plt.plot(train_acc_history,label="Train")
    plt.plot(test_acc_history,label="Test")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    data_root = './data'
    os.makedirs(data_root, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    sym_noise_rates = [0.2, 0.4, 0.6, 0.8]
    asym_noise_rates = [0.1, 0.2, 0.3, 0.4]
    
    full_train_data = CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_data = CIFAR10(root=data_root, train=False, download=True, transform=transform)
    
    loss_functions = {
        "CE": nn.CrossEntropyLoss(),
        "FL": FocalLoss(gamma=2),
        "NCE": NormalizedCrossEntropy(),
        "NFL": NormalizedFocalLoss(),
        "MAE": MeanAbsoluteError(),
        "RCE": ReverseCrossEntropy(),
        "APL-NCE+MAE": ActivePassiveLoss(NormalizedCrossEntropy(), MeanAbsoluteError(), a_wt=1.0, p_wt=1.0),
        "APL-NCE+RCE": ActivePassiveLoss(NormalizedCrossEntropy(), ReverseCrossEntropy(), a_wt=1.0, p_wt=1.0),
        "APL-NFL+MAE": ActivePassiveLoss(NormalizedFocalLoss(gamma=2), MeanAbsoluteError(), a_wt=1.0, p_wt=1.0),
        "APL-NFL+RCE": ActivePassiveLoss(NormalizedFocalLoss(gamma=2), ReverseCrossEntropy(), a_wt=1.0, p_wt=1.0)
    }
    
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    print("Starting experiments with symmetric noise...")
    for rate in sym_noise_rates:
        print(f"\n=== Training with symmetric noise rate η = {rate} ===")
        
        train_labels = np.array(full_train_data.targets)
        noisy_labels = create_symmetric_noise(train_labels, rate)
        
        train_data_noisy = CIFAR10(root=data_root, train=True, download=False, transform=transform)
        train_data_noisy.targets = noisy_labels.tolist()
        
        train_loader = DataLoader(train_data_noisy, batch_size=64, shuffle=True, num_workers=0)
        
        for loss_name, loss_fn in loss_functions.items():
            print(f"\n--- Training with {loss_name} ---")
            
            model = CNN().to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            
            train_acc_history = []
            test_acc_history = []
            best_test_acc = 0.0
            
            for epoch in range(100):
                model.train()
                train_correct = 0
                train_total = 0
                
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += y.size(0)
                    train_correct += (predicted == y).sum().item()
                
                train_acc = 100 * train_correct / train_total
                test_acc = calculate_accuracy(test_loader, model, device)
                
                train_acc_history.append(train_acc)
                test_acc_history.append(test_acc)
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/100, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
                
                scheduler.step()
            
            plot_title = f"{loss_name} - Symmetric Noise Rate {rate}"
            save_path = f"{results_dir}/{loss_name}_sym_{rate}.png"
            plot_accuracy(train_acc_history, test_acc_history, plot_title, save_path)
            
            print(f"Best Test Accuracy with {loss_name}: {best_test_acc:.2f}%")
    
if __name__ == "__main__":
    main()
