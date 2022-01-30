import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision import models
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # get training data
    train_data = CIFAR10(root = "./data/",
                         train = True,
                         download = True)
    
    # normalize and transform
    data_means = train_data.data.mean(axis=(0,1,2,))/255
    data_stds = train_data.data.std(axis=(0,1,2,))/255
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_means,
                             std=data_stds)])
    
    train_data = CIFAR10(root = "./data/",
                         train = True,
                         download = True,
                         transform = train_transforms)
    
    # split into training/valid
    train, valid = torch.utils.data.random_split(train_data, [40000,10000])
    train_batch = torch.utils.data.DataLoader(train,
                                              batch_size=128,
                                              shuffle=True)
    valid_batch = torch.utils.data.DataLoader(valid,
                                              batch_size=128,
                                              shuffle=True)
    
    # get testing data
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_means,
                             std=data_stds)])
    test_data = CIFAR10(root="./data/",
                        train=False,
                        transform=test_transforms)
    test_batch = torch.utils.data.DataLoader(test_data,
                                             batch_size=16)
    
    # set up model and training, use resnet learning transfer
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = models.resnet18(pretrained=True, progress=False)
    resnet.fc = torch.nn.Linear(512, 10, bias=True)
    resnet = resnet.to(device=device)
    
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(resnet.parameters())
    
    # loop
    epochs = 100
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    for epoch in range(epochs):
        
        start = time.time()
        
        # train loop
        losses = 0
        correct = 0
        resnet.train()
        for x, y in train_batch:
    
            x = x.to(device)
            y = y.to(device)
            
            opt.zero_grad()
            
            out = resnet(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            
            losses += loss.item()
            
            pred = torch.argmax(out, axis=1)
            correct += torch.sum(pred == y).item()
        
        train_loss.append(losses/len(train))
        train_acc.append(correct/len(train))
        
        # valid loop
        losses = 0
        correct = 0
        resnet.eval()
        for x, y in valid_batch:
            x = x.to(device)
            y = y.to(device)
            
            out = resnet(x)
            loss = criterion(out, y)
            
            losses += loss.item()
        
            pred = torch.argmax(out, axis=1)
            correct += torch.sum(pred == y).item()
        
        valid_loss.append(losses/len(valid))
        valid_acc.append(correct/len(valid))
        
        # print info
        end = round(time.time() - start)
        print('Epoch: '+str(epoch)+
              ' Time: '+str(end)+
              ' Train Loss: '+str(train_loss[epoch])+
              ' Valid Loss: '+str(valid_loss[epoch])+
              ' Train Acc: '+str(train_acc[epoch])+
              ' Valid Acc: '+str(valid_acc[epoch]))

    # get some testing metrics at end
    losses = 0
    correct = 0
    resnet.eval()
    for x, y in test_batch:
        x = x.to(device)
        y = y.to(device)
        
        out = resnet(x)
        loss = criterion(out, y)
        
        losses += loss.item()
    
        pred = torch.argmax(out, axis=1)
        correct += torch.sum(pred == y).item()
    
    test_loss = losses/len(test_data)
    test_acc = correct/len(test_data)
    print('Test Loss: '+str(test_loss)+
          ' Test Acc: '+str(test_acc))

    # make loss and accuracy plots
    plt.plot(list(range(100)), train_loss, label='train')
    plt.plot(list(range(100)), valid_loss, label='valid')
    plt.plot(99, test_loss, '.', label='test')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('CIFAR-10 ResNet-18 Transfer Learning Loss Curves')
    plt.savefig('plots/loss.png')
    plt.show()
    
    plt.plot(list(range(100)), train_acc, label='train')
    plt.plot(list(range(100)), valid_acc, label='valid')
    plt.plot(99, test_acc, '.', label='test')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('CIFAR-10 ResNet-18 Transfer Learning Accuracy Curves')
    plt.savefig('plots/accuracy.png')
    plt.show()