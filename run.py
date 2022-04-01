import torch
import os
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F


transform = transforms.Compose([
                                transforms.Resize(227),
                                transforms.CenterCrop(227),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


nc = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# log_dir = "/content/logs"
device_ids = [0]
batch_size = 128
lr = 0.01
momentum = 0.9
weight_decay = 0.0005
epochs = 90
checkpoint_dir = "/content/checkpoints"


class AlexNet(nn.Module):
  def __init__(self,noc=1000):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4), # Output size --> (bx96x55x55)
        nn.ReLU(),
        nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
        nn.MaxPool2d(kernel_size=3,stride=2), # Output size --> (bx96x27x27)
        nn.Conv2d(96,256,5,padding=2), # Output size --> (bx256x27x27)
        nn.ReLU(),
        nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
        nn.MaxPool2d(kernel_size=3,stride=2), # Output size --> (bx256x13x13)
        nn.Conv2d(256,384,3,padding=1), # Output size --> (bx384x13x13)
        nn.ReLU(),
        nn.Conv2d(384,384,3,padding=1), # Output size --> (bx384x13x13)
        nn.ReLU(),
        nn.Conv2d(384,256,3,padding=1), # Output size --> (bx256x13x13)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2) # Output size --> (bx256x6x6)
    )

    self.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=(256*6*6), out_features=4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096,out_features=nc)
    )
    
    self.bias()

  def bias(self):
    for layer in self.net:
      if isinstance(layer,nn.Conv2d):
        nn.init.normal_(layer.weight,mean=0,std=0.01)
        nn.init.constant_(layer.bias,0)
      
    nn.init.constant_(self.net[4].bias,1)
    nn.init.constant_(self.net[10].bias,1)
    nn.init.constant_(self.net[12].bias,1)
  
  def forward(self,x):
    x = self.net(x)
    x = x.view(-1,256*6*6)
    res = self.classifier(x)
    return res
  
  
  
if __name__ == '__main__':
  seed = torch.initial_seed()
  

  alexnet = AlexNet(noc=nc).to(device)
  alexnet = torch.nn.parallel.DataParallel(alexnet,device_ids=device_ids)
  print(alexnet)

  train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
  test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,
                                           shuffle=True, num_workers=2)
  

  optimizer = optim.Adam(alexnet.parameters(),lr=0.0001)
      # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=lr,
    #     momentum=momentum,
    #     weight_decay=weight_decay)

  lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)


  print("TRAINING STARTS")
  total_steps = 1
  for epoch in range(epochs):
    optimizer.step()
    for imgs,classes in trainloader:
      imgs = imgs.to(device)
      classes = classes.to(device)

      output = alexnet(imgs)
      loss = F.cross_entropy(output,classes)

      optimizer.zero_grad()
      torch.autograd.set_detect_anomaly(True)
      loss.backward()
      lr_scheduler.step()
    
    checkpoint_path = os.path.join(checkpoint_dir, 'alexnet_states_e{}.pkl'.format(epoch + 1))
    state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
    torch.save(state, checkpoint_path)
    print(f"Epoch {epoch} done")
