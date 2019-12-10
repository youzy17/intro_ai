#!/usr/bin/env python
# coding: utf-8

# In[58]:


import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
from sklearn.metrics import accuracy_score,f1_score,roc_curve,precision_recall_curve,average_precision_score,auc
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,matthews_corrcoef,roc_auc_score
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from PIL import Image


# In[59]:



pre_train = np.load("E:\\Document_Yzy\\人工智能基础\\第二次大作业\\train.npy")
pre_test = np.load("E:\\Document_Yzy\\人工智能基础\\第二次大作业\\test.npy")
pre_label = pd.read_csv("E:\\Document_Yzy\\人工智能基础\\第二次大作业\\train.csv") 
print(pre_label.info())
print(pre_label.head())

label = pre_label['label']
label_list = label
print(type(label_list))
print(type(pre_train))


# In[80]:


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(1),  #图像一半的概率翻转，一半的概率不翻转
])


# In[76]:


BATCH_SIZE = 128




class adddataset(Data.Dataset):
    def __init__(self,trans=None):
        self.trans = trans
        self.data,self.labels = pre_train,label_list
        
    def __getitem__(self,index):
        image2=[]
        image1=self.train_data[index]
        for i in range(28):
            for j in range(28):
                if j == 0:
                    image2.append([])
                image2[i].append(image1[j + i * 28])
                
        img, target = image2, self.train_labels[index]
        
        if self.trans is not None:
            img = self.transform(img)
            
        return img, target
    
    def __len__(self):
        return 30000

class mydataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set


        if self.train:
            self.train_data, self.train_labels = pre_train, label_list
        else:
            self.test_data = pre_test

    def __getitem__(self, index):
        if self.train:
            image2=[]
            image1=self.train_data[index]
            for i in range(28):
                for j in range(28):
                    if j == 0:
                        image2.append([])
                    image2[i].append(image1[j + i * 28])
            
            img, target = image2, self.train_labels[index]
        else:
            image2=[]
            image1 = self.test_data[index]
            for i in range(28):
                for j in range(28):
                    if j == 0:
                        image2.append([])
                    image2[i].append(image1[j + i * 28])

            img = image2


        if self.transform is not None:
            img = np.array(img)
            img = self.transform(img)
        

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target
        
        else :
            return img

    def __len__(self):
        if self.train:
            return 30000
        else:
            return 5000

data1 = train_dataset
train_dataset = adddataset(trans=transform_train)
print(type(train_dataset))
data2 = train_dataset
train_dataset = data1+data2
print(len(train_dataset))
print(len(data1))
print(len(data2))

train_dataset = mydataset(train=True,transform=transforms.ToTensor())
test_dataset = mydataset(train=False,transform=transforms.ToTensor())
#加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor
#train_dataset = datasets.MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
#test_dataset = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor())

#加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)


# In[74]:


# image2=[]
# image1=pre_train[0]
# for i in range(28):
#     for j in range(28):
#         if j == 0:
#             image2.append([])
#         image2[i].append(image1[j + i * 28])
# img = np.array(image2)


# In[66]:


#定义resnet
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        x = F.dropout(x, training=self.training)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)






# #定义网络模型亦即Net 这里定义一个简单的全连接层784->10
# class Model(nn.Module):
#     # def __init__(self):
#     #     super(Model,self).__init__()
#     #     self.linear1 = nn.Linear(784,784)
#     #     self.linear2 = nn.Linear(784,10)
#     # def forward(self,X):
#     #     X = F.relu(self.linear1(X))
#     #     return F.relu(self.linear2(X))

#     # def __init__(self):
#     #     super(Model,self).__init__()
#     #     self.linear1 = nn.Linear(784,784)
#     #     self.linear2 = nn.Linear(784,784)
#     #     self.linear3 = nn.Linear(784,10)
#     # def forward(self,X):
#     #     X = F.relu(self.linear1(X))
#     #     X = F.relu(self.linear2(X))
#     #     return F.relu(self.linear3(X))

#     def __init__(self):
#         super(Model,self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
 
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         #x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x


# In[67]:


model = ResNet18()#.cuda() #实例化卷积层
loss = nn.CrossEntropyLoss() #损失函数选择，交叉熵函数
optimizer = optim.SGD(model.parameters(),lr = 0.1)
num_epochs = 18


# In[68]:


losses = [] 
acces = []
eval_losses = []
eval_acces = []
print(len(train_loader))
for echo in range(num_epochs):
    train_loss = 0   #定义训练损失
    train_acc = 0    #定义训练准确度
    model.train()    #将网络转化为训练模式
    for i,(X,label) in enumerate(train_loader):     #使用枚举函数遍历train_loader
        #X = X.view(-1,784)       #X:[64,1,28,28] -> [64,784]将X向量展平
        X = Variable(X)#.cuda()          #包装tensor用于自动求梯度
        label = Variable(label)#.cuda()
        out = model(X)           #正向传播
        lossvalue = loss(out,label)         #求损失值
        optimizer.zero_grad()       #优化器梯度归零
        lossvalue.backward()    #反向转播，刷新梯度值
        optimizer.step()        #优化器运行一步，注意optimizer搜集的是model的参数
        
        #计算损失
        train_loss += float(lossvalue)      
        #计算精确度
        _,pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        train_acc += acc
        if i%10==0:
            print("times:"+str(i))

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print("echo:"+' ' +str(echo))
    print("lose:" + ' ' + str(train_loss / len(train_loader)))
    print("accuracy:" + ' '+str(train_acc / len(train_loader)))


# In[ ]:


# eval_loss = 0
# eval_acc = 0

pred_all = None
pred_pro_all = None
model.eval() #模型转化为评估模式
for X in test_loader:
    #X = X.view(-1,784)
    X = Variable(X)#.cuda()
    with torch.no_grad():
        testout = model(X)
    #testloss = loss(testout)
    #eval_loss += float(testloss)

    _, pred = testout.max(1)


    if pred_all is None:
        pred_all = torch.cat([pred])
    else:
        pred_all = torch.cat([pred_all,pred])

    if pred_pro_all is None:
        pred_pro_all = torch.cat([F.sigmoid(testout)])
    else:
        pred_pro_all = torch.cat([pred_pro_all,F.sigmoid(testout)])
#     num_correct = (pred == label).sum()
#     acc = int(num_correct) / X.shape[0]
#     eval_acc += acc

#y_test = label_all.cpu().detach().numpy()
#print(y_test)
y_pred = pred_all.cpu().detach().numpy()
#print(y_pred)
y_pred_pro = pred_pro_all.cpu().detach().numpy()

y_pred=y_pred.tolist()


# In[ ]:


list = y_pred
column=['label']
#column = ['dr','label']  #列表对应每列的列名

test = pd.DataFrame(columns=column,data=list)
print(test)
test.to_csv("E:\\Document_Yzy\\人工智能基础\\第二次大作业\\pred2.csv") 


# In[ ]:


print('ACC:%.7f' %accuracy_score(y_true=y_test, y_pred=y_pred))
print('Precision-macro:%.7f' %precision_score(y_true=y_test, y_pred=y_pred,average='macro'))
print('Recall-macro:%.7f' %recall_score(y_true=y_test, y_pred=y_pred,average='macro'))
print('F1-macro:%.7f' %f1_score(y_true=y_test, y_pred=y_pred,average='macro'))       


# In[ ]:


fpr = dict()
tpr = dict()
roc_auc = dict()
average_precision = dict()
recall = dict()
precision = dict()
for i in range(10):
    y_test2 = copy.deepcopy(y_test)
    y_test2[y_test2!=i] = 10
    y_test2[y_test2==i] = 1
    y_test2[y_test2==10] = 0
    y_pred_pro2 = y_pred_pro[:,i]
    #print(y_pred_pro2)
    #print(y_test2)
    fpr[i], tpr[i], _ = roc_curve(y_test2, y_pred_pro2)
    roc_auc[i] = roc_auc_score(y_test2,y_pred_pro2)


    average_precision[i] = average_precision_score(y_test2, y_pred_pro2)
    #print('Average precision-recall score: %.7f' % average_precision)
    precision[i], recall[i], _ = precision_recall_curve(y_test2,y_pred_pro2)

# Plot of a ROC curve for a specific class
colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - MNIST')
for i in range(10):
    plt.plot(fpr[i], tpr[i], label="class" + str(i) + ':ROC curve (area = %0.3f)' % roc_auc[i],color=colors[i])
plt.legend(loc="lower right")
#plt.savefig("roc.png")


# Plot of a ROC curve for a specific class
colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('precision recall curve - MNIST')
for i in range(10):
    plt.plot(recall[i], precision[i], label="class" + str(i) + ':AP (score = %0.3f)' % average_precision[i],color=colors[i])
plt.legend(loc="lower right")
#plt.savefig("pro.png")


# In[ ]:




