import numpy as np
import pandas as pd
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility
import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from models import Deep_SEA, CNN_BiLSTM_Attension

save_to_dir = 'output/result.csv'
print ('loading data')

try:
    print(len(trainmat),'Already dataset here')
except:
    trainmat = h5py.File('data/train.mat')
    validmat = scipy.io.loadmat('data/valid.mat')
    testmat = scipy.io.loadmat('data/test.mat')
    
    X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
    y_train = np.array(trainmat['traindata']).T
    X_test = np.transpose(testmat['testxdata'],axes=(0,2,1))
    y_test = testmat['testdata']


torch.cuda.set_device(4)
device = torch.device('cuda')
model = CNN_BiLSTM_Attension(conv_dr=0.2 , lstm_dr=0.3,fc_dr=0.5)
model = Deep_SEA(conv1_ch=320,conv2_ch=480,conv3_ch=960,drop_rate1 =0.5, drop_rate2 = 0.2, drop_rate3 = 0.2)
model.to(device)

data_size = len(X_train)
batch_division_size = 10
batch_size = data_size//batch_division_size
mini_batch_size = 16
max_epoch = 100
l1 = 1e-08
l2 = 5e-07
wd = 8e-7
learning_rate = 1e-2
cnn_optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr':l2}],lr= learning_rate ,weight_decay =wd)
loss_fn = torch.nn.BCELoss()
test_auc_list = []

for epoch in range(max_epoch):
    losses= 0
    for batch in range(batch_division_size):
        X_tr = X_train[batch_size*(batch):batch_size*(batch+1)]
        y_tr = y_train[batch_size*(batch):batch_size*(batch+1)]
        
        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr.astype(int)))
        trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size= mini_batch_size, shuffle=False, num_workers= 0)
        
        model.train()
        for i, (x, target) in enumerate(trainLoader):
            y = target.float().to(device)
            x = x.to(device)
            pred_y = model(x)
            pred_y = pred_y.to(device)
            loss = loss_fn(pred_y,y)
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
                
            losses += loss.item()+l1*regularization_loss
            cnn_optimizer.zero_grad()
            loss.backward()
            cnn_optimizer.step()
            
        print("batch:",batch, "  losses:", losses/(data_size*(batch+1)))
            
    with torch.no_grad():
        temp_AUC_list = []
        for i in range(len(X_test)//100):
            x_ts = X_test[100*(i):100*(i+1)]
            y_ts = y_test[100*(i):100*(i+1)]
            x_ts  = torch.FloatTensor(x_ts)
            y_ts  = torch.FloatTensor(y_ts)
            x_ts = x_ts.to(device)
            
            model.eval()
            pred_y = model(x_ts)
            pred_y = pred_y.cpu()
            test_AUC = roc_auc_score(y_ts.view(-1,1).detach().numpy(),pred_y.view(-1,1).detach().numpy())
            temp_AUC_list.append(test_AUC)
            
        temp_mean_auc = torch.mean(torch.Tensor(temp_AUC_list))
        test_auc_list.append(temp_mean_auc)
        print("test_AUC: ",temp_mean_auc)
        temp_df = pd.DataFrame(data=test_auc_list)
        temp_df.to_csv(save_to_dir,sep='\t',index=None)
            
    print("epoch: ",epoch, "  avg_losses: ", losses/data_size)
            
        
