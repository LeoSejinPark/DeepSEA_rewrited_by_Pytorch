import torch.nn.functional as F
import torch
import torch.nn as nn

class Deep_SEA(nn.Module):
    def __init__(self, conv1_ch=320,conv2_ch=480,conv3_ch=960,drop_rate1 =0.5, drop_rate2 = 0.2, drop_rate3 = 0.2):
        super(Deep_SEA, self).__init__( )
        NUM_OUTPUTS = 919
        
        self.convs1 = nn.Conv2d(1, conv1_ch, (8, 4))
        self.pool = nn.MaxPool2d(1,4)
        self.dropout1 = nn.Dropout(drop_rate1)
        self.convs2 = nn.Conv2d(1, conv2_ch, (conv1_ch, 8))
        self.dropout2 = nn.Dropout(drop_rate2)
        self.convs3 = nn.Conv2d(1, conv3_ch, (conv2_ch, 8))
        self.dropout3 = nn.Dropout(drop_rate3)
        self.fc1 = nn.Linear(conv3_ch*54, 925)
        self.relu = nn.ReLU()
        self.dropout4 = nn.Dropout(drop_rate3)
        self.fc2 = nn.Linear(925, NUM_OUTPUTS)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.convs1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.squeeze(-1)
        x = x.unsqueeze(1)
        x = self.relu(self.convs2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = x.squeeze(-2)
        x = x.unsqueeze(1)
        x = self.relu(self.convs3(x))
        x = self.dropout3(x)
        x = x.squeeze(-2)
        x = torch.flatten(x,start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        x = self.sigmoid(x)
        return x
   
class CNN_BiLSTM_Attension(nn.Module):
    def __init__(self, conv_dr=0.2 , lstm_dr=0.3,fc_dr=0.5):
        super(CNN_BiLSTM_Attension, self).__init__()
        
        self.convs1 = nn.Conv2d(1, 320, (26, 4))
        self.pool = nn.MaxPool2d(1,13) 
        self.pool3d = nn.MaxPool3d(1,4) 
        self.lstm = nn.LSTM(320, 320,
                            num_layers=1,
                            dropout=lstm_dr,
                            bidirectional=True,
                            batch_first=True)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(conv_dr)
        self.sigmoid = nn.Sigmoid()
        
        self.dropout2 = nn.Dropout(fc_dr)
        self.fc = nn.Linear(320*2, 919)
        
    def attention_net(self, lstm_output, final_hidden_state):
        hidden = final_hidden_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
        return new_hidden_state
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.convs1(x)
        x = self.pool(x)
        
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        sum_hidden = torch.cat((final_hidden_state[0],final_hidden_state[1]),1)
        new_hidden_state = self.attention_net(x,sum_hidden.unsqueeze(0))
        x = new_hidden_state
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc(x) 
    
        x = self.sigmoid(x)
        return x