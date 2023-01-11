import torch
import numpy as np
import torch.nn as nn

class Network(nn.Module):
    
    def __init__(self):
        super(Network,self).__init__()
        
        ###### parameters ######
        self.kernel = 6
        self.m = 30
        self.drop_ratio = 0.5
        self.CNNhid = 100
        self.RNNhid = 100
        
        ###### CNN ######
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=self.CNNhid,kernel_size=(self.kernel,self.m)),
            nn.Tanh(),
            nn.Dropout(self.drop_ratio)
        )
        
        ###### GRU ######
        self.gru = nn.GRU(input_size=self.CNNhid, hidden_size=self.RNNhid)
        
        ###### output ######
        self.output = nn.Sequential(
            nn.Linear(self.RNNhid,self.m),
            # nn.Sigmoid()
        )
        
    def forward(self,x):
        
        ###### CNN ######
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.squeeze(3)
        
        ###### GRU ######
        x = x.permute(2,0,1).contiguous()
        _, hidden = self.gru(x)
        self.gru_drop = nn.Dropout(self.drop_ratio)
        hidden = self.gru_drop(hidden.squeeze(0))
        
        ###### output ######
        output = self.output(hidden)
        
        return output

if __name__ == "__main__":
    # input_data: [batch, window_size, state_size]
    # output should have shape: [batch, state_size]
    # e.g.
    # input: [128,41,30] -> output: [128,30]
    x = torch.randn(128,41,30)
    model = Network()
    out = model(x)
    print("shape of output: {}".format(out.shape))