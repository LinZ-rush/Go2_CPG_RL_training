import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard import SummaryWriter

class BaseVelocityNet(nn.Module):
    def __init__(self,input_dim=45*5,output_dim=3,device="cuda"):
        super(BaseVelocityNet,self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.fc1=nn.Linear(input_dim,128)
        self.bn1=nn.BatchNorm1d(128)
        self.fc2=nn.Linear(128,64)
        self.bn2=nn.BatchNorm1d(64)
        self.fc3=nn.Linear(64,output_dim)
        self.relu=nn.ReLU()
        self.criterion=nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=0.001)
        self.to(self.device)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
    
    def train_model(self,input_data,true_vel,epochs=5,batch_size=256):
        dataset=TensorDataset(input_data,true_vel)
        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
        self.train()
        for epoch in range(epochs):
            total_loss=0.0
            for x_batch,y_batch in dataloader:
                pred=self(x_batch)
                loss=self.criterion(pred,y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
                total_loss+=loss.item()
            avg_loss = total_loss / len(dataloader)
        return avg_loss
    

    def predict(self,input_data):
        self.eval()
        with torch.no_grad():
            return self(input_data)


    def save(self, path="/home/song/unitree_rl_gym/logs/rough_go2/velocity_network/model.pt"):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load(self, path="/home/song/unitree_rl_gym/logs/rough_go2/velocity_network/model.pt"):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])