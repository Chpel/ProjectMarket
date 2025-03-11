#DRL_Agent
from torch import nn,stack,cat,unflatten

class BrokerRL(nn.Module):
    def __init__(self, k_actions, k_agents=1, k_outputs=1):
        super(BrokerRL, self).__init__()
        self.k_nns = k_agents
        self.k_actors = k_outputs
        self.model = nn.Sequential( # 1x1x2x9x10
            nn.Conv3d(1,4,(2,3,3), stride=1), #1x4x1x7x8
            nn.Flatten(1,2),
            nn.ReLU(),
            nn.Conv2d(4,8,(3,3), stride=1), #1x8x5x6
            nn.ReLU(),
            nn.Conv2d(8,16,(3,3), stride=1), #1x16x3x5
            nn.ReLU(),
            nn.Conv2d(16,32,(3,3), stride=1), #1x32x1x2
            nn.ReLU(),
            nn.Flatten(1)) #1x128

        self.fcs = nn.Linear(64, k_actions)
            

    def forward(self, x): #Centralised only (yet)
        x = self.model(x)
        res = self.fcs(x).unsqueeze(1)
        #print(res.shape)
        return res
       