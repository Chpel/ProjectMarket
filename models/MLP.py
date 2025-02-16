#DRL_Agent
from torch import nn,stack,cat,unflatten

class DispatcherRL(nn.Module):
    def __init__(self, k_actions, k_agents=1, k_outputs=1):
        super(DispatcherRL, self).__init__()
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
       

class DispatcherRL_M(nn.Module): #MARL
    def __init__(self, k_actions, k_agents=1, k_outputs=1, device='cpu'):
        super(DispatcherRL_M, self).__init__()
        self.k_nns = k_agents
        self.k_actors = k_outputs
        self.k_actions = k_actions
        if k_agents == 1: #Dispatcher
            self.sm = nn.Sequential( # bxfx2x9x10
                nn.Flatten(0,1), #(bxf)x2x9x10
                nn.Conv2d(2,4,(3,3), stride=1, device=device, groups=1), #(bxf)x4x7x8
                nn.ReLU(),
                nn.Conv2d(4,8,(3,3), stride=1, device=device, groups=1), #(bxf)x8x5x6
                nn.ReLU(),
                nn.Conv2d(8,16,(3,3), stride=1, device=device, groups=1), #(bxf)x16x3x5
                nn.ReLU(),
                nn.Conv2d(16,32,(3,3), stride=1, device=device, groups=1), #(bxf)x32x1x2
                nn.ReLU(),
                nn.Flatten(1)) #(bxf)x64

            self.dispatcher = nn.Sequential(# bxfx64
                nn.Flatten(1), #bx(fx64)
                nn.Linear(64 * k_outputs, k_actions * k_outputs, device=device)) #bx(fxa)

        else:
            pass #Decentralized

    def forward(self, x): #Centralised only (yet)
        b = x.shape[0]
        f = x.shape[1]
        x = self.sm(x)
        x = unflatten(x, 0, (b, f))
        x = self.dispatcher(x)
        return x.unfold(1, self.k_actions, self.k_actions) #1xfleetxactions