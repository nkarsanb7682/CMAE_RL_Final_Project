import torch
from torch import nn


'''
X = torch.rand(1, 28, 28, device=device) # Generate random data
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
'''


class HER_DQN(nn.Module):
    def __init__(self):
        super(HER_DQN, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)

        self.l1 = nn.Linear(1 * 1, 8)
        self.r1 = nn.ReLU()
        # self.l2 = nn.Linear(20, 20)
        # self.r2 = nn.ReLU()
        self.l2_1 = nn.Linear(8, 4)
        self.r2_1 = nn.ReLU()
        self.l3_agent1 = nn.Linear(4, 4)
        self.l3_agent2 = nn.Linear(4, 4)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # flatten = self.flatten(x)
        l1 = self.l1(x)
        r1 = self.r1(l1)
        # l2 = self.l2(r1)
        # r2 = self.r2(l2)
        l2_1 = self.l2_1(r1)
        r2_1 = self.r2_1(l2_1)
        l3_agent1 = self.l3_agent1(r2_1)
        l3_agent2 = self.l3_agent2(r2_1)
        # print("l3_agent2", l3_agent2)
        # print("self.softmax(l3_agent2)", self.softmax(l3_agent2))
        return [self.softmax(l3_agent1), self.softmax(l3_agent2), l3_agent1, l3_agent2]


# class HER_DQN(nn.Module):
#     def __init__(self):
#         super(HER_DQN, self).__init__()
#         self.flatten = nn.Flatten(start_dim=1)

#         self.l1_agent1 = nn.Linear(4 * 1, 20)
#         self.l1_agent2 = nn.Linear(4 * 1, 20)
#         self.r1 = nn.ReLU()
#         self.l2 = nn.Linear(40, 20)
#         self.r2 = nn.ReLU()
#         self.l3_agent1 = nn.Linear(20, 4)
#         self.l3_agent2 = nn.Linear(20, 4)

#     def forward(self, x):
#         # flatten = self.flatten(x)
#         agent1Coords = x[:, :2]
#         agent2Coords = x[:, 2:4]
#         boxCoords = x[:, 4:6]
#         l1_agent1 = self.l1_agent1(torch.cat((agent1Coords, boxCoords), 1))
#         l1_agent2 = self.l1_agent2(torch.cat((agent2Coords, boxCoords), 1))
#         r1 = self.r1(torch.cat((l1_agent1, l1_agent2), 1))
#         l2 = self.l2(r1)
#         r2 = self.r2(l2)
#         l3_agent1 = self.l3_agent1(r2)
#         l3_agent2 = self.l3_agent2(r2)
#         return [l3_agent1, l3_agent2]