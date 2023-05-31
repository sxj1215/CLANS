import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from functools import reduce


class ActorCritic(nn.Module):
    def __init__(self, t, space):  # t为旧任务总数，space为搜索空间
        super(ActorCritic, self).__init__()
        self.t = t
        self.affine = nn.Linear(t, 100)
        # self.affine1=nn.Linear(t,64)
        # self.affine2=nn.Linear(64,100)

        self.action_layer1 = nn.GRUCell(100, 100)
        self.action_layer2 = nn.Linear(100, space)

        self.value_layer = nn.Linear(100, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state, action_num):  # state为旧任务相似度序列，长度为t，action_num为需要决策的扩展动作个数

        state = F.relu(self.affine(state))
        # state=F.relu(self.affine2(F.relu(self.affine1(state))))

        state_value = self.value_layer(state)

        hidden_state = torch.zeros_like(state)  # 初始化
        action = []
        logprobs = []
        for _ in range(action_num):
            hidden_state = self.action_layer1(state.view(1, 100), hidden_state.view(1, 100))
            opt=self.action_layer2(hidden_state)
            opt_max=torch.max(opt)
            action_probs = F.softmax(opt-opt_max)#防止softmax溢出
            #print('hid',hidden_state,'prob',action_probs)
            action_distribution = Categorical(action_probs)
            act = action_distribution.sample()
            action.append(act)
            logprobs.append(action_distribution.log_prob(act))

        '''action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        self.logprobs.append(action_distribution.log_prob(action))'''

        self.logprobs.append(logprobs)
        self.state_values.append(state_value)

        return action

    def calculateLoss(self, num_layers):  # 可变更为不同的方法

        # calculating discounted rewards
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + 1 * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        a=rewards.std()
        if not torch.isnan(a):
            rewards = (rewards - rewards.mean()) / (rewards.std())

        loss = 0
        action_loss=0
        for logprobs,value, reward in zip(self.logprobs,self.state_values,rewards):
                advantage = reward - value.item()
                for i in range(num_layers):
                    action_loss += reduce(lambda x, y: x * y, logprobs[i])
                action_loss = (-action_loss)* advantage
                value_loss = F.smooth_l1_loss(value.cuda(), reward.cuda())
                loss += (action_loss.cuda() + value_loss)
        return loss

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
