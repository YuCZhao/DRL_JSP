
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_FNN(nn.Module):#卷积神经网络&前馈神经网络
    def __init__(self,J_num,O_max_len):
        super(CNN_FNN, self).__init__()
        self.fc1 = nn.Linear(6 * int(J_num / 2) * int(O_max_len / 2), 258)
        self.fc2 = nn.Linear(258, 258)
        self.out = nn.Linear(258, 17)

        def forward(self, x):
            x = self.conv1(x)
            x = x.view(x.size(0), -1)#将卷积层的输出展平成一维向量。
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            #通过输出层 self.out 生成最终的动作概率（或者称为 Q 值，具体取决于上下文）。
            action_prob = self.out(x)
            return action_prob

class CNN_dueling(nn.Module):
    def __init__(self,J_num,O_max_len):
        super(CNN_dueling, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=3,
                      stride=1,
                      padding=1,),# 输出形状为 (3, J_num, O_max_len)
            nn.ReLU(),nn.MaxPool2d(kernel_size=2,ceil_mode=False)# 最大池化层，核大小为 2x2，不使用 ceil_mode
        )

        # 值函数（Value Function）的隐藏层、优势函数（Advantage Function）的隐藏层、值函数和优势函数。
        self.val_hidden=nn.Linear(6*int(J_num/2)*int(O_max_len/2),258)
        self.adv_hidden=nn.Linear(6*int(J_num/2)*int(O_max_len/2),258)
        self.val=nn.Linear(258,1)
        self.adv=nn.Linear(258,17)

    def forward(self,x):
        x=self.conv1(x)
        x = x.view(x.size(0), -1)  # 将卷积层的输出展平成一维向量。

        val_hidden=self.val_hidden(x)
        val_hidden=F.relu(val_hidden)

        adv_hidden=self.adv_hidden(x)
        adv_hidden=F.relu(adv_hidden)

        val=self.val(val_hidden)
        adv=self.adv(adv_hidden)

        adv_ave=torch.mean(adv,dim=1,keepdim=True)#优势函数的均值 adv_ave
        x=adv+val-adv_ave

        return x







