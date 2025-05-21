import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt

# 1⃣️【第二周作业】

# 改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

# 输入：5维向量
# 输出：五维--->取最大值下标 为类别
# 模型  线性->relu->线性->交叉熵

# 创建torch模型
class TorchModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(TorchModel,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # dim=1 表示在维度1上进行softmax
        self.loss = nn.functional.cross_entropy
    def forward(slef,x,y=None):
        x = slef.linear1(x)
        hidden1 = slef.relu(x)
        y_pred = slef.linear1(hidden1)
        if y is not None:
            return slef.loss(y_pred,y)
        else:
            y_pred = slef.softmax(y_pred)
            return y_pred


 



def main():
    #配置参数
    epoch = 20 #训练轮数
    batch_size = 20 #每轮训练个数(每轮输入)
    train_sample = 50000 # 总训练样本(总输入)
    input_size = 5 # 输入维度
    learning_rate = 0.001 # 学习率
    hidden_size = 10 # 隐藏层维度
    output_size = 5 # 输出维度

    #建立模型
    model = TorchModel(input_size,hidden_size,output_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # 画图
    log = []
    # 创建训练集，正常任务是读取训练集
    x_
    pass

if __name__ == '__main__':
    main()