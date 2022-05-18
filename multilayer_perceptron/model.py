from torch import nn


class MLPModel(nn.Module):
    """ MLP模型功能介绍
        这是一个输出维度大于1，只有一层的的线性回归模型，可以进行分类任务
        该模型可以单独使用，也可以和其他模型叠加使用，用softmax做分类
    """
    def __init__(self, input_dim, hid_dim, label_num):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hid_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hid_dim, label_num)

    def forword(self, x):
        return self.layer2(self.relu(self.layer1(x)))