from torch import nn

class Model(nn.Module):
    """ 模型功能介绍
        这是一个只有一层线性层的神经网络，可以用来求解房价预测等线性回归问题
    """
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forword(self, x):
        return self.linear(x)