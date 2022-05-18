from torch import nn


class LinearModel(nn.Modlue):
    """ linear模型功能介绍
        这是一个只有一层输出维度为1的线性回归模型
        可以用来求解房价预测等线性回归问题
    """
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forword(self, x):
        return self.linear(x)