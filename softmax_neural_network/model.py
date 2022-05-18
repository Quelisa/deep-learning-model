from torch import nn

class Model(nn.Module):
    """ 模型功能介绍
        这是一个输出维度大于1，只有一层的的线性回归模型，可以进行分类任务
        该模型可以单独使用，也可以和其他模型叠加使用，用softmax做分类
    """
    def __init__(self, input_dim, label_num):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, label_num)

    def forword(self, x):
        return self.linear(x)