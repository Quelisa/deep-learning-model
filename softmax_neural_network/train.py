from model import Model
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def GetDataset(filename, batch_size):
    data = []
    label = []
    with open(filename, 'r', encoding='-utf-8') as f:
        for line in f:
            line = line.strip('').split('\t')
            data = line[0]
            label = line[1]
    data_tensor = torch.tensor(data, dtype=torch.long)
    label_tensor = torch.tensor(label, dtype=torch.long)
    return data.TensorDataset(data_tensor, label_tensor)


def accuracy(y_pred, y):
    """
    y_pred是一个矩阵，第二个维度储存每个类的预测分数
    使用argmax获取其每行中最大元素的索引
    因为“==”对数据类型敏感，所以先对y_pred进行数据类型转换
    """
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(axis=1)
    cmp = y_pred.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def dev(model, loss, dataloader):
    total_num = 0
    acc_num = 0
    for step, batch in enumerate(tqdm(dataloader)):
        batch = tuple(b.cuda for b in batch)
        X = batch[0]
        y = batch[1]
        total_num += len(y)
        acc_num += accuracy(model(X), y)
    return acc_num/total_num


def train(dataset, model, loss, optimazer, epochs):
    train_dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dataset["dev"], batch_size=args.batch_size, shuffle=True)
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = tuple(b.cuda for b in batch)
            X = batch[0]
            y = batch[1]
            l = loss(model(X), y)
            optimazer.zero_grad()
            l.backward()
            optimazer.step()
        acc = dev(model, loss, dev_dataloader)
        print('epoch: %d acc: %f' % (epoch, acc))

def predict(dataset, model, epoch):
    acc_num = 0
    total_num = 0
    test_dataloader = DataLoader(dataset["test"], batch_size=args.batch_size, shuffle=True)
    for epoch in range(epoch):
        batch = tuple(b.cuda for b in batch)
        for step, batch in enumerate(tdqm(test_dataloader)):
            X = batch[0]
            y = batch[1]
            y_pred = model(X)
            total_num += len(y)
            acc_num += accuracy(y_pred, y)
    return acc_num/total_num


def main(args):
    dataset["train"] = GetDataset(args.train_file, args.batch_size)
    dataset["dev"] = GetDataset(args.dev_file, args.batch_size)
    dataset["test"] = GetDataset(args.test_file, args.batch_size)
    model = Model(args.input_dim)
    loss = nn.CrossEntropyLoss()
    optimazer = torch.optim.SGD(model.paramers(), args.lr)
    train(dataset, model, loss, optimazer, args.epoch)

    if args.predict:
        predict(dataset["test"], model, args.epoch)
        print('test acc: %f'% acc)


if __name__ == '__main__':
    """ 模块功能介绍
        训练一个线性回归模型，来预测实际问题
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=True, help="train file data directory")
    parser.add_argument("--dev_file", default=None, type=str, required=True, help="dev file data directory")
    parser.add_argument("--test_file", default=None, type=str, required=False, help="test file data directory")
    parser.add_argument("--batch_size", default=None, type=int, required=True, help="batch size")
    parser.add_argument("--epoch", default=None, type=int, required=True, help="epoch")
    parser.add_argument("--lr", default=None, type=float, required=True, help="learning rate")
    parser.add_argument("--input_dim", default=None, type=float, required=True, help="number of features")
    parser.add_argument("--label_num", default=None, type=float, required=True, help="number of labels")
    parser.add_argument("--predict", default=None, type=bool, required=True, help="whether need to predict")
    args = parser.parse_args()

    main(args)