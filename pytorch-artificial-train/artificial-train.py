import hydra
import sys
# sys.path.append('./model_utils')
# print(f'sys.path:{sys.path}')
import torch.optim
from omegaconf import DictConfig
from trainer import Trainer
# from model_utils import kerasmodel,summary
from model_utils.kerasmodel import KerasModel
from model_utils.summary import summary

from torch import nn
from torch.utils.data import DataLoader
from dataset import My_mnist_dataset

from model_utils.kerascallbacks import WandbCallback, MiniLogCallback, TensorBoardCallback
tensorboard_record = TensorBoardCallback(save_dir='runs',
                                         model_name='mnist_cnn',
                                         log_weight=True,
                                          log_weight_freq=5
                               )

# https://github.com/lyhue1991/torchkeras.git

# 模型超参数
batch_size = 32

# 构建数据
trainset = My_mnist_dataset(data_dir=r'D:\my_projects\shiyan\pytorch_lighting_explore\minist_data\MNIST\train',
                            datatype="train")
valset = My_mnist_dataset(data_dir=r'D:\my_projects\shiyan\pytorch_lighting_explore\minist_data\MNIST\test',
                          datatype="test")

# 即使 trainset 每次迭代出来的数据不是tensor, 经过DataLoader也会自动将其转换为tensor的..
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers>=1 就会报有关多线程的错
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)





def create_net():
    net = nn.Sequential()
    net.add_module("conv1",nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3))
    net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5))
    net.add_module("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("dropout",nn.Dropout2d(p = 0.1))
    net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
    net.add_module("flatten",nn.Flatten())
    net.add_module("linear1",nn.Linear(64,32))
    net.add_module("relu",nn.ReLU())
    net.add_module("linear2",nn.Linear(32,10))
    return net


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

        self.correct = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):  # 在后续中,每一步的损失都是调用这里的
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0]
        self.correct += m
        self.total += n

        return m / n

    def compute(self):
        return self.correct.float() / self.total

    def reset(self):
        self.correct -= self.correct
        self.total -= self.total



net = create_net()


# @hydra.main(config_path="./conf", config_name="config")
# def main(cfg: DictConfig):
def main():
    global net
    # print(f'--main--cfg:{cfg}')
    # trainer = Trainer(cfg)
    loss_fn = torch.nn.CrossEntropyLoss()
    model = KerasModel(net,
        loss_fn=loss_fn,
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001),
        metrics_dict = {"acc":Accuracy()},)  # 这里运用的很巧妙
    #-------------------- 查看模型的结构 --------------------#
    input_feature = torch.zeros(32, 1, 28, 28)
    print(f'input_feature.shape:{input_feature.shape}')
    # train_loader
    summary(model, input_data=input_feature)
    #-----------------------------------------------------#
    dfhistory = model.fit(train_data=train_loader,
                          val_data=val_loader,
                          epochs=20,
                          patience=3,
                          monitor="val_acc",
                          mode="max",
                          ckpt_path='checkpoint.pt',
                          plot=True,
                          quiet=False,
                          callbacks=[tensorboard_record] # TODO: 下面要探索这类钩子函数的用法
                          )


if __name__ == '__main__':
    main()
