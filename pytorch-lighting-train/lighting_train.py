import os
"""
怎么感觉这个还相当不成熟(就是垃圾),还是我没有熟悉,怎知感觉代码是规范了,但是完全不知道数据的流向(训练日志与验证日志交叉打印)
"""
from abc import ABC
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
# import lightning.pytorch as pl
import pytorch_lightning as pl
from dataset import My_mnist_dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint  # 保存模型相关的

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    # dirpath='./lightning_checkpoints/',
    dirpath = r'D:\my_projects\shiyan\pytorch_lighting_explore\pytorch-lighting-train\lightning_checkpoints',
    filename='model-{epoch:02d}-{val_loss:.2f}'
)

early_stopping = pl.callbacks.EarlyStopping('val_acc')
# -------------训练过程中的超参数--------------------#
batch_size = 32
# -------------step1: 构建数据加载器-----------------#
####### ---------传统写法,可能不能很好的在trainer().fit中使用------------ ###############
trainset = My_mnist_dataset(data_dir=r'D:\my_projects\shiyan\pytorch_lighting_explore\minist_data\MNIST\train',
                            datatype="train")
valset = My_mnist_dataset(data_dir=r'D:\my_projects\shiyan\pytorch_lighting_explore\minist_data\MNIST\test',
                          datatype="test")
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

####### ---------


# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def computer(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.computer(batch)
        self.log('val_loss', val_loss)  # 只有了这行代码, 这样ModelCheckpoint才能监控它。才能根据该指标保留最有模型
        return val_loss

    def test_step(self, batch, batch_idx):
        loss = self.computer(batch)
        return {'test_loss': loss}

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self.computer(batch)
        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
print(f'autoencoder:{autoencoder}')

# trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
# trainer.fit(model=autoencoder,train_dataloaders=train_loader)

trainer = Trainer(limit_train_batches=100, max_epochs=3, default_root_dir='./lightning_checkpoints/') # 可以保存模型
# trainer = Trainer(limit_train_batches=100, max_epochs=3, callbacks=[checkpoint_callback]) # callbacks=[checkpoint_callback,early_stopping])
trainer.fit(model=autoencoder,train_dataloaders=train_loader,val_dataloaders=val_loader)



