import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
from torch.optim import Adam

class Trainer:
    def __init__(self, config):
        '''
        :param config: 训练过程中的一些配置文件
        '''
        # 模型训练中的配置
        self.model = None
        self.optimizer = None
        self.start_epoch = 0
        # 使用tensorboard记录一些训练过程中的一些值
        print(f'--in-trainer-config:{config}')
        # self.tb_writer = SummaryWriter(log_dir=config.dir)

    def train(self, train_loader,val_loader):
        for epoch_i in range(self.start_epoch, self.options['epoch']): #
            start = time.time()
            train_loss, train_acc = self.train_epoch(train_loader) # 这个引入的好



    def train_epoch(self,data_loader):
        self.model.train()

        for batch in tqdm(data_loader, desc=' -(Training) ', leave=False): #
            # forward
            self.optimizer.zero_grad()
            loss, metric = self.model(batch)
            # backward and update parameters
            loss.backward()
            self.optimizer.step()