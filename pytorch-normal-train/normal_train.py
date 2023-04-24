import torch.cuda
from dataset import My_mnist_dataset
from model import Net
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

tensorboard_writer = SummaryWriter('./log') # 指定tensorboard记录位置, 且这个文件夹不需要手动创建...
"""
tensorboard --logdir=log/ --port=8889 在浏览器中就可以看到对应的数据   log/一定不要加引号
"""
# 训练超参数
EPOCH = 10
num_classes = 10
batch_size = 32

# 构建数据
trainset = My_mnist_dataset(data_dir=r'D:\my_projects\shiyan\pytorch_lighting_explore\minist_data\MNIST\train',
                            datatype="train")
valset = My_mnist_dataset(data_dir=r'D:\my_projects\shiyan\pytorch_lighting_explore\minist_data\MNIST\test',
                          datatype="test")

# 即使 trainset 每次迭代出来的数据不是tensor, 经过DataLoader也会自动将其转换为tensor的..
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers>=1 就会报有关多线程的错
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 训练
tot_step = 0
for epoch in tqdm(range(EPOCH)):
    model.train()
    print(f'epoch:{epoch}')
    train_loss = 0.0
    for step, train_epoch in enumerate(train_loader):
        img_Datas, labels = train_epoch  # train_epoch中的2个元素都是tensor格式的..
        img_Datas = img_Datas.to(device)
        labels = labels.to(device)
        model = model.to(device)
        # print(f'img_Data-shape:{img_Datas.shape}')
        optimizer.zero_grad()  ###############---------------三件套1

        output = model(img_Datas)

        predicted = torch.max(output, 1)
        acc = (predicted[1].numpy() == labels.numpy()).sum() / batch_size
        # print(f'acc:{acc}')
        label_one_hot = F.one_hot(labels, num_classes)
        # print('labels-shapeL',label_one_hot.shape)
        # loss = loss_fn(output, label_one_hot.float())
        loss = loss_fn(output, labels)

        train_loss += loss.item() / batch_size
        tensorboard_writer.add_scalar('loss/train_step', loss, tot_step+step)  # 其中loss是要存的值, epoch是轴
        tensorboard_writer.flush()
        if step % 200 == 0:
            print('=' * 6)
            # best_weights_path = './mnist_train_acc_%.3f' % acc
            # torch.save(model.state_dict(), best_weights_path) # best_weights_path里面有个":"居然就不能保存了!!!!
            print(f'-------- 模型输入相关的 -----------------')
            print(f'img_Datas-shape:{img_Datas.shape}')  # [batch_size, 1, 28, 28] .dtype=torch.float
            print(f'labels:{labels}')  # torch([1,2,..,3]) ->共有batch_size个元素,.dtype = torch.int64
            print(f'label_one_hot:{label_one_hot.float()}')  # (batch_size, num_class)
            print(f'-------- 模型输出相关的 -----------------')
            print(f'output-shape:{output.shape}')
            print(f'predicted:{predicted[1].numpy()[:5]}')
            print(f'acc:{acc}')
            print(f'labels:{labels[:5]}')
            print(f'epoch:{epoch},step:{step},loss:{loss}')
            print('-' * 6)

        loss.backward()  ###############---------------三件套2
        optimizer.step()  ###############---------------三件套3
    tot_step += len(train_loader)
    tensorboard_writer.add_scalar('loss/train_epoch', loss, epoch)  # 其中loss是要存的值, epoch是轴
    tensorboard_writer.add_scalar('acc/train_epoch', acc, epoch)
    tensorboard_writer.flush()
    model.eval()
    # 开始做模型的评估了
    right_count, tot_count = 0, 0
    eval_acc = 0.0
    with torch.no_grad():
        for step, val_epoch in enumerate(val_loader):
            img_Datas, labels = val_epoch  # train_epoch中的2个元素都是tensor格式的..
            img_Datas = img_Datas.to(device)
            labels = labels.to(device)
            model = model.to(device)
            output = model(img_Datas)

            predicted = torch.max(output, 1)
            right_count += (predicted[1].numpy() == labels.numpy()).sum()
            tot_count += len(labels)
    val_acc = right_count / tot_count
    print(f'this eval acc:{val_acc}')
    if val_acc > eval_acc:
        eval_acc = val_acc
        best_weights_path = './mnist_val_acc_%.3f.pt' % val_acc
        print(f'正在保存最优模型到:{best_weights_path}')
        torch.save(model.state_dict(), best_weights_path)
    else:
        print(f'当前验证集的准确率仅有:{val_acc},还不是目前的最有解,继续往下寻找....')

    print(f'epoch:{epoch},loss:{loss}')
    tensorboard_writer.close()
