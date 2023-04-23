import torch
from dataset import My_mnist_dataset
from model import Net
from torch.utils.data import DataLoader
import torch.nn.functional as F
batch_size = 32
testset = My_mnist_dataset(data_dir=r'D:\my_projects\shiyan\pytorch_lighting_explore\minist_data\MNIST\train', datatype="test")
val_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
model = Net()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('./mnist_val_acc_0.980.pt'))
model.eval()
with torch.no_grad():
    for step, test_data in enumerate(val_loader):
            model = model.to(device)
            # test_epoch = test_epoch.to(device)
            data, label = test_data
            data = data.to(device)
            out = model(data)
            predicted = torch.max(out, 1)
            # print(f'predicted:{predicted}')
            acc = (predicted[1].numpy() == label.numpy()).sum() / batch_size
            print(f'acc:{acc}')







