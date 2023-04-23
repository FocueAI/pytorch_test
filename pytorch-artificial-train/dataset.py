import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms as T
import numpy as np

# 建立自己数据的dataset类
class My_mnist_dataset(data.Dataset):
    def __init__(self, data_dir, datatype=None):
        '''
        这个是做分类数据集的, 文件家中放的  是一个 .jpg/.png 一个是.txt 是一一对应的
        :param data_dir: './train/'
        :param datatype: train/val/test
        '''
        self.datatype = datatype
        self.root = data_dir
        self.img_name_list = []
        self.support_img_format = ['.png', '.jpg']
        for no, file_name in enumerate(os.listdir(data_dir)):
            real_file_name, extend_name = os.path.splitext(file_name)
            if extend_name in self.support_img_format:
                if os.path.exists(os.path.join(data_dir, real_file_name + '.txt')):  # 只有标签存在,该数据才是有效的
                    self.img_name_list.append(file_name)
        # 图像数据处理+增强
        """
        ToTensor()(pil_img)--->一共做了3件事情
        1. 将pil的图像数据转换成张量. 
        2. channel变成第一维度
        3. 将原先的数据/255 变成小数的形式
        """
        self.transforms = T.Compose([T.ToTensor()])
        if datatype == 'train':
            pass
        # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # if datatype == 'train':
        #     self.transforms = T.Compose(
        #         [T.Resize(256), T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        # else:
        #     self.transforms = T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor(),normalize])

    def __getitem__(self, item):
        """

        :param item: 图像序号,eg: 0,1,2,3,4,5,6,,
        :return: 对应序号的一张图片数据和对应的标签
        """
        pil_img = Image.open(os.path.join(self.root, self.img_name_list[item]))
        real_name, _ = os.path.splitext(self.img_name_list[item])
        with open(os.path.join(self.root, real_name + '.txt')) as reader:
            label_con = reader.readline()
        pil_img = self.transforms(pil_img)
        # if self.datatype == 'test':
        #     return pil_img
        return pil_img, int(label_con)
        # return pil_img.numpy(), int(label_con) # 这种格式也是可以的..

    def __len__(self):
        """
        返回有效数据的个数
        :return:
        """
        return len(self.img_name_list)


if __name__ == '__main__':
    import torchvision.transforms as transforms

    datasets = My_mnist_dataset(data_dir=r'D:\my_projects\shiyan\pytorch_lighting_explore\minist_data\MNIST\train')
    print(f'datasets:{datasets}')
    for dataset in datasets:
        pil_img_to_tensor, label = dataset
        print(f'pil_img:{type(pil_img_to_tensor)}') # < class 'torch.Tensor'>
        print(f'label:{type(label)}')     # < class 'int'>
        print('-=-=-=-=')
        print(f'shape:{pil_img_to_tensor.shape}')
        pil_img = transforms.ToPILImage()(pil_img_to_tensor)
        pil_img.show()
        # img_data = pil_img.numpy()
        # print(f'img_data.shape:{img_data.shape}')
        # img_data = (img_data*255).astype(np.uint8)
        # Image.fromarray(np.transpose(img_data,(1,2,0))).show()
