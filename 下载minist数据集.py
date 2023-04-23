import os.path
import shutil
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import matplotlib
from PIL import Image

path = './minist_data/'

trainData = torchvision.datasets.MNIST(path, train=True, download=False)
testData = torchvision.datasets.MNIST(path, train=False, download=False)


def save_image(image, filename):
    # Save the image
    image.save(filename)


def convert_to_image(tensor, filename):
    # Convert tensor to PIL image
    # image = transforms.ToPILImage()(tensor)

    # Save the image
    save_image(tensor, filename)


# ----------------- train ----------------------#
train_corpus_dir = r'./minist_data/MNIST/train'
if os.path.exists(train_corpus_dir):
    shutil.rmtree(train_corpus_dir)
os.mkdir(train_corpus_dir)
# ------------------ test ---------------------- #
test_corpus_dir = f'./minist_data/MNIST/test'
if os.path.exists(test_corpus_dir):
    shutil.rmtree(test_corpus_dir)
os.mkdir(test_corpus_dir)

# Convert train dataset to images
for i in range(len(trainData)):
    detail_train_img_path = os.path.join(train_corpus_dir, str(i) + '.jpg')
    detail_train_label_path = os.path.join(train_corpus_dir, str(i) + '.txt')
    convert_to_image(trainData[i][0], detail_train_img_path)
    label = int(trainData.targets[i].data.numpy())
    with open(detail_train_label_path, 'w', encoding='utf-8') as writer:
        writer.write(str(label))

# Convert test dataset to images
for i in range(len(testData)):
    detail_test_img_path = os.path.join(test_corpus_dir, str(i) + '.jpg')
    detail_test_label_path = os.path.join(test_corpus_dir, str(i) + '.txt')
    convert_to_image(testData[i][0], detail_test_img_path)
    label = int(testData.targets[i].data.numpy())
    with open(detail_test_label_path, 'w', encoding='utf-8') as writer:
        writer.write(str(label))
