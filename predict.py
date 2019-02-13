import json
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import glob
import os, os.path
data_dir = 'data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

nThreads = 4
batch_size = 8
model_name = "object-detect.pth"
output_class=2
use_gpu = torch.cuda.is_available()
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = dict()
image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
image_datasets['test'] = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders

dataloaders = dict()
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size)
dataloaders['test']  = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location='cpu') ####################### CPU USE OR GPU torch.load(filepath)
    model = models.densenet161(pretrained=False)
    num_features = model.classifier.in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_features, 512)),
        ('relu', nn.ReLU()),
        ('drpot', nn.Dropout(p=0.5)),
        ('hidden', nn.Linear(512, 100)),
        ('fc2', nn.Linear(100, output_class)),
        ('output', nn.LogSoftmax(dim=1)),
    ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['class_to_idx']


# get index to class mapping
loaded_model, class_to_idx = load_checkpoint(model_name)
idx_to_class = {v: k for k, v in class_to_idx.items()}

loaded_model.eval()
images, labels = next(iter(dataloaders['test']))
output = loaded_model.forward(Variable(images[:2]))
ps = torch.exp(output).data
print(ps.max(1)[1])

#print ("Hello",predict('flower_data/test/2/image_05100.jpg', loaded_model))

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage / 255.

    imgA = npImage[:, :, 0]
    imgB = npImage[:, :, 1]
    imgC = npImage[:, :, 2]

    imgA = (imgA - 0.485) / (0.229)
    imgB = (imgB - 0.456) / (0.224)
    imgC = (imgC - 0.406) / (0.225)

    npImage[:, :, 0] = imgA
    npImage[:, :, 1] = imgB
    npImage[:, :, 2] = imgC

    npImage = np.transpose(npImage, (2, 0, 1))

    return npImage


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    image = torch.FloatTensor([process_image(Image.open(image_path))])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]

    top_idx = np.argsort(pobabilities)[-topk:][::-1]
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class
def view_classify(img, probabilities, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img_filename = img.split('/')[-2]
    img = Image.open(img)
    print(img_filename)
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 10), ncols=1, nrows=2)
    flower_name = mapper[img_filename]

    ax1.set_title(flower_name)
    ax1.imshow(img)
    ax1.axis('off')

    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in classes])
    ax2.invert_yaxis()
    plt.show()
#img1 = 'flower_data/image_01313.jpg'
img1 = 'images.jpg'
#img1 = 'data/test/1/image_00007.jpg'
def multiple_predict():
    imgs = []
    path = "data/test/1"
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(Image.open(os.path.join(path, f)))
        image_path = path + "/" + f
        print(image_path)
        #p, c = predict(str(image_path), loaded_model)
        #print(p,c)
def image_show(img_path,probabilities,classes, mapper):
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)
    plt.title(cat_to_name[classes[0]])
    plt.show()
#multiple_predict(loaded_model)
def main():
    img1 = 'data/test/1/image_00007.jpg'
    img2 = 'data/test/1/image_00004.jpg'
    img3 = 'data/test/2/image_00018.jpg'
    img4 = 'data/test/2/image_00010.jpg'
    img5 = 'images.jpg'
    p, c = predict(img1, loaded_model)
    p2, c2 = predict(img2, loaded_model)
    p3, c3 = predict(img3, loaded_model)
    p4, c4 = predict(img4, loaded_model)
    p5, c5 = predict(img5, loaded_model)
    image_show(img1, p, c, cat_to_name)
    image_show(img2, p2, c2, cat_to_name)
    image_show(img3, p3, c3, cat_to_name)
    image_show(img4, p4, c4, cat_to_name)
    image_show(img5, p5, c5, cat_to_name)
    multiple_predict()

if __name__ == '__main__':
    main()
