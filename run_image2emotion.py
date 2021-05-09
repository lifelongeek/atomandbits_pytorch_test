import warnings
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]
img_dim=224 # where can i find?

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']
"""
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    
    return np.argmax(memory_available)
"""

# set transformation
resample_method = Image.LANCZOS
normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
img_transform = transforms.Compose([transforms.Resize((img_dim, img_dim), resample_method),
                                                  transforms.ToTensor(),
                                                  normalize])

# get image data
img_path='example_images/img2.png' # img0, img1, img2, img3
img = Image.open(img_path)
img = img.convert('RGB')
img = img_transform(img)

# get model
pretrained_model_path='image2emotion_pretrained.pt'
#gpuIdx = get_freer_gpu()
#device = torch.device("cuda:" + str(gpuIdx))
device = torch.device("cpu")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.load(pretrained_model_path, map_location=device) # image to emotion classifier
    model.eval()

# image
# forward
with torch.no_grad():
    img = img.unsqueeze(0) # insert batch dimension
    img = img.to(device)
    pred = model(img)
    pred = pred.exp() # logsoftmax to softmax
    pred = pred.squeeze()


# print info
pred_info = {ARTEMIS_EMOTIONS[l]:pred[l].item() for l in range(pred.size(-1))}
print(pred_info)

best_prediction_idx = pred.argmax().item()
print('highest emotion = ' + str(ARTEMIS_EMOTIONS[best_prediction_idx]))