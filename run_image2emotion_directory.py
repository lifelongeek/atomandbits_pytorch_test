import warnings
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]
img_dim=224

ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']

# set transformation
resample_method = Image.LANCZOS
normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
img_transform = transforms.Compose([transforms.Resize((img_dim, img_dim), resample_method),
                                                  transforms.ToTensor(),
                                                  normalize])
##for x in range(4):
    # get image data
    ##img_path='example_images/img'+str(x)+'.png' # img0, img1, img2, img3

file_list = os.listdir('example_images')
file_list = [filename for filename in file_list if filename.endswith('png') or filename.endswith('jpg')] # filtering

prob_array = np.zeros((len(file_list), len(ARTEMIS_EMOTIONS))) # will be saved to file

for i in range(len(file_list)):
    img_path='example_images/' + file_list[i]
    img = Image.open(img_path)
    img = img.convert('RGB')

    img = img_transform(img)

    # get model
    pretrained_model_path='image2emotion_pretrained.pt'
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
    send_prediction_idx = torch.topk(input=pred, k=2)
    print('Highest emotion = ' + str(ARTEMIS_EMOTIONS[send_prediction_idx[1][0].item()])+'('+str(send_prediction_idx[0][0].item())+')')
    print('Second emotion = ' + str(ARTEMIS_EMOTIONS[send_prediction_idx[1][1].item()])+'('+str(send_prediction_idx[0][1].item())+')')
    
    # append pred_info
    prob_array[i, :] = pred.squeeze().numpy()

# convert numpy to dataframe
dataframe = pd.DataFrame(prob_array, columns = ARTEMIS_EMOTIONS)

# write to file
dataframe.to_csv("multiimg_serial.csv", mode='w')