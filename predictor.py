# PROGRAMMER: Mario Duran
# DATE CREATED: 08/29/2018
# SAMPLE USAGE: predict.py "flowers/train/2/image_05087.jpg" "chkpnt.pth" --top_k 5 --category_names "cat_to_name.json" --gpu

import argparse
from PIL import Image
import json
import numpy as np
import torch


def init():
    #PARAMETERS
    arguments = get_args()

    #VALIDATE GPU USAGE
    gpu_avail = torch.cuda.is_available()
    use_GPU = False
    if gpu_avail and arguments.gpu:
        use_GPU = True
        print("GPU set to true")
    else:
        print("GPU not available")
    
    #LOAD THE MODEL
    model, training_mapping = load_chkpoint(arguments.checkpoint)

    #LOAD THE CATEGORY TO NAMES
    cat_to_name = None
    if arguments.category_names:
        with open(arguments.category_names, 'r') as f:
            cat_to_name = json.load(f)

    #RUN THE PREDICTION
    predict(arguments.input_image,  model, training_mapping, cat_to_name, arguments.top_k, use_GPU)

    return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str, default=None, help="Input Image")
    parser.add_argument("checkpoint", type=str, default=None, help="Path to the Checkpoint Network")
    parser.add_argument("--category_names", type=str, default=None, help="The human readable labels to use")
    parser.add_argument("--top_k", type=int, default=1, help="Top K categories to display in the results")
    parser.add_argument("--gpu", action='store_true', help="Use GPU")

    return parser.parse_args()

def load_chkpoint(path):
    checkpoint = torch.load(path)
    ld_model = checkpoint['model']
    ld_model.load_state_dict(checkpoint['state_dict'])
    
    return ld_model, checkpoint['train_mapping']


def process_image(image):

    thumb_size = 256, 256
    img = Image.open(image)
    img.thumbnail(thumb_size)
    width, height = img.size
    n_size = 224
    
    left = (width - n_size)/2
    top = (height - n_size)/2
    right = (width + n_size)/2
    bottom = (height + n_size)/2

    cropped = img.crop((left, top, right, bottom))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = np.array(cropped)
    np_image = (np_image - np.min(np_image))/np.ptp(np_image) #NORMALIZE 0 to 1
    
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2,0,1))
    
    return np_image

def predict(image_path, model, training_mapping, cat_to_name, topk, use_gpu):

    #Invert the training mapping
    predict_set = {v: k for k, v in training_mapping.items()}

     #Get the input image       
    image_to_eval = torch.from_numpy(process_image(image_path)).unsqueeze_(0).float()
    
    if(use_gpu):
        model.to('cuda')
        image_to_eval = image_to_eval.to('cuda')
    
    model.train(False)
    model.eval()
    
    outputs = model(image_to_eval)
    predictions = torch.topk(outputs.data, topk)

    probabilitiess = np.array(predictions[0][0])
    classes = np.array(predictions[1][0])
    
    result = zip(classes, probabilitiess)

    for c, p in result:
        if cat_to_name:
            print("Class: {} -- Probs {}".format(cat_to_name[str(predict_set[c])], p))
        else:
            print("Class: {} -- Probs {}".format(predict_set[c], p))
    
    return None


if __name__ == "__main__":
    init()
