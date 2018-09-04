# deep-trainer
A generic deep learning training script (trainer.py) using python and pytorch. This is a modifyed generic version of the final project requirements of the AI Programing with Python Nanodegree by Udacity

I also included a predictor script (predictor.py) which will load the trained network checkpoint and will run a prediction for any given input image.

## Dependencies
- Pytorch [Docs](https://pytorch.org/docs/stable/index.html)
- Numpy [Docs](https://docs.scipy.org/doc/)

## Usage
The basic usage is as follows:

_trainer.py "./path_to_data" --save_dir "./path_to_save/checkpoint.pth" --arch vgg16 --epochs 5 --learning_rate 0.0001 --hidden_units 512  --gpu --batch_size 32_

### Parameters
- (string) Path where the data is located
- --save_dir: (string) If present will save the checkpoint after training is completed.
- --arch: (string) Allows to specify the pre-trained Network to use (vgg16, alexnet and densenet), it will use vgg16 by default.
- --epochs: (int) Number of epochs to use in the training, default is 2.
- --learning_rate: (float) Learning rate value, default is 0.001
- --hidden_units: (int array) You can specify the number of Hidden layers and its output value, the script will generate a classifier based on those values
- --gpu: If present it will try to use CUDA
- --batch_size: the batch size when loading the data. Default is 64

---
predictor.py "./path_to_image/image_file" "./path_to_load/checkpoint.pth" --top_k 5 --category_names "./path_to_json/json_file" --gpu 

### Parameters
- (String) Path to load the image to use in the predictos
- (string) Path to load the Network checkpoint
- --top_k: (int) Number of top X categories to display on the prediction results.
- --category_names: (string) Path to the JSON file containin the labels of the different categories.
- --gpu: If present it will try to use CUDA to load and use the model


## Next Iteration (To Do)
I'm planning to include a parameter to allow users chose optimizer

