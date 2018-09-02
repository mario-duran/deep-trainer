# deep-trainer
A generic deep learning training script using python and pytorch. This is a modifyed generic version of the final project requirements of the AI Programing with Python Nanodegree by Udacity

## Dependencies
- Pytorch [Docs](https://pytorch.org/docs/stable/index.html)
- Numpy [Docs](https://docs.scipy.org/doc/)

## Usage
The basic usage is as follows:
trainer.py "./path_to_data" --save_dir "./path_to_save/checkpoint.pth" --arch vgg16 -- --epochs 5 --learning_rate 0.0001 --hidden_units 512  --gpu --batch_size 32

## Next version
I'm planning to include a parameter to allow users chose optimizer

