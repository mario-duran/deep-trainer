# deep-trainer
A generic deep learning training script using python and pytorch

## Dependencies
- Pytorch
- Numpy

## Usage
The basic usage is as follows:
trainer.py "./path_to_data" --save_dir "./path_to_save/checkpoint.pth" --arch vgg16 -- --epochs 5 --learning_rate 0.0001 --hidden_units 512  --gpu

## Next version
I'm planning to include a parameter to allow users chose optimizer

