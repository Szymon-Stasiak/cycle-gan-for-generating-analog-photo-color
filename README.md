# Color CycleGAN for Digital-to-Analog Image Translation

This repository contains the **main part of the project** for our academic assignment in the **Computer Vision** course. Our project group is developing a pipeline to translate **digital images into analog-style images**, focusing specifically on **color transformation**.  

For the full project, see: [Analog Photo Generator](https://github.com/TMPkl/analog_photo_generator)

In this repository, we focus on **color processing** using a **CycleGAN** model with a modification that allows **exclusive color transformation**, without altering other aspects of the image.

---


## Features

- **Unaligned Dataset Handling**: Supports unpaired images from two domains: digital (Domain A) and analog (Domain B).  
- **Color-Only Translation**: Uses a modified CycleGAN that focuses solely on changing colors using the Lab color space.  
- **PyTorch Implementation**: Fully implemented with PyTorch for training and inference.  
- **Flexible Image Size**: Resize images to any desired resolution (default 256x256).  

> **Note**: `base_model.py` and `network.py` are from the official PyTorch CycleGAN implementation: [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

---

## To Get Started

to train the model use 
```bash
train_color_cyclegan.py 
```

To test the model use 
```bash
model_usage.py 
```

