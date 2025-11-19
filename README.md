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


## Notes on color-space naming (legacy `AB` vs current `HSV`)

Historically this codebase used LAB color-space where the chroma channels were called `A` and `B` (hence variables like `AB_A` / `AB_B`). The repository was later extended to support HSV-based color transforms. That created a small naming mismatch between variable names and actual color channels.

Current conventions in the code:
- Dataset (`my_datasets/color_dataset.py`) returns separated HSV channels as keys: `V_A`, `S_A`, `H_A` and `V_B`, `S_B`, `H_B` (each tensor shaped `[B,1,H,W]`).
- `ColorCycleGANModel.set_input` concatenates these into a 3-channel tensor stored in `self.AB_A` and `self.AB_B` for backward compatibility with older code. IMPORTANT: in the HSV workflow these `AB_*` tensors actually contain channels in the order `[V, S, H]`.

Why this matters:
- Some variable names (like `AB_A`) are legacy and do not literally mean LAB A/B channels anymore — they are simply the model's input tensors. When reading or modifying code, treat `AB_A`/`AB_B` as "input tensors"; check the channel order before applying channel-wise operations.

Recommendations:
- Prefer referring to `self.AB_A` as `self.input_A` or `self.HSV_A` in new code to avoid confusion (aliases are safe to add in model code).
- Ensure `opt.input_nc` and `opt.output_nc` are set to `3` when using HSV. If loading checkpoints trained with different `input_nc`, take care to adapt or retrain.
- Confirm tensor ranges: `rgb_to_hsv_tensor` returns values normalized to `[0,1]`. If your model or checkpoints expect `[-1,1]`, you must normalize accordingly.

If you'd like, we can (a) rename variables across the codebase for clarity, or (b) keep legacy names but document them (current approach). This README documents the current state.

