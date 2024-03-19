# MRFS_CVPR2024
The code of MRFS: Mutually Reinforcing Image Fusion and Segmentation
<br>
#### Recommended Environment:<br>
 - [ ] python = 3.8
 - [ ] torch = 1.8.1+cu111
 - [ ] timm = 0.9.8
 - [ ] numpy = 1.24.4
 - [ ] scipy = 1.10.1
 - [ ] pillow = 10.1.0
 - [ ] tqdm = 4.66.1
 - [ ] tensorboardX = 2.6.2.2
 - [ ] opencv-python = 4.8.1.78

# Training:<br>
* Prepare training data & set the training parameters:<br>
  * Dataset fomula:
  * Dataset name<br>
    ------RGB folder<br>
    ------THE folder<br>
    ------Label folder<br>
    ------train.txt<br>
    ------val.txt<br>
    ------test.txt<br>
* Run ```CUDA_VISIBLE_DEVICES="GPU IDs" python -m torch.distributed.launch --nproc_per_node="GPU numbers you want to use" train.py```<br>
* Example ```CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py```<br>

# Test:<br>
* pretrained weights can be found at here: [FMB](https://drive.google.com/drive/folders/1rtrcdyI5OuEQl5VipDmitf4Z9M4MHBdO?usp=sharing)
* Set the testing parameters:<br>
* Run ```CUDA_VISIBLE_DEVICES="GPU IDs" python eval.py -e="epoch"```<br>
* Example ```CUDA_VISIBLE_DEVICES=0 python eval.py -e=MRFS```<br>
