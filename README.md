# Generate House Number

**I will not put my own GAN network here until the deadline of this assignment, this repo just translates Yuliya's code from pytorch to tensorflow and provide images for you to train**

# Images

## Dir `raw_imgs`

30 house number images from Amazon & Google Images

## Dir `imgs`

30 32 * 32 RGB images, generated from dir `raw_images`

## Dir `gray_imgs`

30 32 * 32 Gray images, the same as dir `imgs` but in gray scales

## `regularize_image.py`

Script to generate `imgs` and `gray_imgs`

# Tensorflow Version

## Dir `tf_version`

Please read [README.md](./tf_version/README.md) in that directory for more detail about DCGAN in Tensorflow


# Reference
[Yuliya's code](https://github.com/YuliyaLab/AIclass/blob/master/L9_DCGAN_housenum_1.ipynb)