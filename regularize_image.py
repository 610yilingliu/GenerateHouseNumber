import cv2
import os
import numpy as np

def regularize_img(path, dest, size):
    """
    :type path: String, image path
    :type path: String, destination path
    :type size: Tuple, (height, width)
    """
    img = cv2.imread(path)
    h = img.shape[0]
    l = img.shape[1]
    # If length: height is too large, it will cause deformation while resize. My solution is to put it over a white background for regularization
    # For height:length is the same.
    if l > 1.3 * h:
        bkg_size = l
        # 255 is used to generate white background. np.zeros will generate black background. 
        # If you still confused with it, just notice that (0, 0, 0) is black and (255, 255, 255) is white
        blank_img = np.zeros((bkg_size, bkg_size, 3)) + 255
        start = (l - h)//2
        blank_img[start:start + h, :] = img[:, :]
        img = blank_img
    elif h > 1.3 * l:
        bkg_size = h
        blank_img = np.zeros((bkg_size, bkg_size, 3)) + 255
        start = (h - l) // 2
        blank_img[:, start:start + l] = img[:, :]
        img = blank_img
    res = cv2.resize(img, size)
    cv2.imwrite(dest, res)

def make_gray(path, dest):
    """
    :type path: String, image path
    :type path: String, destination path
    """
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imwrite(dest, img_gray)

if __name__ == '__main__':
    root = './raw_imgs/'
    out = './imgs/'

    files = os.listdir(root)
    for f in files:
        path = root + f
        regularize_img(path, out + f, (32, 32))

    outgray = './img_gray/'
    files = os.listdir(out)
    for f in files:
        path = out + f
        make_gray(path, outgray + f)
