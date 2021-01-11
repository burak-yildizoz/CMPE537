import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def cv_imshow_scale(winname, img):
    cv.namedWindow(winname, cv.WINDOW_KEEPRATIO)
    cv.imshow(winname, img)
    cv.waitKey()
    cv.destroyWindow(winname)

def plt_imshow(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

NPYDIR = 'Data/'
def save_array(name, array):
    os.makedirs(NPYDIR, exist_ok=True)
    path = NPYDIR + name + '.npy'
    np.save(path, array)
    print('Saved to', path)
