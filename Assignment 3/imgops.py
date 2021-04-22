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

def loop_images(func,   # function to be called on each image
                params, # func(impath, (taxon_id, img_id), *params)
                taxons_path='Dataset/Caltech20/training/',
                dry_run=True):
    taxons = os.listdir(taxons_path)
    for i_t, taxon in enumerate(taxons, start=1):
        taxon_path = os.path.join(taxons_path, taxon)
        imnames = os.listdir(taxon_path)
        for i_i, imname in enumerate(imnames, start=1):
            impath = os.path.join(taxon_path, imname)
            assert os.path.isfile(impath), 'Could not find %s' % (impath)
            if dry_run:
                continue
            print('[%d/%d] [%d/%d] %s %s' %
                  (i_t, len(taxons), i_i, len(imnames), taxon, imname))
            func(impath, (i_t, i_i), *params)
        if dry_run:
            print('%s has %d images' % (taxon, len(imnames)))
    if dry_run:
        print('dry-run finished')
        loop_images(func, params, taxons_path, dry_run=False)

NPYDIR = 'Data/'

def save_array(name, array):
    os.makedirs(NPYDIR, exist_ok=True)
    path = NPYDIR + name + '.npy'
    np.save(path, array)
    print('Saved to', path)

def load_descs(descname, desc_per_img):
    name = 'descs_%s_%d' % (descname, desc_per_img)
    arr = np.load(NPYDIR + name + '.npy')
    assert arr.size != 0, 'Could not find %s' % (name)
    return arr

def load_vocab(dictname, num_cluster, descname, desc_per_img):
    name = 'vocab_%s_%d_%s_%d' % (dictname, num_cluster, descname, desc_per_img)
    arr = np.load(NPYDIR + name + '.npy')
    assert arr.size != 0, 'Could not find %s' % (name)
    assert arr.shape[0] == num_cluster, 'Cluster size should be %d whereas it is %d' % (num_cluster, arr.shape[0])
    return arr

def load_quants(quantname, dictname, num_cluster, descname, desc_per_img):
    name = 'quants_%s_%s_%d_%s_%d' % (quantname, dictname, num_cluster, descname, desc_per_img)
    arr = np.load(NPYDIR + name + '.npy')
    assert arr.size != 0, 'Could not find %s' % (name)
    assert arr.shape[1] == num_cluster, 'Cluster size should be %d whereas it is %d' % (num_cluster, arr.shape[1])
    return arr
