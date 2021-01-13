import cv2 as cv
import os
import numpy as np

def get_dictionary(dictname, K=21):
    if dictname == 'BOW':
        # https://docs.opencv.org/3.4.2/d4/d72/classcv_1_1BOWKMeansTrainer.html
        return cv.BOWKMeansTrainer(clusterCount=K)
    else:
        raise Exception('Invalid option')

def add_descriptors(impath, indices, dictionary, descriptor, desc_per_img=20):
    img = cv.imread(impath)
    assert img is not None
    _, desc = descriptor.detectAndCompute(img, None)
    if desc is None:
        return
    np.random.shuffle(desc)
    desc = desc[0:desc_per_img, :]
    dictionary.add(desc)

if __name__ == '__main__':
    from descriptors import get_descriptor
    import imgops
    from timeit import default_timer as timer
    # parameters
    dictname = 'BOW'
    descname = 'SIFT'
    desc_per_img = 20
    # collect descriptors
    dictionary = get_dictionary(dictname)
    descriptor = get_descriptor(descname)
    start = timer()
    imgops.loop_images(add_descriptors, (dictionary, descriptor, desc_per_img))
    end = timer()
    # print and save the results
    print('Collecting descriptors took %.1f seconds' % (end - start))
    print('Total number of descriptors:', dictionary.descriptorsCount())
    descs = np.vstack(dictionary.getDescriptors())
    name = 'descs_%s_%s_%d' % (dictname, descname, desc_per_img)
    imgops.save_array(name, descs)
    # obtain the vocabulary
    start = timer()
    vocab = dictionary.cluster(descs)
    end =  timer()
    # print and save the results
    print('Clustering took %.1f seconds' % (end - start))
    print('vocabulary:', vocab.shape)
    name = 'vocab_%s_%s_%d' % (dictname, descname, desc_per_img)
    imgops.save_array(name, vocab)
