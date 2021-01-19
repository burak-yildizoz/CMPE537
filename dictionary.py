import cv2 as cv
import os
import numpy as np

def get_dictionary(dictname, num_cluster):
    if dictname == 'BOW':
        # https://docs.opencv.org/3.4.2/d4/d72/classcv_1_1BOWKMeansTrainer.html
        return cv.BOWKMeansTrainer(clusterCount=num_cluster)
    else:
        raise Exception('Invalid option')

def func_add_descriptors(impath, indices, dictionary, descriptor, desc_per_img=20):
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
    num_cluster = 100
    descname = 'SIFT'
    desc_per_img = 20
    # initialize objects
    dictionary = get_dictionary(dictname, num_cluster)
    descriptor = get_descriptor(descname)
    # collect descriptors
    # descriptors should be independent from dictionary
    start = timer()
    imgops.loop_images(func_add_descriptors, (dictionary, descriptor, desc_per_img))
    descs = np.vstack(dictionary.getDescriptors())
    end = timer()
    # print and save the results
    print('Collecting descriptors took %.1f seconds' % (end - start))
    print('descriptors:', descs.shape)
    name = 'descs_%s_%d' % (descname, desc_per_img)
    imgops.save_array(name, descs)
    # obtain the vocabulary
    #descs = imgops.load_descs(descname, desc_per_img)
    start = timer()
    vocab = dictionary.cluster(descs)
    end =  timer()
    # print and save the results
    print('Clustering took %.1f seconds' % (end - start))
    print('vocabulary:', vocab.shape)
    name = 'vocab_%s_%d_%s_%d' % (dictname, num_cluster, descname, desc_per_img)
    imgops.save_array(name, vocab)
    #vocab = imgops.load_vocab(dictname, num_cluster, descname, desc_per_img)
