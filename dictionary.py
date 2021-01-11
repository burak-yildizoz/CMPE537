import cv2 as cv
import os
import numpy as np

def get_dictionary(dictname, K=21):
    if dictname == 'BOW':
        # https://docs.opencv.org/3.4.2/d4/d72/classcv_1_1BOWKMeansTrainer.html
        return cv.BOWKMeansTrainer(clusterCount=K)
    else:
        raise Exception('Invalid option')

def add_descriptors(dictionary, descriptor,
                    desc_per_img=20,
                    dict_path='Dataset/Caltech20/training/',
                    dry_run=False):
    if not dry_run:
        add_descriptors(dictionary, descriptor, desc_per_img, dict_path, True)
    taxons = os.listdir(dict_path)
    for i_t, taxon in enumerate(taxons, start=1):
        taxon_path = os.path.join(dict_path, taxon)
        imnames = os.listdir(taxon_path)
        for i_i, imname in enumerate(imnames, start=1):
            impath = os.path.join(taxon_path, imname)
            assert os.path.isfile(impath), 'Could not find %s' % (impath)
            if dry_run: continue
            img = cv.imread(impath)
            assert img is not None, 'Could not read %s %s' % (taxon, imname)
            print('[%d/%d] [%d/%d] %s %s' %
                  (i_t, len(taxons), i_i, len(imnames), taxon, imname))
            _, desc = descriptor.detectAndCompute(img, None)
            if desc is None: continue
            np.random.shuffle(desc)
            desc = desc[0:desc_per_img, :]
            dictionary.add(desc)
        if dry_run: print('%s has %d images' % (taxon, len(imnames)))
    if dry_run: print('dry-run finished')

if __name__ == '__main__':
    from descriptors import get_descriptor
    from imgops import save_array
    from timeit import default_timer as timer
    # parameters
    dictname = 'BOW'
    descname = 'SIFT'
    desc_per_img = 20
    # collect descriptors
    dictionary = get_dictionary(dictname)
    descriptor = get_descriptor(descname)
    start = timer()
    add_descriptors(dictionary, descriptor, desc_per_img)
    end = timer()
    # print and save the results
    print('Collecting descriptors took %.1f seconds' % (end - start))
    print('Total number of descriptors:', dictionary.descriptorsCount())
    descs = np.vstack(dictionary.getDescriptors())
    save_array('descs_%s_%s_%d' % (dictname, descname, desc_per_img), descs)
    # obtain the vocabulary
    start = timer()
    vocab = dictionary.cluster(descs)
    end =  timer()
    # print and save the results
    print('Clustering took %.1f seconds' % (end - start))
    print('vocabulary:', vocab.shape)
    save_array('vocab_%s_%s_%d' % (dictname, descname, desc_per_img), vocab)
