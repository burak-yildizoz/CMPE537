import cv2 as cv
import os

def get_quantizer(quantname, descriptor):
    if quantname == 'BOW':
        # https://docs.opencv.org/3.4.2/db/d39/classcv_1_1DescriptorMatcher.html
        matcher = cv.FlannBasedMatcher()
        # https://docs.opencv.org/3.4.2/d2/d6b/classcv_1_1BOWImgDescriptorExtractor.html
        return cv.BOWImgDescriptorExtractor(descriptor, matcher)
    else:
        raise Exception('Invalid option')

def get_hist(img, descriptor, quantizer):
    kps, desc = descriptor.detectAndCompute(img, None)
    hist = quantizer.compute(img, kps, desc)
    return hist

def func_compute_histograms(impath, indices, descriptor, quantizer,
                            hists, last_taxon_id, quants):
    taxon_id, img_id = indices
    img = cv.imread(impath)
    assert img is not None
    hist = get_hist(img, descriptor, quantizer)
    if last_taxon_id[0] != taxon_id:
        last_taxon_id[0] = taxon_id
        quant = np.mean(hists, axis=0)
        quants.append(quant)
        del hists[:]
    if hist is not None:
        hists.append(hist)

def findDictionary(data, k, max_iter=10):
    # random initialization
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    # k-means ++ initialization
    """np.random.seed(42)
    centroids = [data[0]]

    for i in range(1, k):
        dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in data])
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        print(i)
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(data[i])"""

    ind = np.zeros((data.shape[0], 1))

    for iter_num in range(max_iter):

        for i in range(data.shape[0]):
            delta = norm((data[i, :] - centroids), axis=1)
            ind[i, :] = np.argmin(delta)

        for j in range(k):
            centroids[j, :] = np.mean(data[ind[:, 0] == j, :], axis=0)
        print(iter_num)
    return centroids


def featureQuantization(descriptor_list, centroids, num_photos, num_words):
    im_features = np.zeros((num_photos, num_words), "float32")
    for i in range(num_photos):
        words, distance = vq(descriptor_list[i], centroids)
        for w in words:
            im_features[i][w] += 1
    return im_features

def hists_in_dir(quantizer, features, directory, print_progress=False):
    imnames = os.listdir(directory)
    #assert len(features) == len(imnames), 'len(features) = %d, directory: %s, # of files: %d' % (len(features), directory, len(imnames))
    hists = []
    for i, imname in enumerate(imnames):
        kps, desc = features[i]
        if desc is None:
            continue
        if print_progress:
            print('[%d/%d] Reading %s' % (i + 1, len(imnames), imname))
        impath = os.path.join(directory, imname)
        img = cv.imread(impath)
        assert img is not None, 'Could not open image %s' % (impath)
        hist = quantizer.compute(img, kps, desc)
        hists.append(hist)
    return hists

if __name__ == '__main__':
    import imgops
    from descriptors import get_descriptor
    import numpy as np
    from timeit import default_timer as timer
    # parameters
    quantname = 'BOW'
    dictname = 'BOW'
    num_cluster = 100
    descname = 'SIFT'
    desc_per_img = 20
    # initialize objects
    vocab = imgops.load_vocab(dictname, num_cluster, descname, desc_per_img)
    descriptor = get_descriptor(descname)
    quantizer = get_quantizer(quantname, descriptor)
    quantizer.setVocabulary(vocab)
    # quantize descriptors
    start = timer()
    hists, last_taxon_id, quants = [], [1], []
    imgops.loop_images(func_compute_histograms,
                       (descriptor, quantizer, hists, last_taxon_id, quants))
    quants.append(np.mean(hists, axis=0))
    quants = np.vstack(quants)
    end = timer()
    # print and save the results
    print('Quantization took %.1f seconds' % (end - start))
    print('mean histograms:', quants.shape)
    name = 'quants_%s_%s_%d_%s_%d' % (quantname, dictname, num_cluster,
                                      descname, desc_per_img)
    imgops.save_array(name, quants)
    #quants = imgops.load_quants(quantname, dictname, num_cluster,
    #                            descname, desc_per_img)
