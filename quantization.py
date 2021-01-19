import cv2 as cv

def get_quantizer(method, descriptor):
    if method == 'BOW':
        # https://docs.opencv.org/3.4.2/db/d39/classcv_1_1DescriptorMatcher.html
        matcher = cv.FlannBasedMatcher()
        # https://docs.opencv.org/3.4.2/d2/d6b/classcv_1_1BOWImgDescriptorExtractor.html
        return cv.BOWImgDescriptorExtractor(descriptor, matcher)
    else:
        raise Exception('Invalid option')

def compute_histograms(impath, indices, descriptor, quantizer,
                       hists, last_taxon_id, quants):
    taxon_id, img_id = indices
    if img_id > 2: return
    img = cv.imread(impath)
    assert img is not None
    kps, desc = descriptor.detectAndCompute(img, None)
    hist = quantizer.compute(img, kps, desc)
    if last_taxon_id[0] != taxon_id:
        last_taxon_id[0] = taxon_id
        quant = np.mean(hists, axis=0)
        quants.append(quant)
        hists = []
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
    
  
if __name__ == '__main__':
    import imgops
    from descriptors import get_descriptor
    import numpy as np
    # parameters
    quantname = 'BOW'
    dictname = 'BOW'
    descname = 'SIFT'
    desc_per_img = 20
    # initialize objects
    vocab = imgops.load_array('vocab', dictname, descname, desc_per_img)
    descriptor = get_descriptor(descname)
    quantizer = get_quantizer(quantname, descriptor)
    # quantize descriptors
    quantizer.setVocabulary(vocab)
    hists, last_taxon_id, quants = [], [1], []
    imgops.loop_images(compute_histograms, (descriptor, quantizer,
                                            hists, last_taxon_id, quants))
    quants.append(np.mean(hists, axis=0))
    quants = np.vstack(quants)
    # print and save the results
    print('mean histograms:', quants.shape)
    name = 'quants_%s_%s_%s_%d' % (quantname, dictname, descname, desc_per_img)
    imgops.save_array(name, quants)
