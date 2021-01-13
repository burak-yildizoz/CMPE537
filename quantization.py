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
