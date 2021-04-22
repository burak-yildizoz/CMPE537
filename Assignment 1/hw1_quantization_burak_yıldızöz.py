import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import operator
import time

################################################################################
# Kmeans quantization                                                          #
################################################################################

def quantize(img, K, max_iter=10, tolerance=1, debug=True):
    x = random_points(img, K)
    #x = select_points(img, K)
    if len(set(p for p in x)) is not K:
        print('Seed coordinates must be different')
        print(x)
        return
    lab = cv.cvtColor(img, cv.COLOR_RGB2XYZ).astype(np.int16)
    state = np.array(list(lab[p] for p in x), float)
    if len(set(tuple(s) for s in state)) != len(state):
        print('Selected seed positions must have different color')
        print(list(img[p] for p in x))
        return
    last_state = np.copy(state)
    it = 0
    while True:
        it += 1
        L = list(np.linalg.norm(state[k] - lab, axis=2) for k in range(K))
        idx = np.argmin(L, axis=0)
        state = np.array(list(np.mean(lab[idx==k], axis=0) for k in range(K)))
        diff = np.linalg.norm(state - last_state)
        if debug:
            print('[it:%d] difference: %f' % (it, diff))
        if diff < tolerance:
            break
        if it >= max_iter:
            if not debug:
                print('Segmentation probably got stuck in a recursive loop')
                print('Difference:', diff)
                #print('Current state\n', state)
                #print('Last state\n', last_state)
            break
        last_state = np.copy(state)
    state = np.reshape(state.astype(np.uint8), (-1, 1, 3))
    return (idx, cv.cvtColor(state, cv.COLOR_XYZ2RGB))

################################################################################
# Choose seed points                                                           #
################################################################################

def select_points(img, K):
    plt.imshow(img)
    x = plt.ginput(K)
    plt.close()
    return list(tuple(map(int, (tup[1], tup[0]))) for tup in x)

def random_points(img, K):
    h, w, _ = img.shape
    x = np.random.uniform(low=[0,0], high=[h,w], size=(K,2))
    return list(tuple(map(int, arr)) for arr in x)

################################################################################
# Connected components                                                         #
################################################################################

def components(label_image, debug=False):
    labels = np.unique(label_image)
    all_results = []
    for label in labels:
        if debug:
            print('label:', label)
        # get binary image of the label and eliminate noise
        mask = label_image == label
        # find seperate components
        result = ()
        while mask.any():   # components found will be erased from mask
            comp_mask = np.zeros(mask.shape, bool)
            idx = np.argmax(mask != 0)  # first nonzero index
            pt = (idx // mask.shape[1], idx % mask.shape[1])
            comp_mask[pt] = True
            comp_mask = grow_region(comp_mask, mask)
            mask = mask & ~comp_mask
            # add to result
            area = cv.countNonZero(comp_mask.astype(np.uint8))
            if debug:
                print('There are', area, 'pixels at component with point', pt)
            result += ((pt, area),)
        if debug:
            print('Total', len(result), 'components')
        all_results += ((label, result),)
    return all_results

################################################################################
# Region growing                                                               #
################################################################################

def grow_region(mask, gt):
    last_mask = np.zeros(mask.shape, bool)
    # grow region until there is no change
    while np.logical_xor(mask, last_mask).any():
        # do not check points that are alreay checked
        prior_mask = np.copy(last_mask)
        last_mask = np.copy(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not prior_mask[i, j]:
                    # check 8-connected neighbors
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            # check boundaries
                            if ((i+dy >= 0) and (i+dy < mask.shape[0])
                                    and (j+dx >= 0)
                                    and (j+dx < mask.shape[1])):
                                if gt[i+dy, j+dx]:
                                    mask[i+dy, j+dx] = True
    return mask

################################################################################
# Miscellaneous                                                                #
################################################################################

def color_list(N, colormap=cv.COLORMAP_HSV):
    cmap = cv.applyColorMap(np.array(range(256), np.uint8), colormap)
    return list(tuple(int(c) for c in cmap[int(256*n/N)][0]) for n in range(N))

def mark_points(img, x, colormap=cv.COLORMAP_HSV):
    marked = np.copy(img)
    if colormap == 'invert':
        colors = list(tuple(int(256-v) for v in img[p]) for p in x)
    else:
        colors = color_list(len(x), colormap)
    for i in range(len(x)):
        marked = cv.drawMarker(marked, (x[i][1], x[i][0]), colors[i],
                               cv.MARKER_CROSS, markerSize=50, thickness=5)
    return marked

def paint_labels(idx, colorlist):
    img = np.zeros([idx.shape[0], idx.shape[1], 3], np.uint8)
    for i in range(len(colorlist)):
        img[idx == i] = colorlist[i]
    return img

################################################################################
# main Color Quantization                                                      #
################################################################################

if __name__ == '__main__':
    impath = 'photos/1.jpg'
    # read image
    img = cv.imread(impath, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # draw results
    plt.imshow(img)
    plt.title('Original Image')
    for K in [2, 4, 8, 16, 32]:
        print('K:', K)
        # get the label image and corresponding colors
        idx, state = quantize(img, K)
        # draw results
        plt.figure()
        plt.imshow(paint_labels(idx, state))
        plt.title('K-means with K=%d' % (K))
    # results are displayed here
    plt.show()

################################################################################
# main Connected Components                                                    #
################################################################################

if __name__ == '__main__CHANGE-THE-OTHER-ONE':
    impath = 'photos/birds1.jpg'
    threshold = 100
    # read image and apply threshold
    gray = cv.imread(impath, cv.IMREAD_GRAYSCALE)
    _, bw = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY_INV)
    # morphological operations
    kernel = np.ones((2, 2), np.uint8)
    bw = cv.morphologyEx(bw.astype(np.uint8), cv.MORPH_OPEN, kernel)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel).astype(bool)
    # draw results
    plt.set_cmap('gray')
    plt.imshow(gray)
    plt.title('Grayscale Image')
    plt.figure()
    plt.imshow(bw)
    plt.title('Thresholded Image')
    # find connected components
    comps = components(bw, debug=True)
    for label in comps:
        print('label:', label[0], 'number of components:', len(label[1]))
        for c in label[1]:
            print('There are', c[1], 'pixels at component with point', c[0])
    # draw results
    colorlist = color_list(len(comps[1][1]))
    drawn = np.zeros([bw.shape[0], bw.shape[1], 3], np.uint8)
    for i in range(len(colorlist)):
        pt = comps[1][1][i][0]
        mask = np.zeros(bw.shape, bool)
        mask[pt] = True
        mask = grow_region(mask, bw)
        drawn[mask] = colorlist[i]
    plt.figure()
    plt.imshow(drawn)
    plt.title('Connected Components: %d' % (len(comps[1][1])))
    # results are displayed here
    plt.show()
