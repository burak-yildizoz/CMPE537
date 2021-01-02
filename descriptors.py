import cv2 as cv

def get_descriptor(descname):
    # https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html
    if descname == 'SIFT':
        return cv.xfeatures2d.SIFT_create()
    else:
        raise Exception('Invalid option')

if __name__ == '__main__':
    import imgops
    # read an image
    imname = 'Dataset/Caltech20/training/airplanes/image_0001.jpg'
    img = cv.imread(imname)
    assert img is not None
    # choose a descriptor and find keypoints
    descriptor = get_descriptor('SIFT')
    kps, desc = descriptor.detectAndCompute(img, None)
    # display the results
    print('descriptors:', desc.shape)
    print(desc.astype(int))
    disp_img = cv.drawKeypoints(img, kps, None)
    imgops.plt_imshow(disp_img)
