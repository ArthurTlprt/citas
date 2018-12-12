from glob import glob
import cv2
import numpy as np
from multiprocessing import Pool

# for training set
# ori_path = '../../augmented_box_1/fluorescent/'
# new_path = '../../augmented_box_1/fluo_brighter/'

# for test set
ori_path = '../../dataset/box_1/test/fluorescent/'
new_path = '../../dataset/box_1/test/fluo_brighter/'

labels = ['clean/', 'tr4/']
alpha = 20
beta = 50

def process_image(im_path):
    image_name = im_path.split('/')[-1]
    im = cv2.imread(im_path)
    im = cv2.addWeighted(im, alpha, np.zeros(im.shape, im.dtype), 0, beta)
    cv2.imwrite(new_path+label+'1/'+image_name, im)

if __name__ == '__main__':

    for label in labels:
        images_path = glob(ori_path+label+'1/*')
        p = Pool(8)
        p.map(process_image, images_path)
