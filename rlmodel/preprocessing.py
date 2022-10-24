import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt


class Prepocessing:

    image = []
    image_pp = []
    h,w = 0,0

    def __init__(self,image) :

        self.image = image
        self.h = image.shape[0]
        self.w = image.shape[1]

    def image_preprocess(self):    
        img = self.image
        new_img = img[:,:,0] * 0.3 + img[:,:,1] * 0.3 + img[:,:,2] * 0.3
        new_img = new_img[0:84,:]

        self.image_pp = new_img

    '''
    Return if car is in track
    '''
    def get_pos_car(self):
        img_pp = self.image_pp
        if img_pp[67:43] > 100 and img_pp[77:52] > 100:
            return 1
        return 0

    def plot_img(self,img):
        np.savetxt('img.txt',self.image_pp)
        img = im.fromarray(img)
        plt.imshow(img)
        plt.show()