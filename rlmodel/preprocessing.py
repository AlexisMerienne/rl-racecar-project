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
        self.new_w = 0
        self.new_h = 0

    def image_preprocess(self):    
        img = self.image
        new_img = img[:,:,0] * 0.3 + img[:,:,1] * 0.3 + img[:,:,2] * 0.3
        new_img = new_img[0:84,:]
        self.new_h,self.new_w = new_img.shape[0],new_img.shape[1]

        self.image_pp = new_img

    '''
    Return if car is in track
    '''
    def get_pos_car(self):
        img_pp = self.image_pp
        for i in range(self.new_h-1):
            for j in range(self.new_w-1):
                if img_pp[i,j]==0 and (img_pp[i][j-1] >100 or img_pp[i][j+1]>100 or img_pp[i][j-1] <10 or img_pp[i][j+1]<10):
                    return 1
        
        return 0

    def plot_img(self,img):
        np.savetxt('img.txt',self.image_pp)
        img = im.fromarray(img)
        plt.imshow(img)
        plt.show()