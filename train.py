#!/usr/bin/env python
#coding=utf-8
from beta_TCVAE import beta_TCVAE
import numpy as np
import gc
import os
from PIL import Image
from scipy import misc
from tqdm import tqdm

beta_TCVAE = beta_TCVAE(training=True)
test_head = Image.open('test_head.png')
test_right = Image.open('test_right.png')
test_left = Image.open('test_left.png')
test_head = np.array(test_head)
test_right = np.array(test_right)
test_left = np.array(test_left)

_ = os.system("clear")

data_id = [0, 5, 10, 15, 20, 25, 30, 35]
epoch = [200, 150, 100, 100, 100, 50, 50]
lr = [0.0007, 0.0003, 0.0003, 0.0001, 0.0001, 0.00005, 0.00005]
# beta_TCVAE.load()

def main():
    for i in tqdm(range(7)):
        beta_TCVAE.lr = lr[i]
        beta_TCVAE.load_data(data_id=data_id[i])
        beta_TCVAE.fit(epoch=epoch[i])
        beta_TCVAE.Save()
        test = beta_TCVAE.image_test(test_head, test_right, test_left)
        misc.toimage(test, cmin=0, cmax=255).save('test_image/test_'+str(i)+'.png')
        del beta_TCVAE.dataset
        gc.collect()

if __name__ == '__main__':
    main()