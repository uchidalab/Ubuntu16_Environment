# coding: utf-8
import cv2
print(cv2.__version__)
print(cv2.__file__)
# contribの機能が使えるか
sift = cv2.xfeatures2d.SIFT_create()

import matplotlib.pyplot as plt
img = cv2.imread('lenna.png')
show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(show_img)
plt.show()