import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(r'codes\photos\noise2.jpg')
b,g,r = cv2.split(img)           # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb

# Denoising
dst = cv2.fastNlMeansDenoisingColored(img,None,15,10,7,21)

b,g,r = cv2.split(dst)           # get b,g,r
rgb_dst = cv2.merge([r,g,b])     # switch it to rgb

#cv2.imshow(rgb_img)
#plt.show()

#plt.subplot(212),plt.imshow(rgb_dst)
plt.imshow(rgb_dst)
plt.show()
