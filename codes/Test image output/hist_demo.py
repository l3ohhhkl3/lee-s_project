import pylab as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# load image to numpy arrayb
# matplotlib 1.3.1 only supports png images
# use scipy or PIL for other formats
img = np.uint8(mpimg.imread(r'C:\Users\Akshay Patil\Desktop\DESKTOP\Test image output\testimg.jpg')*255.0)
img1=cv2.imread(r'C:\Users\Akshay Patil\Desktop\DESKTOP\Test image output\testimg.jpg')
cv2.imshow('original', img1)
# convert to grayscale
# do for individual channels R, G, B, A for nongrayscale images

img = np.uint8((0.2126* img[:,:,0]) + \
  		np.uint8(0.7152 * img[:,:,1]) +\
			 np.uint8(0.0722 * img[:,:,2]))

# use hist module from hist.py to perform histogram equalization
from hist import histeq
new_img, h, new_h, sk = histeq(img)

# show old and new image
# show original image
#plt.subplot(121)
plt.imshow(img)
plt.title('original image')

plt.set_cmap('gray')

#plt.show()
# show original image
#plt.subplot(122)
plt.imshow(new_img)
plt.title('hist. equalized image')
#plt.set_cmap('gray')
plt.show()
cv2.imshow('histogram', new_img)
c = imagem = cv2.bitwise_not(img1)
d = imagem = cv2.bitwise_not(c)
cv2.imshow('result', d)

# plot histograms and transfer function
fig = plt.figure()
fig.add_subplot(221)
plt.plot(h)
plt.title('Original histogram') # original histogram

fig.add_subplot(222)
plt.plot(new_h)
plt.title('New histogram') #hist of eqlualized image

fig.add_subplot(223)
plt.plot(sk)
plt.title('Transfer function') #transfer function

plt.show()