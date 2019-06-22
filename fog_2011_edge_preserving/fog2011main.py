# File: fog2011main.py
# Author: Alex Oarga <alexandru.oargahateg@edu.unito.it>
# Last Modified Date: 13.06.2019


input_image = '../../images/ex.png'
k = 0.8

import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import colorsys
import os
from fog2011edge import show_image, norm_show_image ,estimation_skylight, white_balance, atmospheric_veil_1, atmospheric_veil_2, recover_albedo


img_read = cv2.imread(input_image)
XMAX = len( img_read )
YMAX = len( img_read[0] )
show_image(img_read)
img = img_read
img = img[:,:,[2,1,0]]


# Skylight estimation
img2, aux = estimation_skylight(XMAX, YMAX, np.copy(img))
aux = np.array(aux)


# Normalizing
aux = aux/[255,255,255]
for elem in aux:
	if elem > 1:
		elem = 1
img = img/[255,255,255]


# WHite balance
img = white_balance(XMAX, YMAX, img, aux)
V = atmospheric_veil_1(XMAX, YMAX, img, 1)


#WSFILTER
V = V*[255,255,255]
cv2.imwrite('v.jpg', V)
print("Running .m")
os.system('./main.m')
print("Done running")
V = cv2.imread('v2.jpg')
V = V/[255,255,255]


# ADJUSTMENT
V = V*k
norm_show_image(1-V)


# Compute atmospheric veil
t = atmospheric_veil_2(XMAX, YMAX, img, V)


# Recovery
p = recover_albedo(XMAX, YMAX, img, V, t)
