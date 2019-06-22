# File: main.py
# Author: Alex Oarga <alexandru.oargahateg@edu.unito.it>
# Last Modified Date: 13.06.2019


import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import colorsys
import math


k = 0.5


def show_image(img, img2=None, img2bool=False, name='aux'):
	if img2bool:
		cv2.namedWindow('image12', cv2.WINDOW_NORMAL)
		cv2.imshow('image12',img2)
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	#cv2.imshow('image1',img[:,:,[2,1,0]])
	cv2.imshow(name,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def norm_show_image(img):
	XMAX = len( img )
	YMAX = len( img[0] )
	img = np.copy(img)
	img = img*[255,255,255]
	img = img.astype(np.uint8)
	cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
	#cv2.imshow('image1',img[:,:,[2,1,0]])
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def restore_image(test, A, t, show):
	test = test/255

	out = ( test - A) / t
	XMAXO = len( out )
	YMAXO = len( out[0] )

	print("a",  A )
	print("t", t)

	for x in range(XMAXO):
		for y in range(YMAXO):
			[r,g,b] = out[x][y]
			[r,g,b] = [r*255,g*255,b*255]
			if r > 255:
				r = 255
			elif r < 0:
				r = 0
			if g > 255:
				g = 255
			elif g < 0:
				g = 0
			if b > 255:
				b = 255
			elif b < 0:
				b = 0
			out[x][y] = [r,g,b]

	out = out.astype(np.uint8)

	if show:
		show_image(out)


def same_depth(fog, clear, correction, show=False):
	J = clear/255
	XMAXJ = len( J )
	YMAXJ = len( J[0] )
	m = XMAXJ * YMAXJ

	I = fog/255
	XMAXI = len( I )
	YMAXI = len( I[0] )

	j_mean = np.array([np.mean(J[:,:,0]), np.mean(J[:,:,1]), np.mean(J[:,:,2])]) 
	i_mean = np.array([np.mean(I[:,:,0]), np.mean(I[:,:,1]), np.mean(I[:,:,2])]) 

	num = 0
	dem = 0
	sumxy = 0
	sumx = 0
	sumy = 0
	sumx2 = 0
	for x in range(XMAXJ):
		for y in range(YMAXJ):
			sumxy = sumxy + J[x][y]*I[x][y]
			sumx = sumx + J[x][y]
			sumy = sumy + I[x][y]
			sumx2 = sumx2 + (J[x][y]**2)
	A = ((m*sumxy)-(sumx*sumy)) / ((m*sumx2)-(sumx**2))
	A = A*correction

	t = (sumy - (A*sumx))/m

	test = cv2.imread('../images/middtest.jpg')

	restore_image(test, A, t, show)

	return A, t


def different_depth():
	# 1) Basic principles of different depth information
	fog = cv2.imread("../images/middfog.jpg")
	clear = cv2.imread("../images/midd.jpg")
	fog3 = fog[80:180, 200:300]
	cv2.imwrite('f3.jpg', fog3)
	clear3 = clear[80:180, 200:300]
	cv2.imwrite('c3.jpg', clear3)
	fog2 = fog[130:230, 300:400]
	cv2.imwrite('f2.jpg', fog2)
	clear2 = clear[130:230, 300:400]
	cv2.imwrite('c2.jpg', clear2)
	fog1 = fog[210:310, 300:400]
	cv2.imwrite('f1.jpg', fog1)	
	clear1 = clear[210:310, 300:400]
	cv2.imwrite('c1.jpg', clear1)

	A1, t1 = same_depth(fog1, clear1, 0.2)
	A3, t3 = same_depth(fog3, clear3, 0.2)

	# 2) Estimating the depth
	bt1 = -np.log(t1)
	bt3 = -np.log(t3)
	bt2 = bt1 + k*( bt3 - bt1 )
	
	t2 = np.exp( -bt2 )
	A2 = (A1+A3)/2
	
	restore_image(fog2, A2, t2, True)
	
# different depth
different_depth()

# Same depth
#J = cv2.imread('../images/midd.jpg')
#I = cv2.imread('../images/middfog.jpg')
#show_image(I)
#same_depth(I, J, 0.2, show=True)