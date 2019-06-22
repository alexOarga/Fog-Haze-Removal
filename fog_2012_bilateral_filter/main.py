# File: main.py
# Author: Alex Oarga <alexandru.oargahateg@edu.unito.it>
# Last Modified Date: 13.06.2019


import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import collections


input_image = '../images/ex.png'


def get_neighbors(img, x, y, n):
	XMAX = len(img[0])
	YMAX = len(img)
	x_min = 0 if x - n < 0 else x - n
	y_min = 0 if y - n < 0 else y - n
	x_max = XMAX - 1 if x + n >= XMAX else x + n
	y_max = YMAX - 1 if y + n >= YMAX else y + n
	return img[x_min:x_max, y_min:y_max]
	
# This function needs optimization
def C(img, m):
	XMAX = len(img[0])
	YMAX = len(img)
	lm = np.copy( img ).astype(np.float)
	lv = np.copy( img ).astype(np.float)
	c = np.copy( img ).astype(np.float)
	r =  1 / ((2*m+1)**2)
	for y in range( len(img[0]) ):
		for x in range( len(img) ):
			ng = get_neighbors(img, x, y, m)
			lm[x][y] = r * np.array([ np.sum(ng[:,:,0]), np.sum(ng[:,:,1]), np.sum(ng[:,:,2]) ])
	r =  (1/((2*m+1)**2)) 
	for y in range( len(img[0]) ):
		for x in range( len(img) ):	
			ng = get_neighbors(img, x, y, m)
			ng = (ng - lm[x][y]) ** 2
			lv[x][y] = np.array([ np.sum(ng[:,:,0]), np.sum(ng[:,:,1]), np.sum(ng[:,:,2]) ])
			c[x][y] = lv[x][y] / lm[x][y]
			if np.isnan(c[x][y][0]):
				c[x][y] = np.array([0,0,0])
	print("lm", lm[0][0])
	CL =  np.array([ np.sum(c[:,:,0]), np.sum(c[:,:,1]), np.sum(c[:,:,2]) ])
	CL = (1/(XMAX*YMAX)) * CL
	print(CL)
	return CL


def CGAIN(orig, enh, m):
	CGAIN = np.mean( np.abs(C(enh, m) - C(orig, m)))
	return CGAIN
	

def saturated(orin, enh):
	sai = 0
	sao = 0
	for y in range( len(orin[0]) ):
		for x in range( len(orin) ):	
			if  np.array_equal( orin[x][y], [255,255,255] ) or  np.array_equal( orin[x][y], [0,0,0]):
				sai = sai + 1
			if  np.array_equal( enh[x][y], [255,255,255]) or  np.array_equal( enh[x][y], [0,0,0]):
				sao = sao + 1
	return (sao - sai) / (len(orin[0])*len(orin))


# Read input image
img = cv2.imread(input_image)
orig = np.copy( img )
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape((450,956)).tolist()
# print(len(img))

#cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
#cv2.imshow('image1',img)

X = len(img[0])
Y = len(img)
if isinstance( img[0][0], np.ndarray ): 
	channels = len( img[0][0] )
else:
	channels = 1

print(X, Y, channels)


# Preprocessing
if channels == 1:
	img = cv2.equalizeHist(img)
else:
	img = cv2.addWeighted( img, 1.2, img, 0, 0)
	cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
	cv2.imshow('image1',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Normalize image
img = img/[255,255,255]


# Airlight init
K = 0.7
if channels == 1:
	A = []
	for pixel in img:
		A.append( pixel * K )
else:
	A = []
	for row in img:
		new_row = []
		for pixel in row:
			new_pixel = pixel.min()	* K
			new_row.append( new_pixel )
		A.append( new_row )

A = np.array( A ) 
A = A*255
A = A.astype(np.uint8)


# Airlight refinement
A = cv2.bilateralFilter(A,5,100,100)

blackA = 255 - A
cv2.namedWindow('filter', cv2.WINDOW_NORMAL)
cv2.imshow('filter',blackA)
cv2.waitKey(0)
cv2.destroyAllWindows()

A = A.astype(np.float)
A = A/255


# Restoration
if channels == 1:
	for y in range( len(img[0]) ):
		for x in range( len(img) ):
			pixel = img[x][y]
			air = A[x][y]
			img[x][y] = (pixel - air) if 1-air/1 != 0 else pixel-air
else:
	for y in range( len(img[0]) ):
		for x in range( len(img) ):
			[r, g ,b] = img[x][y]
			air = A[x][y]
			if 1 - air == 0:
				rn = 0
				gn = 0
				bn = 0
			else:
				rn = (r-air)/(1-air) if (r-air) > 0 else 0
				gn = (g-air)/(1-air) if (g-air) > 0 else 0
				bn = (b-air)/(1-air) if (b-air) > 0 else 0
			img[x][y] = [rn, gn, bn]

img = img*[255,255,255]
img = img.astype(np.uint8)


# Enhance constrast image
img = cv2.addWeighted( img, 1.2, img, 0, 0)


# Show image
cv2.namedWindow('after', cv2.WINDOW_NORMAL)
cv2.imshow('after',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Performance measures
#print("gain:", CGAIN(orig, img, 2) )
#print("saturated:", saturated(orig, img) )