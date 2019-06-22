# File: fog2011edge.py
# Author: Alex Oarga <alexandru.oargahateg@edu.unito.it>
# Last Modified Date: 13.06.2019
# Description: 
#		Auxiliar file for fog2011main.py
#


import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import colorsys


def get_neighbors(XMAX, YMAX, img, x, y, n):
	XMAX = len( img )
	YMAX = len( img[0] )

	x_min = 0 if x - n < 0 else x - n
	y_min = 0 if y - n < 0 else y - n
	x_max = XMAX - 1 if x + n >= XMAX else x + n
	y_max = YMAX - 1 if y + n >= YMAX else y + n
	return img[x_min:x_max, y_min:y_max]


def show_image(img, img2=None, img2bool=False):
	if img2bool:
		cv2.namedWindow('image12', cv2.WINDOW_NORMAL)
		cv2.imshow('image12',img2)
	cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
	cv2.imshow('image1',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def norm_show_image(img):
	XMAX = len( img )
	YMAX = len( img[0] )
	img = np.copy(img)
	img = img*[255,255,255]
	img = img.astype(np.uint8)
	cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
	cv2.imshow('image1',img[:,:,[2,1,0]])
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def convert_hls_to_rgb(h, l, s):
	(h,l,s) = (h/255, l/255, s/255) 
	r, g, b = colorsys.hls_to_rgb(h, l, s)
	(r,g,b) = (r*255, g*255, b*255) 
	return (r,g,b) 


def estimation_skylight(XMAX, YMAX, img):
	# min filter
	kernel = np.ones((5,5), np.uint8)
	img = cv2.erode(img, kernel, iterations=1)
	
	# Canny operator
	edges = cv2.Canny(img,50,50)
	show_image(edges)

	# Nedge
	Nedge = np.copy(edges)
	Nedge = Nedge.astype(float)
	for x in range( XMAX ):
		for y in range( YMAX ):
			nhood = get_neighbors(XMAX, YMAX, edges, x, y, 3)
			unique, counts = np.unique(nhood, return_counts=True)
			dic = dict(zip(unique, counts))
			if 255 in dic:
				n_edge = dic[255]
			else:
				n_edge = 0
			size = len(nhood)*len(nhood[0])
			Nedge[x][y] = n_edge/size

	# get brightness
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	L = hls[:, :, 1]
	max_l = np.amax(L) * 0.95

	#check Nedge < 0.001 and brightness > max_L
	comp = np.copy(Nedge)
	for x in range( XMAX ):
		for y in range( YMAX ):
			if comp[x][y] < 0.001 and L[x][y] > max_l:
				comp[x][y] = 255
			else:
				comp[x][y] = 0
	show_image(comp)

	# find first connect component
	# TODO: change code to find connected components
	"""
		code found at: https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
			import cv2
			import numpy as np

			img = cv2.imread('eGaIy.jpg', 0)
			img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
			ret, labels = cv2.connectedComponents(img)

			def imshow_components(labels):
				# Map component labels to hue val
				label_hue = np.uint8(179*labels/np.max(labels))
				blank_ch = 255*np.ones_like(label_hue)
				labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

				# cvt to BGR for display
				labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

				# set bg label to black
				labeled_img[label_hue==0] = 0

				cv2.imshow('labeled.png', labeled_img)
				cv2.waitKey()

			imshow_components(labels)
	"""
	values = []
	for x in range( XMAX ):
		for y in range( YMAX ):
			if comp[x][y] == 255:
				values.append(hls[x][y])
	try:
		values = np.array(values)
		skylight = np.amax(values[:,1])
		for hls_pixel in values:
			if hls_pixel[1] == skylight:
				skylight = hls_pixel
				break
		(r,g,b) = convert_hls_to_rgb(skylight[0], skylight[1], skylight[2])
		print(r,g,b)	
		sky = np.array([r,g,b])
	except Exception:
		sky = np.array([255,255,255])
	return img, sky


def white_balance(XMAX, YMAX, img, skylight):
	img = img.astype(float)
	[rs, gs, bs] = skylight
	for x in range( XMAX ):
		for y in range( YMAX ):
			[r,g,b] = img[x][y]
			rn = min( r/rs, 1 )
			gn = min( g/gs, 1 )
			bn = min( b/bs, 1 )
			img[x][y] = [rn, gn, bn]
			#print(r, g, b, rs, gs, bs, r/rs, g/gs, b/bs)
	# TO show image multily each pixel by 255: [r,g,b] -> [r*255,g*255,b*255]
	#img = img.astype(np.uint8)
	norm_show_image(img)
	return img
	

def atmospheric_veil_1(XMAX, YMAX, img, k):
	# Coarse Estimation
	V = np.copy(img)
	for x in range( XMAX ):
		for y in range( YMAX ):
			new = img[x][y].min()
			V[x][y] = [new, new, new]
	V = V*k
	return V


def atmospheric_veil_2(XMAX, YMAX, img, V):	
	t = np.copy(img)
	for x in range( XMAX ):
		for y in range( YMAX ):
			[m1, m2, m3] = V[x][y]
			t[x][y] = [1-m1, 1-m2, 1-m3]
	return t


def recover_albedo(XMAX, YMAX, img, V, t):
	print("recover albedo")
	p = np.copy(img)
	p = p.astype(np.float)
	for x in range( XMAX ):
		for y in range( YMAX ):
			[r,g,b] = img[x][y]
			[vr, vg, vb] = V[x][y]
			[tr, tg, tb] = t[x][y]
			#print(img[x][y], V[x][y], t[x][y], VANT[x][y])
			if tr == 0 or (r-vr) < 0: 
				newr = 0
			else:
				newr = (r-vr)/tr if tr > 0 else 0
			if tg == 0 or (g-vg) < 0: 
				newg = 0
			else:
				newg = (g-vg)/tg if tg > 0 else 0
			if tb == 0 or (b-vb) < 0: 
				newb = 0
			else:
				newb = (b-vb)/tb if tb > 0 else 0
			#print(newr, newg, newb, tr, tg, tb)			
			p[x][y] = [newr, newg, newb]
	p = np.array(p)
	###
	norm_show_image(p)
	###
	return p
	

