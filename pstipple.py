from numpy import *
from random import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from heapq import heappush,heappop
import numpy as np
import cv2 as cv



def build_heap(img):
	h=[]
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			heappush(h,(-calculate_priority(img[x,y,0]),(x,y),img[x,y,0]))
			M[(x,y)]=False
	return h


def calculate_priority(intensity):
	if abs(255-intensity)>abs(intensity):
		priority=255-intensity
	else:
		priority=intensity
	return priority



def edges(img):
	img = cv.imread('disp2.png',0)
	edges = cv.Canny(img,100,200)
	z = np.zeros((edges.shape[0],edges.shape[1],3))
	for x in range(edges.shape[0]):
		for y in range(edges.shape[1]):
			z[x,y,:]=edges[x,y]
	return z
	# plt.subplot(121),plt.imshow(img,cmap = 'gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	# plt.show()


def edge_heap(depth_map):
	eimg=edges(depth_map)
	eh=[]
	for x in range(eimg.shape[0]):
		for y in range(eimg.shape[1]):
			heappush(eh,(-eimg[x,y,0],(x,y),eimg[x,y,0]))
			M[(x,y)]=False
	return eh


def stipple_size(aimg,aloc):
	rmax=2
	rmin=1
	size=rmin+((rmax-rmin)*(255-aimg[aloc[0],aloc[1],0])/255)
	return size


def SAS(G0,G1,k,D,img):
	# edgeimg=edges(deep)
	aheap=build_heap(img)
	# aheap=edge_heap(deep)
	stipplelist={}
	while (aheap):
		pixel=heappop(aheap)
		P,loc,I=pixel[0],pixel[1],pixel[2]
		if P!=-calculate_priority(img[loc[0],loc[1],0]):
			heappush(aheap,(-calculate_priority(img[loc[0],loc[1],0]),loc,img[loc[0],loc[1],0]))
		else:
			if not M[loc]:
				R=stipple_size(img,loc)
				if I<=135:
					App=0
					stipplelist[loc]=R
				else:
					App=255
					# stipplelist[loc]=255
				errorxy=I-App
				error_diffusion(loc,errorxy,R,G0,G1,k,D,img)
				M[loc]=True
	return stipplelist


def error_diffusion(loc,errorxy,R,G0,G1,k,D,img):
	sxy=sh_ex(errorxy,R,G0,G1)
	# error0=(Astipple-Apixcel)*k
	error0=0
	# calculate_total weight
	for i in range(-k//2+1,k//2+1):
		for j in range(-k//2+1,k//2+1):
			if not M[(loc[0]+i,loc[1]+j)]:
				rmn=(i**2+j**2)**(0.5)
				wmn=calculate_weight(rmn)
				wtotal=wtotal+wmn12
	
	# modifty signle pixel's intensity
	for i in range(-k//2+1,k//2+1):
		for j in range(-k//2+1,k//2+1):
			if not M[(loc[0]+i,loc[1]+j)]:
				iimn=img[loc[0]+i,loc[1]+j,0]
				rmn=sqrt(i**2+j**2)
				wmn=calculate_weight(rmn)
				Nwmn=wmn/wtotal
				iimn=iimn+Nwmn*(errorxy+error0)*sxy
				iimn=max(0,min(iimn,255))
				img[loc[0]+i,loc[1]+j,0]=(iimn,iimn,iimn)

	

def calculate_weight(rmn):
	if errorxy>0:
		wmn=img[loc[0]+i,loc[1]+j,0]/(rmn**2)
	else:
		wmn=(255-img[loc[0]+i,loc[1]+j,0])/(rmn**2)
	return wmn


def sh_ex(errorxy,Rsize,Gamma0,Gamma1):
	if errorxy<0:
		sxy=(1/Rsize)**Gamma0
	else:
		sxy=(Rsize)**Gamma1
	return sxy










  
if __name__== "__main__":
	# man = mpimg.imread('man.png')
	man = mpimg.imread('b2.png')
	deep = mpimg.imread('disp2.png')
	man255=man*255
	# nrow = man.shape[0]
	# ncolumn = man.shape[2]	
	# print(ncolumn)
	# print(vase)
	M={}
	# stippleimg=SAS(5,5,0,7,man255)
	stippleimg=SAS(10,10,0,15,man255)
	new_resolution=np.ones((man255.shape[0]*5,man255.shape[1]*5,3))*255
	for key, R in stippleimg.items():
		a,b=key[0],key[1]
		new_resolution[a*5,b*5,:]=(0,0,0);
		for i in range(-int(R*3)//2+1,int(R*3)//2+1):
			for j in range(-int(R*3)//2+1,int(R*3)//2+1):
				new_resolution[min(new_resolution.shape[0]-1,a*5+i),min(new_resolution.shape[1]-1,b*5+j),:]=(0,0,0)
	# for x in range(man255.shape[0]):
	# 		for y in range(man255.shape[1]):
	# 			man255[x,y]=(stippleimg[(x,y)],stippleimg[(x,y)],stippleimg[(x,y)])
	
	# for x in range(man255.shape[0]):
	# 	for y in range(man255.shape[1]):
	# 		new_resolution[x*5,y*5,:]=man255[x,y,:]
	# 		for i in range(3):
	# 			for j in range(3):
	# 				new_resolution[x*5+i,y*5+j,:]=man255[x,y,:]
	# 		new_resolution[x*5+1,y*5,:]=man255[x,y,:]
	# 		new_resolution[x*5,y*5+1,:]=man255[x,y,:]
	# 		new_resolution[x*5-1,y*5,:]=man255[x,y,:]
	# 		new_resolution[x*5,y*5-1,:]=man255[x,y,:]
	# print(new_resolution.shape)
	# image_resized = resize(image, (image.shape[0] / 4, image.shape[1] / 4),
 #                       anti_aliasing=True)
	# plt.axis("off")
	plt.imshow(new_resolution/255)
	plt.show()
	mpimg.imsave('new_resolution22', new_resolution/255)
	
		

