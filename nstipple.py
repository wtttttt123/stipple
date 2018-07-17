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
			heappush(h,(-calculate_priority(img[x,y]),(x,y),img[x,y,0]))
			M[(x,y)]=False
	return h

def calculate_priority(pixel):
	if abs(255-pixel[0])>abs(pixel[0]):
		priority=255-pixel[0]
	else:
		priority=pixel[0]
	return priority


# def build_heap_depth1(img,edges):
# 	h=[]
# 	for x in range(img.shape[0]):
# 		for y in range(img.shape[1]):
# 			if edges[x,y]==255:
# 				heappush(h,(-300,(x,y),img[x,y,0]))
# 			else:
# 				heappush(h,(-calculate_priority_depth(img,x,y,edges),(x,y),img[x,y,0]))
# 			M[(x,y)]=False
# 	return h


def build_heap_depth(img,edge_list):
	h=[]
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			priority=calculate_priority_depth(img[x,y],(x,y),edge_list)
			heappush(h,(-priority,(x,y),img[x,y,0]))
			M[(x,y)]=False
	return h



def calculate_priority_depth(pixel,loc,edge_list):
	if loc in edge_list:
		priority=300		
	else:	
		if abs(255-pixel[0])>abs(pixel[0]):
			priority=255-pixel[0]
		else:
			priority=pixel[0]
	return priority


# def edges(img):
# 	edges = cv.Canny(img,100,200)
# 	z = np.zeros((edges.shape[0],edges.shape[1],3))
# 	for x in range(edges.shape[0]):
# 		for y in range(edges.shape[1]):
# 			z[x,y,:]=edges[x,y]
# 	return z
# 	plt.subplot(121),plt.imshow(img,cmap = 'gray')
# 	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# 	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# 	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# 	plt.show()


# def edge_heap(depth_map):
# 	eimg=edges(depth_map)
# 	eh=[]
# 	for x in range(eimg.shape[0]):
# 		for y in range(eimg.shape[1]):
# 			heappush(eh,(-eimg[x,y,0],(x,y),eimg[x,y,0]))
# 			M[(x,y)]=False
# 	return eh


def stipple_size(aimg,aloc):
	rmax=2
	rmin=1
	size=rmin+((rmax-rmin)*(255-aimg[aloc[0],aloc[1],0])/255)
	return size

# close smaller dot
def stipple_size_depth_small(adeep,aloc):
	rmax=2
	rmin=1
	size=rmin+((rmax-rmin)*(255-adeep[aloc[0],aloc[1]])/255)
	return size

# close bigger dot
def stipple_size_depth_big(adeep,aloc):
	rmax=2
	rmin=1
	size=rmax-((rmax-rmin)*(255-adeep[aloc[0],aloc[1]])/255)
	return size

# def stipple_size_depth_big_intensity(adeep,aloc):
# 	rmax=2
# 	rmin=1
# 	size=rmax-((rmax-rmin)*(255-adeep[aloc[0],aloc[1]])/255)
# 	return size

def SASD(G0,G1,k,D,img,deep):
	edge_list=[]
	edges=cv.Canny(deep,100,200)
	for x in range(edges.shape[0]):
		for y in range(edges.shape[1]):
			if edges[x,y]==255:
				edge_list.append((x,y))
	aheap=build_heap_depth(img,edge_list)
	stipplelist={}
	while (aheap):
		pixel=heappop(aheap)
		P,loc,I=pixel[0],pixel[1],pixel[2]
		if P!=-calculate_priority_depth(img[loc[0],loc[1]],loc,edge_list):
				heappush(aheap,(-calculate_priority_depth(img[loc[0],loc[1]],loc,edge_list),loc,img[loc[0],loc[1],0]))
		else:
			if not M[loc]:
				R=stipple_size_depth_big(deep,loc)
				R1=stipple_size(img,loc)
				if I<=128:
					App=0
					stipplelist[loc]=(R+R1)/2
				else:
					App=255
				errorxy=I-App
				error_diffusion(loc,errorxy,R,G0,G1,k,D,img)
				M[loc]=True
	return stipplelist


def SAS(G0,G1,k,D,img):
	aheap=build_heap(img)
	stipplelist={}
	while (aheap):
		pixel=heappop(aheap)
		P,loc,I=pixel[0],pixel[1],pixel[2]
		if P!=-calculate_priority(img[loc[0],loc[1]]):
			heappush(aheap,(-calculate_priority(img[loc[0],loc[1]]),loc,img[loc[0],loc[1],0]))
		else:
			if not M[loc]:
				R=stipple_size(img,loc)
				if I<=128:
					App=0
					stipplelist[loc]=R
				else:
					App=255
				errorxy=I-App
				error_diffusion(loc,errorxy,R,G0,G1,k,D,img)
				M[loc]=True
	return stipplelist



def error_diffusion(loc,errorxy,R,G0,G1,k,D,img):
	sxy=sh_ex(errorxy,R,G0,G1)
	wtotal=0
	Astipple=0
	for i in range(-int(R*dot),int(R*dot)+1):
		for j in range(-int(R*dot),int(R*dot)+1):
			if (i**2+j**2)**0.5<R*dot:
				Astipple+=1
	Apixcel=1
	error0=(Astipple-Apixcel)*k
	# error0=0
	# calculate_total weight
	for i in [x for x in range(-D//2+1,D//2+1)]:
		for j in [y for y in range(-D//2+1,D//2+1)]:
			if 0<=loc[0]+i<img.shape[0] and 0<=loc[1]+j<img.shape[1] and abs(i)+abs(j)!=0:
				if (not M[(loc[0]+i,loc[1]+j)]) and (i**2+j**2)**0.5<D/2:
					rmn=(i**2+j**2)**0.5
					wmn=calculate_weight(loc,img,errorxy,rmn,i,j)
					wtotal+=wmn
	
	# modifty signle pixel's intensity
	for i in [x for x in range(-D//2+1,D//2+1)]:
		for j in[y for y in range(-D//2,D//2+1)]:
			if 0<=loc[0]+i<img.shape[0] and 0<=loc[1]+j<img.shape[1]and abs(i)+abs(j)!=0:
				if (not M[(loc[0]+i,loc[1]+j)]) and (i**2+j**2)**0.5<D/2:
					iimn=img[loc[0]+i,loc[1]+j,0]
					rmn=(i**2+j**2)**0.5
					wmn=calculate_weight(loc,img,errorxy,rmn,i,j)
					if wtotal!=0:
						Nwmn=wmn/wtotal
						iimn=iimn+Nwmn*(errorxy+error0)*sxy
						iimn=max(0,min(iimn,255))
						img[loc[0]+i,loc[1]+j]=(iimn,iimn,iimn)

	

def calculate_weight(loc,img,errorxy,rmn,i,j):
	if errorxy>0:
		wmn=(img[loc[0]+i,loc[1]+j,0])/(rmn**2)
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
	man = mpimg.imread('im7.png')
	deep = cv.imread('disp7.png',0)
	man255=man*255
	M={}
	dot=2
	resolution=5

	# man255=np.maximum(z, man255)
	# plt.imshow(man255/255)
	# plt.show()
	# stippleimg=SAS(5,5,0,7,man255)
	stippleimg=SASD(5,5,0,7,man255,deep)
	# stippleimg=SASD(10,10,0,15,man255,depth)
	new_resolution=np.ones((man255.shape[0]*resolution,man255.shape[1]*resolution,3))*255
	for key, R in stippleimg.items():
		a,b=key[0],key[1]
		new_resolution[a*resolution,b*resolution,:]=(0,0,0);
		for i in range(-int(R*dot),int(R*dot)+1):
			for j in range(-int(R*dot),int(R*dot)+1):
				if (i**2+j**2)**0.5<=R*dot:
					new_resolution[min(new_resolution.shape[0]-1,a*resolution+i),min(new_resolution.shape[1]-1,b*resolution+j),:]=(0,0,0)
	
	plt.axis("off")
	plt.imshow(new_resolution/255)
	plt.show()
	mpimg.imsave('room2.png', new_resolution/255)
	
	# plt.axis("off")
	# plt.imshow(man255/255)
	# plt.show()
	# mpimg.imsave('255_man', man255/255)

