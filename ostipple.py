from numpy import *
from random import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from heapq import heappush,heappop
import numpy as np
import cv2 as cv


class Stippler():


	def __init__(self,output_name,img,deep):
		# self.parameter=parameter
		self.output_name=output_name
		self.dot=2
		self.resolution=5
		self.img=img
		self.deep=deep

	def edge_detection(self,deep):
		edge_list=[]
		edges=cv.Canny(deep,100,200)
		for x in range(edges.shape[0]):
			for y in range(edges.shape[1]):
				if edges[x,y]==255:
					edge_list.append((x,y))
		return edge_list


	def SASD(self,G0,G1,k,D,img,deep):
		# G0=self.parameter.Gp
		# G1=self.parameter.Gn
		# k=self.parameter.k
		# D=self.mask_size
		man255 = mpimg.imread('img8.png')*255
		deep = cv.imread('disp8.png',0)
		resdeep = cv.resize(deep,None,fx=640/74, fy=480/55, interpolation = cv.INTER_CUBIC)
		self.edge_list=self.edge_detection(deep)
		self.aheap,self.M=self.build_heap_depth(img,self.edge_list)
		self.stipplelist={}
		while (self.aheap):
			pixel=heappop(self.aheap)
			P,loc,I=pixel[0],pixel[1],pixel[2]
			if P!=-self.calculate_priority_depth(img[loc[0],loc[1]],loc,self.edge_list):
					heappush(self.aheap,(-self.calculate_priority_depth(img[loc[0],loc[1]],loc,self.edge_list),loc,img[loc[0],loc[1],0]))
			else:
				if not self.M[loc]:
					R=self.stipple_size_depth_big(deep,loc)
					R1=self.stipple_size(img,loc)
					if I<=128:
						App=0
						self.stipplelist[loc]=(R+R1)/2
					else:
						App=255
					errorxy=I-App
					self.error_diffusion(loc,errorxy,R,G0,G1,k,D,img)
					self.M[loc]=True

		res=self.enlarge(img)		


	def error_diffusion(self,loc,errorxy,R,G0,G1,k,D,img):
		sxy=self.sh_ex(errorxy,R,G0,G1)
		wtotal=0
		Astipple=0
		dot=self.dot
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
					if (not self.M[(loc[0]+i,loc[1]+j)]) and (i**2+j**2)**0.5<D/2:
						rmn=(i**2+j**2)**0.5
						wmn=self.calculate_weight(loc,img,errorxy,rmn,i,j)
						wtotal+=wmn
		
		# modifty signle pixel's intensity
		for i in [x for x in range(-D//2+1,D//2+1)]:
			for j in[y for y in range(-D//2,D//2+1)]:
				if 0<=loc[0]+i<img.shape[0] and 0<=loc[1]+j<img.shape[1]and abs(i)+abs(j)!=0:
					if (not self.M[(loc[0]+i,loc[1]+j)]) and (i**2+j**2)**0.5<D/2:
						iimn=img[loc[0]+i,loc[1]+j,0]
						rmn=(i**2+j**2)**0.5
						wmn=self.calculate_weight(loc,img,errorxy,rmn,i,j)
						if wtotal!=0:
							Nwmn=wmn/wtotal
							iimn=iimn+Nwmn*(errorxy+error0)*sxy
							iimn=max(0,min(iimn,255))
							img[loc[0]+i,loc[1]+j]=(iimn,iimn,iimn)

		return None

	def build_heap_depth(self,img,edge_list):
		ph=[]
		M={}
		for x in range(img.shape[0]):
			for y in range(img.shape[1]):
				priority=self.calculate_priority_depth(img[x,y],(x,y),edge_list)
				heappush(ph,(-priority,(x,y),img[x,y,0]))
				M[(x,y)]=False
		return ph,M



	def calculate_priority_depth(self,pixel,loc,edge_list):
		if loc in edge_list:
			priority=300		
		else:	
			if abs(255-pixel[0])>abs(pixel[0]):
				priority=255-pixel[0]
			else:
				priority=pixel[0]
		return priority

	def stipple_size(self,aimg,aloc):
		rmax=2
		rmin=1
		size=rmin+((rmax-rmin)*(255-aimg[aloc[0],aloc[1],0])/255)
		return size

	# close smaller dot
	def stipple_size_depth_small(self,adeep,aloc):
		rmax=2
		rmin=1
		size=rmin+((rmax-rmin)*(255-adeep[aloc[0],aloc[1]])/255)
		return size

	# close bigger dot
	def stipple_size_depth_big(self,adeep,aloc):
		rmax=2
		rmin=1
		size=rmax-((rmax-rmin)*(255-adeep[aloc[0],aloc[1]])/255)
		return size

	def calculate_weight(self,loc,img,errorxy,rmn,i,j):
		if errorxy>0:
			wmn=(img[loc[0]+i,loc[1]+j,0])/(rmn**2)
		else:
			wmn=(255-img[loc[0]+i,loc[1]+j,0])/(rmn**2)
		return wmn


	def sh_ex(self,errorxy,Rsize,Gamma0,Gamma1):
		if errorxy<0:
			sxy=(1/Rsize)**Gamma0
		else:
			sxy=(Rsize)**Gamma1
		return sxy


	def enlarge(self,img):
		# dot=self.dot
		# resolution=self.resolution
		# output_name=self.output_name
		self.new_resolution=np.ones((img.shape[0]*self.resolution,img.shape[1]*self.resolution,3))*255
		for key, R in self.stipplelist.items():
			a,b=key[0],key[1]
			self.new_resolution[a*self.resolution,b*self.resolution,:]=(0,0,0);
			for i in range(-int(R*self.dot),int(R*self.dot)+1):
				for j in range(-int(R*self.dot),int(R*self.dot)+1):
					if (i**2+j**2)**0.5<=R*self.dot:
						self.new_resolution[min(self.new_resolution.shape[0]-1,a*self.resolution+i),min(self.new_resolution.shape[1]-1,b*self.resolution+j),:]=(0,0,0)
		mpimg.imsave(self.output_name, self.new_resolution/255)
		return None

if __name__== "__main__":
	man255 = mpimg.imread('img8.png')*255
	deep = cv.imread('disp8.png',0)
	resdeep = cv.resize(deep,None,fx=640/74, fy=480/55, interpolation = cv.INTER_CUBIC)
	stippler1=Stippler("OO2");
	stippler1.SASD(5,5,0,7,man255,resdeep)
	# stippleimg=SASD(10,10,0,15,man255,depth)

	
	# plt.axis("off")
	# plt.imshow(new_resolution/255)
	# plt.show()
	# 