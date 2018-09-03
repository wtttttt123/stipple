from heapq import heappush,heappop
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
import time

class Stippler():

	def __init__(self, output_name, img, depth_map, edges=None):
		self.output_name=output_name
		self.img=img
		self.deep=depth_map
		self.edges=edges		
		self.dot=2
		self.resolution=5

	def stippling(self, G0, G1, k, D):
		self.G0=G0
		self.G1=G1
		self.k=k
		self.D=D
		self.img = mpimg.imread(self.img)
		self.img = (np.dot(self.img[...,:3], [0.299, 0.587, 0.114]))*255
		self.deep = cv.imread(self.deep,0)
		if self.edges!=None:
			self.edges = mpimg.imread(self.edges)*255
			self.edge_list = self.__edge_list(self.edges)
		else:
			self.edge_list = self.__edge_detection(self.deep)
		self.aheap,self.M = self.__build_heap_depth()
		self.stipplelist={}
		while (self.aheap): 
			pixel=heappop(self.aheap)
			P,loc,I=pixel[0],pixel[1],pixel[2]
			if P!=-self.__calculate_priority_depth(self.img[loc[0],loc[1]],loc,self.edge_list):
					new_priority=-self.__calculate_priority_depth(self.img[loc[0],loc[1]],loc,self.edge_list)
					heappush(self.aheap,(new_priority,loc,self.img[loc[0],loc[1]]))
			else:
				if not self.M[loc]:
					R=self.__stipple_size_depth_small(self.deep,loc)
					R1=self.__stipple_size(self.img,loc)
					Rz=(R+R1)/2
					if I<=128:
						App=0
						self.stipplelist[loc]=Rz
					else:
						App=255
					self.errorxy=I-App
					self.__error_diffusion(loc,Rz)
					self.M[loc]=True
		self.__enlarge()		


	def __error_diffusion(self,loc,R):
		sxy=self.__sh_ex(R)
		wtotal=0
		Astipple=0
		dot=self.dot
		D=self.D
		for i in range(-int(R*dot),int(R*dot)+1):
			for j in range(-int(R*dot),int(R*dot)+1):
				if (i**2+j**2)**0.5<R*dot:
					Astipple+=1
		Apixcel=1
		self.error0=(Astipple-Apixcel)*self.k
			
		# calculate_total weight
		for i in range(-D//2+1,D//2+1):
			for j in range(-D//2+1,D//2+1):
				if 0<=loc[0]+i<self.img.shape[0] and 0<=loc[1]+j<self.img.shape[1] and abs(i)+abs(j)!=0:
					if (not self.M[(loc[0]+i,loc[1]+j)]) and (i**2+j**2)**0.5<D/2:
						rmn=(i**2+j**2)**0.5
						wmn=self.__calculate_weight(loc,rmn,i,j)
						wtotal+=wmn
		
		# modifty signle pixel's intensity
		for i in range(-D//2+1,D//2+1):
			for j in range(-D//2,D//2+1):
				if 0<=loc[0]+i<self.img.shape[0] and 0<=loc[1]+j<self.img.shape[1]and abs(i)+abs(j)!=0:
					if (not self.M[(loc[0]+i,loc[1]+j)]) and (i**2+j**2)**0.5<D/2:
						iimn=self.img[loc[0]+i,loc[1]+j]
						rmn=(i**2+j**2)**0.5
						wmn=self.__calculate_weight(loc,rmn,i,j)
						if wtotal!=0:
							Nwmn=wmn/wtotal
							iimn=iimn+Nwmn*(self.errorxy+self.error0)*sxy
							iimn=max(0,min(iimn,255))
							self.img[loc[0]+i,loc[1]+j]=iimn
	
	
	def __sh_ex(self,Rsize):
		if self.errorxy<0:
			sxy=(1/Rsize)**self.G0
		else:
			sxy=(Rsize)**self.G1
		return sxy

	def __calculate_weight(self,loc,rmn,i,j):
		if self.errorxy>0:
			wmn=(self.img[loc[0]+i,loc[1]+j])/(rmn**2)
		else:
			wmn=(255-self.img[loc[0]+i,loc[1]+j])/(rmn**2)
		return wmn

	
	def __edge_list(self,edges):
		edge_list={}
		for x in range(edges.shape[0]):
			for y in range(edges.shape[1]):
				if edges[x,y,0]!=255:
					edge_list[(x,y)]="E"
				else:
					edge_list[(x,y)]="N"
		return edge_list

	def __edge_detection(self,depth_map):
		edge_list={}
		edges=cv.Canny(depth_map,100,200)
		plt.imshow(edges,cmap = 'gray')
		plt.show()
		for x in range(edges.shape[0]):
			for y in range(edges.shape[1]):
				if edges[x,y]==255:
					edge_list[(x,y)]="E"
				else:
					edge_list[(x,y)]="N"
		return edge_list
 

	

	def __build_heap_depth(self):
		ph=[]
		M={}
		for x in range(self.img.shape[0]):
			for y in range(self.img.shape[1]):
				priority=self.__calculate_priority_depth(self.img[x,y],(x,y),self.edge_list)
				heappush(ph,(-priority,(x,y),self.img[x,y]))
				M[(x,y)]=False
		return ph,M

	def __calculate_priority_depth(self,intensity,loc,edge_list):
		if edge_list[(loc)]=="E":
			priority=300		
		else:	
			if abs(255-intensity)>abs(intensity):
				priority=255-intensity
			else:
				priority=intensity
		return priority

	
	def __stipple_size(self,aimg,aloc):
		rmax=2
		rmin=1
		size=rmin+((rmax-rmin)*(255-aimg[aloc[0],aloc[1]])/255)
		return size

	# close smaller dot
	def __stipple_size_depth_small(self,adeep,aloc):
		rmax=2
		rmin=1
		size=rmax-((rmax-rmin)*(255-adeep[aloc[0],aloc[1]])/255)
		return size

	# close bigger dot
	def __stipple_size_depth_big(self,adeep,aloc):
		rmax=2
		rmin=1
		size=rmin+((rmax-rmin)*(255-adeep[aloc[0],aloc[1]])/255)
		return size

	



	def __enlarge(self):
		self.new_resolution=np.ones((self.img.shape[0]*self.resolution,self.img.shape[1]*self.resolution,3))*255
		for key, R in self.stipplelist.items():
			a,b=key[0],key[1]
			self.new_resolution[a*self.resolution,b*self.resolution,:]=(0,0,0);
			for i in range(-int(R*self.dot),int(R*self.dot)+1):
				for j in range(-int(R*self.dot),int(R*self.dot)+1):
					if (i**2+j**2)**0.5<=R*self.dot:
						self.new_resolution[min(self.new_resolution.shape[0]-1,a*self.resolution+i),min(self.new_resolution.shape[1]-1,b*self.resolution+j),:]=(0,0,0)
		mpimg.imsave("./test/"+self.output_name, self.new_resolution/255)

if __name__== "__main__":
	start_time = time.time()
	# stippler1=Stippler("dic.png",'01444_img.png','01444_dep.png',"01444ee.png");
	stippler1=Stippler("dic.png",'00432_img.png','00432_dep.png');
	stippler1.stippling(5,5,0,7)
	#stippler1.stippling(10,10,0,15)
	# print(stippler1._Stippler__sh_ex(2,1,5,5))
	print("--- %s seconds ---" % (time.time() - start_time))