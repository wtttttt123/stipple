from numpy import *
from random import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


man = mpimg.imread('man.png')
cat = mpimg.imread('cat.png')

nrow = man.shape[0]
ncolumn = man.shape[1]

print(nrow)
print (ncolumn)



def stipple(img):
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			if img[x,y,0]>0.5:
				k=1.0
				error=img[x,y,:]-k
				img[x,y]=(1,1,1)
				
			else:
				k=0.0
				error=img[x,y,:]-k
				img[x,y]=(0,0,0)
			if (y<299):
				img[x,y+1,:]=img[x,y+1,:]+error

# def build_list(img):
# 	h={}
# 	for x in range(img.shape[0]):
# 		for y in range(img.shape[1]):
# 			h[(x,y)]=calculate_priority(img[x,y,0]),(x,y),img[x,y,0]
# 			sorted(h, key=lambda pixel: student[2])
# 			M[(x,y)]=False
# 	return h

# def error_diffusion:
# 	error=


stipple(man)
plt.axis("off")
plt.imshow(man)
plt.show()		

