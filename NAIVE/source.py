
import imageio
import numpy as np

# THIS IS WITHOUT THREADS

FILENAME = '../dummy.png'
FILENAME_OUT = 'dummyout.png'
MODN = 256

def copyRow(oi,ni,IDX):
	row_length = len(oi)
	for i in range(0,row_length):
		ni[i][IDX] = oi[i][IDX]

def add(a,b):
	if a + b >= 256 :
		return a + b - 256
	return a + b

def doNeighbours(oi,i,j):
	return oi[i-1][j] + oi[i][j-1] \
		+ oi[i-1][j-1] + oi[i+1][j+1] \
		+ oi[i+1][j] + oi[i][j+1] \
		+ oi[i-1][j+1] + oi[i+1][j-1]


old_image = imageio.imread(FILENAME)
w,h = old_image.shape
new_image = np.ndarray(shape=(w,h),dtype='uint8')

new_image[0] = old_image[0]
new_image[w-1] = old_image[w-1]
copyRow(old_image,new_image,0)
copyRow(old_image,new_image,h-1)

for i in range(1,w-1):
	for j in range(1,h-1):
		new_image[i][j] = doNeighbours(old_image,i,j) % MODN

imageio.imwrite(FILENAME_OUT,new_image[:,:])

