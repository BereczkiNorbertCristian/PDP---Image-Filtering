
from __future__ import division

import numpy as np
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools

import pycuda.autoinit
import imageio

# ---------------------- FUNCTIONS -------------------------


def preproc_old_image(old_image):
    w, h = old_image.shape
    WHITE = 255
    SQUARE_SIZE = 450
    proc_image = np.full((SQUARE_SIZE, SQUARE_SIZE), WHITE, np.int32)
    for i in range(0, w):
        for j in range(0, h):
            proc_image[i][j] = old_image[i][j]
    return proc_image


def recompute_image(old_image, new_image_gpu):
    new_image = old_image
    w, h = new_image.shape
    for i in range(0, w):
        for j in range(0, h):
            new_image[i][j] = new_image_gpu[i][j]
    return new_image

# ----------------------- KERNEL CODE TEMPLATE ----------------
kernel_code_template = '''
	#include<stdio.h>

	__global__ void Convolution(int *old_image,int *new_image){{

		const uint MATRIX_SIZE = {MATRIX_SIZE} ;
		const uint BLOCK_SIZE = {BLOCK_SIZE} ;

		int row_block = blockIdx.y * MATRIX_SIZE * BLOCK_SIZE;
		int col_block = blockIdx.x * BLOCK_SIZE;
		int current_row = row_block + threadIdx.y * MATRIX_SIZE;
		int current_col = col_block + threadIdx.x;
		
		if(current_row != 0 && current_col != 0 && current_row != MATRIX_SIZE - 1 && current_col != MATRIX_SIZE - 1) {{

			int row_above = current_row - MATRIX_SIZE;
			int row_below = current_row + MATRIX_SIZE;
			int col_left = current_col - 1;
			int col_right = current_col + 1;

			int idx_current = current_row + current_col;
			int idx_above = row_above + current_col;
			int idx_below = row_below + current_col;
			int idx_left = current_row + col_left;
			int idx_right = current_row + col_right;

			int sum = 0;
			sum += old_image[idx_above];
			sum += old_image[idx_below];
			sum += old_image[idx_left];
			sum += old_image[idx_right];

			new_image[idx_current] = sum / 4;
		}}

	}}

'''

# ---------------------- MODULE CONSTANTS --------------------
FILENAME = '../dummy.png'
FILENAME_OUT = 'dummyout.png'
MODN = 256

# ---------------------- READ AND PREPROC IMAGE ---------------
old_image = imageio.imread(FILENAME)
w, h = old_image.shape
old_image_preproc = preproc_old_image(old_image)
w, h = old_image_preproc.shape

# ----------------------- INIT CUDA PARAMS --------------------
BLOCK_SIZE = 10
MATRIX_SIZE = w

assert w == h
assert w % BLOCK_SIZE == 0

old_image_gpu = gpuarray.to_gpu(old_image_preproc)
new_image_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.int32)

placeholder_dict = {
    'MATRIX_SIZE': MATRIX_SIZE,
    'BLOCK_SIZE': BLOCK_SIZE
}

kernel_code = kernel_code_template.format(**placeholder_dict)
mod = compiler.SourceModule(kernel_code)

convolution = mod.get_function("Convolution")

convolution(
    # input
    old_image_gpu,
    # ouput
    new_image_gpu,
    grid=((MATRIX_SIZE // BLOCK_SIZE), (MATRIX_SIZE // BLOCK_SIZE)),
    block=(BLOCK_SIZE, BLOCK_SIZE, 1),
)

new_image = recompute_image(old_image,new_image_gpu.get())

imageio.imwrite(FILENAME_OUT, new_image[:, :])
