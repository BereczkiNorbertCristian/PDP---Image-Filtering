from threading import Thread

import imageio
import numpy as np


# THIS IS WITHOUT THREADS

FILENAME = '../dummy.png'
FILENAME_OUT = 'dummyout.png'
MODN = 256
WORKERS = 4


def copyRow(oi, ni, IDX):
    row_length = len(oi)
    for i in range(0, row_length):
        ni[i][IDX] = oi[i][IDX]

# USING MEDIAN FILTERING


def compute_median_n(oi, i, j):
    dist = [0] * 8
    cnt = 0
    for ii in range(i - 1, i + 2):
        for jj in range(j - 1, j + 2):
            if ii == i and jj == j:
                continue
            dist[cnt] = oi[ii][jj]
            cnt += 1

    dist.sort()
    return (dist[3] + dist[4]) // 2


def compute_linear_convolution(oi, i, j):
    sum_over = 0
    sum_over += oi[i][j + 1]
    sum_over += oi[i][j - 1]
    sum_over += oi[i + 1][j]
    sum_over += oi[i - 1][j]
    return sum_over // 4


def compute_convolution(oi, i, j):
    return compute_linear_convolution(oi, i, j)


def doWork(oi, ni, beg_row, end_row, row_length):
    for i in range(beg_row, end_row):
        for j in range(1, row_length - 1):
            ni[i][j] = compute_convolution(oi, i, j)

# READ IMAGE AND METADATA
old_image = imageio.imread(FILENAME)
w, h = old_image.shape

# INITIALIZE NEW IMAGE
new_image = np.ndarray(shape=(w, h), dtype='uint8')
new_image[0] = old_image[0]
new_image[w - 1] = old_image[w - 1]
copyRow(old_image, new_image, 0)
copyRow(old_image, new_image, h - 1)

# PROCESS ROWS IN INTERVAL (0...(w-1)) => [1...(w-2)]
threads = [0] * (WORKERS + 1)
cnt = 0
step = (w - 2) // WORKERS
stepi = 1
while stepi < w - 1:
    beg_row = stepi
    end_row = stepi + step
    if end_row > w - 1:
        end_row = w - 1
    t = Thread(target=doWork, args=(old_image, new_image, beg_row, end_row, h))
    threads[cnt] = t
    t.start()
    cnt += 1
    stepi += step

for i in range(0, cnt):
    threads[i].join()

imageio.imwrite(FILENAME_OUT, new_image[:, :])
