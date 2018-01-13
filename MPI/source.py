from mpi4py import MPI
import imageio as imageio
import numpy as np

FILENAME = '../dummy.png'
FILENAME_OUT = 'dummyout.png'
MODN = 256
comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()
WORKERS = SIZE - 1


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
def copyRow(oi, ni, IDX):
    row_length = len(oi)
    for i in range(0, row_length):
        ni[i][IDX] = oi[i][IDX]


new_image = np.ndarray(shape=(w, h), dtype='uint8')
new_image[0] = old_image[0]
new_image[w - 1] = old_image[w - 1]
copyRow(old_image, new_image, 0)
copyRow(old_image, new_image, h - 1)

# PROCESS ROWS IN INTERVAL (0...(w-1)) => [1...(w-2)]
step = (w - 2) // WORKERS
stepi = 1
if RANK == 0:
    comm.bcast({"old_image": old_image}, root=0)
    for x in range(WORKERS):
        beg_row = stepi
        end_row = stepi + step
        if end_row > w - 1:
            end_row = w - 1
        # t = Thread(target=doWork, args=(old_image, new_image, beg_row, end_row, h))
        stepi += step
        comm.send({"beg_row": beg_row, "end_row": end_row, "h": h}, dest=x + 1, tag=x + 11)
        new_image[beg_row:end_row] = comm.recv(source=x + 1, tag=x + 11)["new_image"]
else:
    old_image = comm.bcast({"old_image": old_image}, root=0)["old_image"]
    data = comm.recv(source=0, tag=RANK + 10)
    doWork(old_image, new_image, data["beg_row"], data["end_row"], data["h"])
    comm.send({"new_image": new_image[data["beg_row"]:data["end_row"]]}, dest=0, tag=RANK + 10)

imageio.imwrite(FILENAME_OUT, new_image[:, :])
