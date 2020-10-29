from pathlib import Path
import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time


def regionfill(I, mask, factor=1.0):
    if np.count_nonzero(mask) == 0:
        return I.copy()
    start_time = time.time()
    resize_mask = cv2.resize(mask.astype(float),
                             (0, 0), fx=factor, fy=factor) > 0
    resize_I = cv2.resize(I.astype(float), (0, 0), fx=factor, fy=factor)
    print(f'resize time: {time.time() - start_time}')
    start_time = time.time()
    maskPerimeter = findBoundaryPixels(resize_mask)
    print(f'findBoundaryPixels time: {time.time() - start_time}')
    start_time = time.time()
    regionfillLaplace(resize_I, resize_mask, maskPerimeter)
    print(f'regionfillLaplace time: {time.time() - start_time}')
    resize_I = cv2.resize(resize_I, (I.shape[1], I.shape[0]))
    resize_I[mask == 0] = I[mask == 0]
    return resize_I


def findBoundaryPixels(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    maskDilated = cv2.dilate(mask.astype(float), kernel)
    return (maskDilated > 0) & (mask == 0)


def regionfillLaplace(I, mask, maskPerimeter):
    height, width = I.shape
    rightSide = formRightSide(I, maskPerimeter)

    # Location of mask pixels
    maskIdx = np.where(mask)

    # Only keep values for pixels that are in the mask
    rightSide = rightSide[maskIdx]

    # Number the mask pixels in a grid matrix
    grid = -np.ones((height, width))
    grid[maskIdx] = range(0, maskIdx[0].size)
    # Pad with zeros to avoid "index out of bounds" errors in the for loop
    grid = padMatrix(grid)
    gridIdx = np.where(grid >= 0)

    # Form the connectivity matrix D=sparse(i,j,s)
    # Connect each mask pixel to itself
    i = np.arange(0, maskIdx[0].size)
    j = np.arange(0, maskIdx[0].size)
    # The coefficient is the number of neighbors over which we average
    numNeighbors = computeNumberOfNeighbors(height, width)
    s = numNeighbors[maskIdx]
    # Now connect the N,E,S,W neighbors if they exist
    for direction in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        # Possible neighbors in the current direction
        neighbors = grid[gridIdx[0] + direction[0], gridIdx[1] + direction[1]]
        # ConDnect mask points to neighbors with -1's
        index = (neighbors >= 0)
        i = np.concatenate((i, grid[gridIdx[0][index], gridIdx[1][index]]))
        j = np.concatenate((j, neighbors[index]))
        s = np.concatenate((s, -np.ones(np.count_nonzero(index))))

    D = sparse.coo_matrix((s, (i.astype(int), j.astype(int)))).tocsr()
    sol = spsolve(D, rightSide)
    I[maskIdx] = sol
    return I


def formRightSide(I, maskPerimeter):
    height, width = I.shape
    perimeterValues = np.zeros((height, width))
    perimeterValues[maskPerimeter] = I[maskPerimeter]
    rightSide = np.zeros((height, width))

    rightSide[1:height - 1, 1:width -
              1] = (perimeterValues[0:height - 2, 1:width - 1] +
                    perimeterValues[2:height, 1:width - 1] +
                    perimeterValues[1:height - 1, 0:width - 2] +
                    perimeterValues[1:height - 1, 2:width])

    rightSide[1:height - 1, 0] = (perimeterValues[0:height - 2, 0] +
                                  perimeterValues[2:height, 0] +
                                  perimeterValues[1:height - 1, 1])

    rightSide[1:height - 1, width -
              1] = (perimeterValues[0:height - 2, width - 1] +
                    perimeterValues[2:height, width - 1] +
                    perimeterValues[1:height - 1, width - 2])

    rightSide[0, 1:width - 1] = (perimeterValues[1, 1:width - 1] +
                                 perimeterValues[0, 0:width - 2] +
                                 perimeterValues[0, 2:width])

    rightSide[height - 1, 1:width -
              1] = (perimeterValues[height - 2, 1:width - 1] +
                    perimeterValues[height - 1, 0:width - 2] +
                    perimeterValues[height - 1, 2:width])

    rightSide[0, 0] = perimeterValues[0, 1] + perimeterValues[1, 0]
    rightSide[0, width - 1] = (perimeterValues[0, width - 2] +
                               perimeterValues[1, width - 1])
    rightSide[height - 1, 0] = (perimeterValues[height - 2, 0] +
                                perimeterValues[height - 1, 1])
    rightSide[height - 1, width -
              1] = (perimeterValues[height - 2, width - 1] +
                    perimeterValues[height - 1, width - 2])
    return rightSide


def computeNumberOfNeighbors(height, width):
    # Initialize
    numNeighbors = np.zeros((height, width))
    # Interior pixels have 4 neighbors
    numNeighbors[1:height - 1, 1:width - 1] = 4
    # Border pixels have 3 neighbors
    numNeighbors[1:height - 1, (0, width - 1)] = 3
    numNeighbors[(0, height - 1), 1:width - 1] = 3
    # Corner pixels have 2 neighbors
    numNeighbors[(0, 0, height - 1, height - 1),
                 (0, width - 1, 0, width - 1)] = 2
    return numNeighbors


def padMatrix(grid):
    height, width = grid.shape
    gridPadded = -np.ones((height + 2, width + 2))
    gridPadded[1:height + 1, 1:width + 1] = grid
    gridPadded = gridPadded.astype(grid.dtype)
    return gridPadded


def read_flow_file(flow_file_path: Path):
    with flow_file_path.open('rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        w, h = int(w), int(h)
        flow_data = np.fromfile(f, np.float32, count=2 * w * h)
        flow_data = np.resize(flow_data, (h, w, 2))
        return flow_data


if __name__ == '__main__':
    flow_file_path = Path('./data/debug/inpaint_flow/forward_flo/00000.flo')
    flow = read_flow_file(flow_file_path)
    mask_path = Path(
        './data/debug/inpaint_flow/demo视频-3-dilate-mask/frm_1-mask.png')
    mask = cv2.imread(mask_path.as_posix(), 0)
    mask = cv2.resize(mask, (flow.shape[1], flow.shape[0]))

    flow[:, :, 0] = regionfill(flow[:, :, 0], mask)
