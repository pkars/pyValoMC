"""Mesh functions.

createRectangularMesh.m
createGridMesh.m
"""
import numpy as np

def create_rectangular_mesh(xsize, ysize, dh):
    xstart = -xsize/2 + dh/2
    xend = 0.5 * (xsize - dh)
    ystart = -ysize/2 + dh/2
    yend = 0.5 * (ysize - dh)
    xvec = np.arange(xstart, xend+dh, dh)
    yvec = np.arange(ystart, yend+dh, dh)
    return create_grid_mesh(xvec, yvec)

def create_grid_mesh(xvec, yvec, zvec=None):
    if zvec is None:
        output = _2d_grid_mesh(xvec, yvec)
    else:
        output = _3d_grid_mesh(xvec, yvec, zvec)
    return output

def _2d_grid_mesh(xvec, yvec):
    r = []
    H = []
    BH = []

    dx = abs(xvec[1] - xvec[0])
    dy = abs(yvec[1] - yvec[0])

    gridvecx = xvec - 0.5*dx
    gridvecy = yvec - 0.5*dy

    gridvecx = np.append(gridvecx, gridvecx[-1] + dx)
    gridvecy = np.append(gridvecy, gridvecy[-1] + dy)

    x, y = np.meshgrid(gridvecx, gridvecy)
    r = np.array([x.flatten(), y.flatten()]).T

    num_x_voxels = len(xvec)
    num_y_voxels = len(yvec)


    # [ORIGINAL] TODO: for loops can be optimised away
    ysize = num_y_voxels + 1
    xsize = num_x_voxels + 1

    H = np.zeros((num_x_voxels * num_y_voxels * 2, 3))

    k = 1
    n = 0
    for i in range(1, num_x_voxels+1):
        for j in range(1, num_y_voxels+1):
            H[n, :] = np.array([k, k+1, k+ysize])
            k += 1
            n += 1
        k += 1
    k = 1
    for i in range(1, num_x_voxels+1):
        for j in range(num_y_voxels):
            H[n, :] = np.array([k+1, k+ysize+1, k+ysize])
            n += 1
            k += 1
        k += 1

    for j in range(1, num_y_voxels+1):
        BH.append([j, j+1])
    for j in range(1, num_x_voxels+1):
        BH.append([ysize*j, ysize*(j+1)])
    for j in range(1, num_y_voxels+1):
        BH.append([ysize*xsize - (j-1), ysize*xsize - j])
    for j in range(1, num_x_voxels+1):
        BH.append([ysize*(xsize-1)+1-ysize*(j-1), ysize*(xsize-1)+1-ysize*j])

    BH = np.asarray(BH)

    return r, H, BH

def _3d_grid_mesh(xvec, yvec, zvec):
    raise NotImplementedError
