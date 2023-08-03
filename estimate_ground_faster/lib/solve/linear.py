

import numpy as np

def solve_ground_from_xyz(points:np.ndarray) -> (np.ndarray):
    '''
    input:
        points [n, 3]
    output :
        ground [4] which is (A, B, C, D) of the ground Ax+By+Cz+D=0
        
    solve the linear system AX=0 |X|=1
        * A[n, 4] each line is a 3d point [x, y, z, 1]
        * X[4, 1] is the ground 
        * 0[n, 1] 
    '''
    '''
    specificly
    [[x1, y1, z1, 1]  [[A]    [[0]
     [x2, y2, z2, 1]   [B]  =  [0]
                       [C]     [0]
     [xn, yn, zn, 1]]  [D]]    [0]] 
    '''
    A = np.ones((points.shape[0], 4))
    A[:, 0:3] = points
    ____, s, v = np.linalg.svd(A)
    X = v[np.argmin(s), :]
    return X