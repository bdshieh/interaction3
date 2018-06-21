# transducer / bem.py

import numpy as np
from scipy import sparse as sps, linalg
from scipy.constants import epsilon_0 as e_0
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from scipy.io import loadmat


## DECORATORS ##

def _make_hashable(arg):

    if isinstance(arg, (np.ndarray, sps.spmatrix)):
        return hash(str(arg))
        # return hash(arg.tostring())
    elif isinstance(arg, list):
        return tuple(arg)

    return arg


def memoize(func):

    memo = dict()

    def decorator(*args):

        key = tuple(_make_hashable(arg) for arg in args)

        if key not in memo:
            memo[key] = func(*args)

        return memo[key]

    return decorator


## BEM MATRICES ##

@memoize
def m_matrix(rho, h, nnodes):
    '''
    Mass matrix, generalized to composite membranes.
    '''
    return sps.eye(nnodes) * sum([x * y for x, y in zip(rho, h)])


@memoize
def b_matrix(att_mech, nnodes):
    '''
    Damping matrix.
    '''
    return sps.eye(nnodes) * att_mech


@memoize
def flexural_rigidity(E, h, eta):
    '''
    N-layer thin-plate flexural rigidity.
    '''
    if len(h) == 1:
        D = E[0] * h[0] ** 3 / 12 / (1 - eta[0] ** 2)

    elif len(h) > 1:

        z = np.cumsum([0] + h)
        a = z[1:] - z[:-1]
        b = (z[1:]**2 - z[:-1]**2) / 2
        c = (z[1:]**3 - z[:-1]**3) / 3
        q = np.array(E) / (1 - np.array(eta)**2)

        A = np.sum(q * a)
        B = np.sum(q * b)
        C = np.sum(q * c)

        D = (A*C - B**2) / A

    return D


@memoize
def k_matrix_fd1(E, h, eta, nodes, dx, dy):
    '''
    Stiffness matrix from old method finite-differences (from Sarp's code).
    '''
    D = flexural_rigidity(E, h, eta)

    x = nodes[:, 0]
    y = nodes[:, 1]

    nnodes = len(nodes)

    K = sps.dok_matrix((nnodes, nnodes), dtype=np.float64)
    eps = np.finfo(K.dtype).eps

    for node in range(nnodes):

        node_x = x[node]
        node_y = y[node]

        # self
        K[node, node] = 6 / dx**4 + 6 / dy**4

        # horizontal neighbors
        K[node, (np.abs(x - node_x + 2 * dx) <= eps) & (np.abs(y - node_y) <= eps)] = 1 / dx**4
        K[node, (np.abs(x - node_x - 2 * dx) <= eps) & (np.abs(y - node_y) <= eps)] = 1 / dx**4
        K[node, (np.abs(x - node_x + 1 * dx) <= eps) & (np.abs(y - node_y) <= eps)] = -4 / dx**4
        K[node, (np.abs(x - node_x - 1 * dx) <= eps) & (np.abs(y - node_y) <= eps)] = -4 / dx**4

        # vertical neighbors
        K[node, (np.abs(y - node_y + 2 * dx) <= eps) & (np.abs(x - node_x) <= eps)] = 1 / dy**4
        K[node, (np.abs(y - node_y - 2 * dx) <= eps) & (np.abs(x - node_x) <= eps)] = 1 / dy**4
        K[node, (np.abs(y - node_y + 1 * dx) <= eps) & (np.abs(x - node_x) <= eps)] = -4 / dy**4
        K[node, (np.abs(y - node_y - 1 * dx) <= eps) & (np.abs(x - node_x) <= eps)] = -4 / dy**4

        # all other neighbors
        K[node, (np.abs(x - node_x + 1 * dx) <= eps) & (np.abs(y - node_y + 1 * dy) <= eps)] = 2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x - 1 * dx) <= eps) & (np.abs(y - node_y - 1 * dy) <= eps)] = 2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x - 1 * dx) <= eps) & (np.abs(y - node_y + 1 * dy) <= eps)] = 2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x + 1 * dx) <= eps) & (np.abs(y - node_y - 1 * dy) <= eps)] = 2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x + 2 * dx) <= eps) & (np.abs(y - node_y + 2 * dy) <= eps)] = 2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x - 2 * dx) <= eps) & (np.abs(y - node_y - 2 * dy) <= eps)] = 2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x + 2 * dx) <= eps) & (np.abs(y - node_y - 2 * dy) <= eps)] = 2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x - 2 * dx) <= eps) & (np.abs(y - node_y + 2 * dy) <= eps)] = 2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x - 2 * dx) <= eps) & (np.abs(y - node_y + 1 * dy) <= eps)] = -2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x + 2 * dx) <= eps) & (np.abs(y - node_y - 1 * dy) <= eps)] = -2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x - 2 * dx) <= eps) & (np.abs(y - node_y - 1 * dy) <= eps)] = -2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x + 2 * dx) <= eps) & (np.abs(y - node_y + 1 * dy) <= eps)] = -2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x - 1 * dx) <= eps) & (np.abs(y - node_y + 2 * dy) <= eps)] = -2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x + 1 * dx) <= eps) & (np.abs(y - node_y - 2 * dy) <= eps)] = -2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x - 1 * dx) <= eps) & (np.abs(y - node_y - 2 * dy) <= eps)] = -2 / 9 / dx**2 / dy**2
        K[node, (np.abs(x - node_x + 1 * dx) <= eps) & (np.abs(y - node_y + 2 * dy) <= eps)] = -2 / 9 / dx**2 / dy**2

    return D*K


@memoize
def k_matrix_fd2(E, h, eta, dx, dy, nnodes, shape):
    '''
    Stiffness matrix from new method 4th-order finite-differences using stencils.
    '''
    D = flexural_rigidity(E, h, eta)

    nx, ny = shape

    # 5 point central difference 2nd derivative (4th order)
    fd_xx = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])

    # 7 point central difference 4th derivative (4th order)
    fd_xxxx = np.array([-1 / 6, 2, -13 / 2, 28 / 3, -13 / 2, 2, -1 / 6])

    # create finite difference stencil
    stencil_xxxx = np.zeros((7, 7))
    stencil_xxxx[3, :] = fd_xxxx
    stencil_yyyy = np.zeros((7, 7))
    stencil_yyyy[:, 3] = fd_xxxx

    a = np.zeros((7, 7))
    a[3, 1:-1] = fd_xx
    b = np.zeros((7, 7))
    b[1:-1, 3] = fd_xx
    stencil_xxyy = convolve2d(a, b, mode='same')

    stencil = stencil_xxxx / dx**4 + stencil_yyyy / dy**4 + 2 * stencil_xxyy / (dx**2 * dy**2)

    K = sps.dok_matrix((nnodes, nnodes), dtype=np.float64)

    for node_no in range(nnodes):

        Krow = np.zeros((nx + 6, ny + 6), dtype=np.float64)

        ix = (node_no % nx) + 3
        iy = (node_no // nx) + 3

        Krow[(ix - 3):(ix + 4), (iy - 3):(iy + 4)] = stencil

        # apply clamped boundary conditions
        Krow[-4, :] += Krow[-2, :]
        Krow[:, -4] += Krow[:, -2]
        Krow[-5, :] += Krow[-1, :]
        Krow[:, -5] += Krow[:, -1]
        Krow[3, :] += Krow[1, :]
        Krow[:, 3] += Krow[:, 1]
        Krow[4, :] += Krow[0, :]
        Krow[:, 4] += Krow[:, 0]

        K[node_no, :] = Krow[3:-3, 3:-3].ravel(order='F')

    return D*K


@memoize
def k_matrix_comsol(filename):
    '''
    Stiffness matrix from COMSOL simulation.
    '''
    return linalg.inv(loadmat(filename)['x'].T)


@memoize
def kss_matrix(K, e_mask, dc_bias, h_gap, h_isol, e_r, nnodes):
    '''
    '''
    h_eff = h_gap + h_isol / e_r

    u0, collapsed = solve_static_displacement(K, e_mask, dc_bias, h_eff)

    kss_diag = e_0 * dc_bias**2 / (h_eff + u0)**3
    kss_diag[~e_mask] = 0 # zero out non-electrode nodes
    Kss = sps.diags(kss_diag, 0)

    if dc_bias == 0:

        t_ratios = np.zeros(nnodes, dtype=np.float64)

    else:

        t_ratios = 2 * -e_0 * dc_bias**2 / (2 * (h_eff + u0)**2) / dc_bias
        t_ratios[~e_mask] = 0 # zero out non-electrode nodes

    return Kss, collapsed, u0, t_ratios


def zr_matrix(nodes, f, rho, c, a_n):
    '''
    Acoustic impedance matrix.
    '''
    dist = distance(nodes, nodes)
    k = 2 * np.pi * f / c
    a_eff = np.sqrt(a_n / np.pi)

    with np.errstate(divide='ignore', invalid='ignore'):
        zr = 1j * 2 * np.pi * f * rho * a_n / (2 * np.pi) * np.exp(-1j * k * dist) / dist

    zr[np.eye(*dist.shape).astype(bool)] = rho * c * (0.5 * (k * a_eff) ** 2 + 1j * 8 / (3 * np.pi) * k * a_eff)

    return zr


@memoize
def zr1_matrix(nodes, f, rho, c, a_n):
    '''
    Acoustic impedance matrix.
    '''
    dist = distance(nodes, nodes)
    k = 2 * np.pi * f / c
    a_eff = np.sqrt(a_n / np.pi)

    with np.errstate(divide='ignore', invalid='ignore'):
        zr = 1j * 2 * np.pi * f * rho * a_n / (2 * np.pi) * np.exp(-1j * k * dist) / dist

    zr[np.eye(*dist.shape).astype(bool)] = rho * c * (0.5 * (k * a_eff) ** 2 + 1j * 8 / (3 * np.pi) * k * a_eff)

    return zr


@memoize
def peq_matrix(filename):
    '''
    '''
    with np.load(filename) as root:
        Peq = root['Peq']
        freqs = root['freqs']

    return Peq, freqs


@memoize
def g1inv_matrix(g1):
    '''
    '''
    g1inv = linalg.inv(g1)

    return g1inv


## UTILITY FUNCTIONS ##

def distance(*args):
    return cdist(*np.atleast_2d(*args))


def solve_static_displacement(K, e_mask, dc_bias, h_eff, tol=1.0, maxiter=100):
    '''
    '''

    def p_es(u, e_mask):

        p = -e_0 * dc_bias ** 2 / (2 * (h_eff + u) ** 2)
        p[~e_mask] = 0

        return p

    if sps.issparse(K):
        Kinv = linalg.inv(K.todense())
    else:
        Kinv = linalg.inv(K)

    nnodes = K.shape[0]
    u0 = np.zeros(nnodes)

    for i in range(maxiter):

        err = K.dot(u0) - p_es(u0, e_mask)

        if np.max(np.abs(err)) < tol:
            is_collapsed = False
            return u0, is_collapsed

        u0 = Kinv.dot(p_es(u0, e_mask)).squeeze()

    is_collapsed = True
    return u0, is_collapsed
