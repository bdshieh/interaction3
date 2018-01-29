## bem / core / fma_functions.pyx
'''
Core functions for fast multipole calculations.
Author: Bernard Shieh (bshieh@gatech.edu)
'''
import numpy as np
cimport numpy as np
#import scipy as sp

import cython
cimport cython

from libc.math cimport sqrt, cos, sin, exp
from complex cimport cabs, carg, cexp, creal, cimag, conj
from scipy.fftpack import fft, ifft, fft2, ifft2, fftshift, ifftshift
from scipy.special import eval_legendre, hankel2


## BASIC FUNCTIONS ##

#cdef double pi = np.pi
pi = np.pi

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mag(double[:] r):
    '''
    Computes the magnitude of a vector.
    '''
    return sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray distance(double[:,:] a, double[:,:] b):
    '''
    Calculates the pair-wise distance between two sets of points.
    '''
    cdef int M1 = a.shape[0]
    cdef int N1 = a.shape[1]
    cdef int M2 = b.shape[0]
    cdef int N2 = b.shape[1]
    cdef double[:,:] ret = np.zeros((M1, M2), dtype=np.double)
    cdef double d, tmp
    cdef int i, j, k

    for i in range(M1):
        for j in range(M2):

            d = 0.0

            for k in range(N1):

                tmp = a[i, k] - b[j, k]
                d += tmp * tmp

            ret[i, j] = sqrt(d)

    return np.asarray(ret)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray dir2coord(kdir):
    '''
    Transforms angular directions from spherical to cartesian.
    '''
    theta = kdir[:, :, 0]
    phi = kdir[:, :, 1]

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)

    kcoord = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)

    return kcoord


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double complex sph_hankel2(int l, double z):
    '''
    Spherical Hankel function of the second kind of order l and argument z
    calculated from Hankel function (does not handle z=0 case).
    '''
    return sqrt(pi / (2 * z)) * hankel2(l + 0.5, z)


@cython.boundscheck(False)
@cython.wraparound(False)
def sph_hankel2_np(l, z):
    '''
    Spherical Hankel function of the second kind of order l and argument z
    calculated from Hankel function (does not handle z=0 case) (numpy version).
    '''
    return sqrt(pi / (2 * z)) * hankel2(l + 0.5, z)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray direct_eval(double complex[:] q, double[:, :] dist, double k, double rho, double c):
    '''
    Evaluates the field pressure directly using the exact method.
    '''
    cdef M = dist.shape[0]
    cdef N = dist.shape[1]
    cdef double complex[:] ret = np.zeros(M, dtype=np.complex128)
    cdef int i, j
    cdef double complex tmp
    cdef double r

    for i in range(M):

        tmp = 0.0j

        for j in range(N):

            #if i == j: continue
            r = dist[i, j]

            if r == 0.0: continue

            #tmp += 1j*k*rho*c/(4*pi)*cexp(-1j*k*r)/r*q.pointer[j]
            tmp += cexp(-1j * k * r) / r * q[j]

        #ret[i] =  tmp
        ret[i] = 1j * k * rho * c / (4 * pi) * tmp

    return np.asarray(ret)


## FAST MULTIPOLE FUNCTIONS ##
'''
Implements fast multipole functions based on 'high-frequency' fast multipole 
method. A good primer on this method can be found here:
    
[1] Rahola, Jussi. "Diagonal forms of the translation operators in the fast 
multipole algorithm for scattering problems." BIT Numerical Mathematics 36.2 
(1996): 333-358.
    
To make the method compatible with an FFT-based uniform sampling scheme,
a modified translation operator is used which is explained here:
    
[2] Cecka, Cris, and Eric Darve. "Fourier-based fast multipole method for the 
Helmholtz equation." SIAM Journal on Scientific Computing 35.1 (2013): A79-A103.
'''

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray ff_coeff(double complex[:] q, double complex[:, :, :] exp_part):
    '''
    Far-field signature coefficients of a collection of sources in
    the specified far-field directions.
    '''
    cdef int M1 = q.shape[0]
    cdef int M2 = exp_part.shape[1]
    cdef int N2 = exp_part.shape[2]
    cdef double complex[:, :] coeff = np.zeros((M2, N2), dtype=np.complex128)
    cdef double complex tmp1
    cdef int i, j, l

    for i in range(M2):
        for j in range(N2):

            tmp1 = 0j

            for l in range(M1):
                tmp1 += q[l] * conj(exp_part[l, i, j])

            coeff[i, j] = tmp1

    return np.asarray(coeff)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray calc_exp_part(double[:, :] nodes, double[:] center, double[:, :, :] kcoord, double k):
    '''
    Calculates the exponential part of the evaluation equation (see nf_eval).
    '''
    cdef int M1 = nodes.shape[0]
    cdef int N1 = nodes.shape[1]
    cdef int M2 = kcoord.shape[0]
    cdef int N2 = kcoord.shape[1]
    cdef int O2 = kcoord.shape[2]
    cdef double complex[:, :, :] exp_part = np.zeros((M1, M2, N2), dtype=np.complex128)
    cdef double tmp1
    cdef int i, j, l, m

    for i in range(M1):
        for j in range(M2):
            for l in range(N2):

                tmp1 = 0.0

                for m in range(N1):
                    tmp1 += (nodes[i, m] - center[m]) * kcoord[j, l, m]

                exp_part[i, j, l] = cexp(1j * k * tmp1)

    return np.asarray(exp_part)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray nf_eval(double complex[:, :] coeff, double complex[:, :, :] exp_part, double k, double rho, double c):
    '''
    Evaluate the pressure field at the specified field point(s) using
    near-field signature coefficients.
    '''
    cdef int M1 = coeff.shape[0]
    cdef int N1 = coeff.shape[1]
    cdef int nnodes = exp_part.shape[0]
    cdef double prefactor = k * k * rho * c / (16 * pi * pi)
    cdef double complex[:] total = np.zeros(nnodes, dtype=np.complex128)
    cdef int i, j, l
    cdef double complex tmp1

    for i in range(nnodes):

        tmp1 = 0j

        for j in range(M1):
            for l in range(N1):

                tmp1 += coeff[j,l]*exp_part[i,j,l]

        total[i] = prefactor*tmp1

    return np.asarray(total)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray ff2nf_op(double r, double[:,:] cos_angle, double k, int trans_order):
    '''
    Standard far-field to near-field translation operator (faster version).
    '''
    cdef int M = cos_angle.shape[0]
    cdef int N = cos_angle.shape[1]
    cdef double complex[:, :] ret = np.zeros((M, N), dtype=np.complex128)
    cdef double z = k * r
    cdef double complex tmp
    cdef int l, i, j
    cdef double[:] leg = np.zeros(trans_order + 1, dtype=np.float64)
    cdef double complex[:] sphkl = np.zeros(trans_order + 1, dtype=np.complex128)
    
    sphkl = sph_hankel2_np(np.arange(trans_order + 1), z)
    
    for i in range(M):
        for j in range(N):

            tmp = 0j
        
            leg = eval_legendre(np.arange(trans_order + 1), cos_angle[i, j])
            
            for l in range(trans_order + 1):

                tmp += (2 * l + 1) * (1j ** l) * sphkl[l] * leg[l]

            ret[i, j] = tmp

    return np.asarray(ret)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray ff2nf_op2(double r, double[:,:] cos_angle, double k, int trans_order):
    '''
    Standard far-field to near-field translation operator (slower version).
    '''
    cdef int M = cos_angle.shape[0]
    cdef int N = cos_angle.shape[1]
    cdef double complex[:, :] ret = np.zeros((M, N), dtype=np.complex128)
    cdef double z = k * r
    cdef double complex tmp
    cdef int l, i, j

    for i in range(M):
        for j in range(N):

            tmp = 0j

            for l in range(trans_order + 1):

                tmp += (2 * l + 1) * (1j ** l) * sph_hankel2(l, z) * eval_legendre(l, cos_angle[i, j])

            ret[i,j] = tmp

    return np.asarray(ret)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray mod_ff2nf_op(double r, double[:, :] cos_angle, double k, int trans_order):
    '''
    Bandlimited modified far-field to near-field translation operator.
    '''
    cdef int kdir_dim1 = cos_angle.shape[0]
    cdef int kdir_dim2 = cos_angle.shape[1]
    cdef int theta_order = (kdir_dim1 - 1) // 2
    cdef int phi_order = (kdir_dim2 - 1)
    cdef np.ndarray translation, sinabs, inter1, inter2

    translation = ff2nf_op(r, cos_angle, k, trans_order)
    sinabs = bandlimited_abs_sin(trans_order + theta_order)

    if phi_order > trans_order:
        inter1 = fft_interpolate(translation, trans_order + theta_order,
            phi_order)
    else:
        inter1 = fft_interpolate_theta(translation, trans_order + theta_order)

    inter2 = inter1 * sinabs[...,None]

    return fft_filter(0.5 * inter2, kdir_dim1, kdir_dim2)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray bandlimited_abs_sin(int deg):
    '''
    Calculates a bandlimited |sin(theta)| based on its Fourier series.
    '''
    cdef np.ndarray res
    fseries = np.zeros(2 * deg +  1, dtype=np.complex128)
    cdef np.ndarray n = np.arange(-deg, deg + 1)

    if deg % 2 == 0:
        fseries[::2] = 2 / pi / (1 - n[::2] ** 2)
    else:
        fseries[1::2] = 2 / pi / (1 - n[1::2] ** 2)

    res = fftshift(ifft(ifftshift(np.asarray(fseries))))
    res *= res.size

    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray ff2ff_op(double r, double[:, :] cos_angle, double k):
    '''
    Far-field to far-field shift operator.
    '''
    cdef int M = cos_angle.shape[0]
    cdef int N = cos_angle.shape[1]
    cdef double complex[:, :] shifter = np.zeros((M, N), dtype=np.complex128)
    cdef int i, j

    for i in range(M):
        for j in range(N):
            shifter[i, j] = cexp(1j * k * r * cos_angle[i, j])

    return np.asarray(shifter)


## INTERPOLATION AND FILTERING (ANTERPOLATION) FUNCTIONS ##
'''
Implements FFT-based interpolation and filtering based on the following papers:
    
[1] Sarvas, Jukka. "Performing interpolation and anterpolation entirely by fast 
Fourier transform in the 3-D multilevel fast multipole algorithm." SIAM Journal 
on Numerical Analysis 41.6 (2003): 2180-2196.

[2] Cecka, Cris, and Eric Darve. "Fourier-based fast multipole method for the 
Helmholtz equation." SIAM Journal on Scientific Computing 35.1 (2013): A79-A103.
'''

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef np.ndarray half_fft2(np.ndarray x):
    '''
    Half FFT using spherical property.
    '''
    cdef int M = x.shape[0]
    cdef int N = x.shape[1]

    v = fftshift(fft(x, axis=0), axes=0)
    dummy = np.flipud(v).copy()
    v = np.concatenate((v, dummy), axis=1)

    w = fftshift(fft(v[:(M - 1) / 2 + 1, :], axis=1), axes=1)
    dummy = np.flipud(w[:-1, :]).copy()

    n = (-1) ** np.arange(-N, N, dtype=np.float64)
    dummy *= n[None, ...]

    return np.concatenate((w, dummy), axis=0)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray half_ifft2(np.ndarray x):
    '''
    Half IFFT using spherical property.
    '''
    cdef int M = x.shape[0]
    cdef int N = x.shape[1]

    return ifft2(ifftshift(x))[:,:N/2]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray fft_interpolate(np.ndarray coeff, int kdir_dim1, int kdir_dim2):
    '''
    FFT-based interpolation (memory optimized version).
    '''
    cdef int M1 = coeff.shape[0]
    cdef int N1 = coeff.shape[1]
    cdef int M2 = kdir_dim1
    cdef int N2 = kdir_dim2
    cdef int padM, padN

    padM = (M2 - M1) // 2
    padN = (N2 - N1)

    #spectrum1 = half_fft2(coeff)
    spectrum1 = half_fft2(ifftshift(coeff, axes=0))
    spectrum2 = np.pad(spectrum1, ((padM, padM),(padN, padN)),
        mode='constant')

    newcoeff = M2 * N2 / (<double> M1 * N1) * half_ifft2(spectrum2)
    newcoeff = fftshift(newcoeff, axes=0)

    return newcoeff


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray fft_interpolate_theta(np.ndarray coeff, int new_order):
    '''
    FFT-based interpolation for theta direction only.
    '''
    cdef int M1 = coeff.shape[0]
    cdef int N1 = coeff.shape[1]
    cdef int M2 = 2 * (new_order) + 1

    padM = (M2 - M1) // 2

    v = fftshift(fft(ifftshift(coeff, axes=0), axis=0), axes=0)
    dummy = np.flipud(v).copy()

    spectrum1 = np.concatenate((v, dummy), axis=1)
    spectrum2 = np.pad(spectrum1, ((padM, padM),(0, 0)), mode='constant')

    newcoeff = M2 / (<double> M1) * ifft(ifftshift(spectrum2, axes=0), axis=0)[:, :N1]
    newcoeff = fftshift(newcoeff, axes=0)

    return newcoeff


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray fft_filter(double complex[:,:] coeff, int kdir_dim1, int kdir_dim2):
    '''
    FFT-based filtering (memory optimized version).
    '''
    cdef int M1 = coeff.shape[0]
    cdef int N1 = coeff.shape[1]
    cdef int M2 = kdir_dim1
    cdef int N2 = kdir_dim2
    cdef int Mstart, Mstop, Nstart, Nstop

    Mstart = (M1 - M2) / 2
    Mstop = Mstart + M2
    Nstart = (N1 - N2)
    Nstop = Nstart + 2 * N2

    #spectrum1 = half_fft2(coeff)
    spectrum1 = half_fft2(ifftshift(coeff, axes=0))
    spectrum2 = spectrum1[Mstart:Mstop, Nstart:Nstop]

    newcoeff = M2 * N2 / (<double> M1 * N1) * half_ifft2(spectrum2)
    newcoeff = fftshift(newcoeff, axes=0)

    return newcoeff


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict fft_quadrule(int theta_order, int phi_order):
    '''
    Trapezoidal quadrature rule in theta and phi for integration over a unit
    sphere (memory optimized version).
    '''
    cdef int M, N
    cdef int ntheta, nphi

    # 1: theta/polar
    M = 2 * theta_order + 1
    theta = np.linspace(-theta_order * 2 * pi / M, theta_order * 2 * pi / M, M)
    thetaweights = np.ones(M) * 2 * pi / M

    # 2: phi/azimuthal
    N = phi_order + 1
    phi = np.linspace(-pi, 0, N, endpoint=False)
    phiweights = np.ones(N) * 2 * pi / N

    weights = thetaweights[:, None].dot(phiweights[None, :])

    ntheta = theta.shape[0]
    nphi = phi.shape[0]

    ktheta, kphi = np.meshgrid(theta, phi, indexing='ij')
    kdir = np.concatenate((ktheta[..., None], kphi[..., None]), axis=2)

    kdir[:(M - 1) / 2, :, 0] *= -1
    kdir[:(M - 1) / 2, :, 1] += pi

    kcoord = dir2coord(kdir)
    kcoordT = kcoord.transpose((0, 2, 1))

    quadrule = dict()
    quadrule['kdir'] = kdir
    quadrule['kcoord'] = kcoord
    quadrule['kcoordT'] = kcoordT
    quadrule['weights'] = weights
    quadrule['theta'] = theta
    quadrule['phi'] = phi
    quadrule['theta_order'] = theta_order
    quadrule['phi_order'] = phi_order
    quadrule['ntheta'] = ntheta
    quadrule['nphi'] = nphi
    quadrule['theta_weights'] = thetaweights
    quadrule['phi_weights'] = phiweights

    return quadrule


## UTILITY FUNCTIONS ##

def get_unique_coords(dims=2):

    if dims == 2:

        x, y, z = np.mgrid[1:4, 0:4, 0:1:1j]
        unique_coords = np.c_[x.ravel(), y.ravel(), z.ravel()].astype(int)
        unique_coords = unique_coords[2:, :]

    elif dims == 3:
        unique_coords = None

    return unique_coords