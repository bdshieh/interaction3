## bem / core / fma_functions.pxd
'''
Header file for 'interaction / functions.pyx'.
Author: Bernard Shieh (bshieh@gatech.edu)
'''
cimport numpy as np

#cdef double pi = np.pi
cpdef double mag(double[:])
cpdef np.ndarray distance(double[:, :], double[:, :])
cpdef np.ndarray direct_eval(double complex[:], double[:, :], double, double, double)
cpdef np.ndarray ff_coeff(double complex[:], double complex[:, :, :])
cpdef np.ndarray calc_exp_part(double[:, :], double[:], double[:, :, :], double)
cpdef np.ndarray nf_eval(double complex[:, :], double complex[:, :, :], double, double, double)
cpdef double complex sph_hankel2(int, double)
cpdef np.ndarray ff2nf_op(double, double[:, :], double, int)
cpdef np.ndarray mod_ff2nf_op(double, double[:, :], double, int)
cpdef np.ndarray ff2ff_op(double, double[:, :], double)
cpdef np.ndarray half_fft2(np.ndarray)
cpdef np.ndarray half_ifft2(np.ndarray)
cpdef np.ndarray fft_interpolate(np.ndarray, int, int)
cpdef np.ndarray fft_interpolate_theta(np.ndarray, int)
cpdef np.ndarray fft_filter(double complex[:, :], int, int)
cpdef dict fft_quadrule(int, int)
cpdef np.ndarray bandlimited_abs_sin(int)
    
            