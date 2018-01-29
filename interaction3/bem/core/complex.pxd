## interaction / complex.pxd

cdef extern from '<complex.h>' nogil:
    double cabs(double complex)
    double carg(double complex)
    double complex conj(double complex)
    double complex cexp(double complex)
    double creal(double complex)
    double cimag(double complex)
    
    