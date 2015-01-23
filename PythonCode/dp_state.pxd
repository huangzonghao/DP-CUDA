cdef int cmin(int, int) nogil
cdef int cmax(int, int) nogil
cdef int csum(int[:], int) nogil
cpdef int substract(int[:], int, int) nogil
cpdef double revenue(int[:], int, int, int,
                     double, double, double,
                     double, double, double,
                     int, int) nogil
