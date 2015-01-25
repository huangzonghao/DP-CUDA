cdef int cmin(int, int) nogil
cdef int cmax(int, int) nogil
cdef int csum(int[:], int) nogil
cdef void decode(int[:], int, int, int) nogil
cpdef int encode(int[:], int, int) nogil
cpdef int substract(int[:], int, int) nogil
cpdef double revenue(int[:], int, int, int,
                     double, double, double,
                     double, double, double,
                     int, int) nogil
