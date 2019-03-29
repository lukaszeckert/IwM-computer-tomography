#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
import timeit
from libc.math cimport sin, cos
cimport numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange
cimport openmp


cpdef convolve(double[:] a, double[:] b, int a_len, int b_len):
    cdef double[:] res = np.zeros(a_len)
    cdef int center = b_len / 2
    cdef int m,n
    with nogil:
        for n in range(a_len):
            for m in range(b_len):
                if m + center >= b_len and center - m < 0:
                    break
                if m + center < b_len and n + m < a_len:
                    res[n] += a[n + m] * b[center + m]
                if center - m >= 0 and n - m >= 0 and m > 0:
                    res[n] += a[n - m] * b[center - m]
    return res
cdef double pi = np.pi

cdef struct Line:
    int* rr
    int* cc
    int num_points


cdef void rotate(double[:] pos, double alpha) nogil:
    cdef double x = pos[0]
    cdef double y = pos[1]

    pos[0] = x * cos(alpha) - y * sin(alpha)
    pos[1] = x * sin(alpha) + y * cos(alpha)

cdef void create_detector(double[:] pos, double r) nogil:
    pos[0] = r
    pos[1] = 0

cdef void create_emitters(double[:,:] pos, int n, double r, double width, double pi) nogil:
    cdef int i
    cdef double[2] cur
    cur[0] = r
    cur[1] = 0
    cdef double step = width/n

    with gil:

        rotate(cur, pi - width / 2)
        for i in range(n):
            pos[i,0] = cur[0]
            pos[i,1] = cur[1]
            rotate(cur, step)



cpdef radon_transform(double[:,:] img, int number_steps, int number_emitters, double scan_width):

    cdef double pi, radius = np.sqrt((img.shape[0] / 2) ** 2 + ((img.shape[1] / 2) ** 2))
    cdef double[:] detector_position
    cdef double[:,:] emitters_position, res
    cdef int img_height, img_width
    cdef int i,j

    pi = np.pi
    img_height = img.shape[0]
    img_width = img.shape[1]
    radius = np.sqrt((img.shape[0] / 2) ** 2 + ((img.shape[1] / 2) ** 2))

    detector_position = np.zeros(2, dtype=np.float)
    create_detector(detector_position, radius)

    emitters_position = np.zeros((number_emitters,2), dtype=np.float)
    create_emitters(emitters_position, number_emitters, radius, scan_width, pi)

    res = np.zeros((number_steps, number_emitters), dtype=np.float)


    with nogil:
        for i in range(number_steps):
            for j in range(number_emitters):
                res[i,j] = _radon_projection(detector_position, emitters_position[j], img, img_height, img_width)
            rotate(detector_position, 2*pi/number_steps)
            for j in range(number_emitters):
                rotate(emitters_position[j], 2*pi/number_steps)
    return res

cpdef radon_transform_iterative(double[:,:] img, int number_steps, int number_emitters, double scan_width, int step):
    cdef double pi, radius = np.sqrt((img.shape[0] / 2) ** 2 + ((img.shape[1] / 2) ** 2))
    cdef double[:] detector_position
    cdef double[:, :] emitters_position
    cdef double[:] res
    cdef int img_height, img_width
    cdef int i, j
    pi = np.pi
    img_height = img.shape[0]
    img_width = img.shape[1]
    radius = np.sqrt((img.shape[0] / 2) ** 2 + ((img.shape[1] / 2) ** 2))

    detector_position = np.zeros(2, dtype=np.float)
    create_detector(detector_position, radius)

    emitters_position = np.zeros((number_emitters, 2), dtype=np.float)
    create_emitters(emitters_position, number_emitters, radius, scan_width, pi)

    res = np.zeros((number_emitters), dtype=np.float)
    with nogil:
        for i in range(step):
            rotate(detector_position, 2 * pi / number_steps)
            for j in range(number_emitters):  # , nogil=True):
                rotate(emitters_position[j], 2 * pi / number_steps)
    for j in range(number_emitters):  # , nogil=True):
        res[j] = _radon_projection(detector_position, emitters_position[j], img, img_height, img_width)
    return res

cdef double _radon_projection(double[:] detector, double[:] emitter , double[:,:] img, int height, int width) nogil:
    cdef int *lx,*ly, x_offset, y_offset
    cdef int n,i, x,y,count = 0
    cdef Line line
    cdef double res = 0

    line = _line(<int>detector[0],<int>detector[1], <int>emitter[0], <int>emitter[1])
    lx = line.rr
    ly = line.cc
    n = line.num_points

    x_offset = height / 2
    y_offset = width / 2


    while i < n:
        x = lx[i]
        y = ly[i]
        if x + x_offset < height and y + y_offset < width and x + x_offset >= 0 and y + y_offset >= 0:
                res += img[x+x_offset,y+y_offset]
                count += 1
        i += 1
    free(line.rr)
    free(line.cc)

    return res / count if count > 0 else 0


cpdef iradon_transform(double[:,:] img, int number_steps, int number_emitters, int out_width, int out_height, double scan_width):

    cdef double pi = np.pi
    cdef int i,j
    cdef double[:] det_position = np.zeros(2, dtype=np.float)
    cdef double radius = np.sqrt((<double>out_width / 2) ** 2 + ((<double>out_height / 2) ** 2))
    create_detector(det_position, radius)

    cdef double[:,:] em_positions = np.zeros((number_emitters,2), dtype=np.float)
    create_emitters(em_positions, number_emitters, radius,scan_width, pi)

    cdef int x_offset = number_steps/2
    cdef int y_offset = number_emitters/2
    cdef double[:,:] res = np.zeros((out_height, out_width), dtype=np.float)
    with nogil:
        for i in range(number_steps):
            for j in range(number_emitters):
                _iradon_projection(det_position, em_positions[j], res, out_height, out_width, img[i,j])

            rotate(det_position, 2*pi / number_steps)
            for j in range(number_emitters):  # , nogil=True):
                rotate(em_positions[j], 2*pi / number_steps)
    return res

cpdef iradon_transform_iterative(double[:,:] img, int number_steps, int number_emitters, int out_width, int out_height, double scan_width, int step):

    cdef double pi = np.pi
    cdef int i,j
    cdef double[:] det_position = np.zeros(2, dtype=np.float)
    cdef double radius = np.sqrt((<double>out_width / 2) ** 2 + ((<double>out_height / 2) ** 2))
    create_detector(det_position, radius)

    cdef double[:,:] em_positions = np.zeros((number_emitters,2), dtype=np.float)
    create_emitters(em_positions, number_emitters, radius,scan_width, pi)

    cdef int x_offset = number_steps/2
    cdef int y_offset = number_emitters/2
    cdef double[:,:] res = np.zeros((out_height, out_width), dtype=np.float)
    with nogil:
        for i in range(step):
            rotate(det_position, 2*pi / number_steps)
            for j in range(number_emitters):  # , nogil=True):
                rotate(em_positions[j], 2*pi / number_steps)
        for j in range(number_emitters):
            _iradon_projection(det_position, em_positions[j], res, out_height, out_width, img[step, j])

    return res

cdef void _iradon_projection(double[:] detector, double[:] emitter , double[:,:] img, int height, int width, double value) nogil:
    cdef int *lx,*ly
    cdef int x_offset, y_offset
    cdef int n,i,x,y
    cdef Line line = _line(<int>detector[0],<int>detector[1], <int>emitter[0], <int>emitter[1])
    lx = line.rr
    ly = line.cc
    n = line.num_points
    x_offset = height / 2
    y_offset = width / 2

    for i in range(n):
        x = lx[i]
        y = ly[i]
        if x+x_offset < height and y+y_offset < width and x+x_offset>=0 and y+y_offset>=0:
            img[x+x_offset,y+y_offset] += value#-value/n

    free(line.rr)
    free(line.cc)



cdef Line _line(Py_ssize_t r0, Py_ssize_t c0, Py_ssize_t r1, Py_ssize_t c1) nogil:
    """
    Reimplementation of scipy.draw.line. This implementation ommits gil.
    Generate line pixel coordinates.
    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).
    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    See Also
    --------
    line_aa : Anti-aliased line generator
    """

    cdef char steep = 0
    cdef Py_ssize_t r = r0
    cdef Py_ssize_t c = c0
    cdef Py_ssize_t dr = r1 - r0
    if dr < 0:
        dr = -dr
    cdef Py_ssize_t dc = c1 - c0
    if dc < 0:
        dc = -dc
    cdef Py_ssize_t sr, sc, d, i

    cdef int* rr = <int *>malloc(max(dc+1, dr+1)*sizeof(int))#np.zeros(max(dc, dr) + 1, dtype=np.intp)
    cdef int* cc = <int *>malloc(max(dc+1, dr+1)*sizeof(int))#np.zeros(max(dc, dr) + 1, dtype=np.intp)

    if (c1 - c) > 0:
        sc = 1
    else:
        sc = -1
    if (r1 - r) > 0:
        sr = 1
    else:
        sr = -1
    if dr > dc:
        steep = 1
        c, r = r, c
        dc, dr = dr, dc
        sc, sr = sr, sc
    d = (2 * dr) - dc

    for i in range(dc):
        if steep:
            rr[i] = c
            cc[i] = r
        else:
            rr[i] = r
            cc[i] = c
        while d >= 0:
            r = r + sr
            d = d - (2 * dc)
        c = c + sc
        d = d + (2 * dr)

    rr[dc] = r1
    cc[dc] = c1
    cdef Line res = Line()
    res.cc = cc
    res.rr = rr
    res.num_points = max(dc+1,dr+1)
    return res
