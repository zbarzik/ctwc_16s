#!/usr/bin/python
from numpy import zeros
from math import sqrt

def euclidean_dist(a, b):
    return sqrt((a-b)*(a-b))

def string_euclidean_dist(a, b):
    sum  = 0
    for ind in range(0, max(len(a), len(b))):
        c1 = a[ind] if ind < len(a) else '\0'
        c2 = b[ind] if ind < len(b) else '\0'
        sum += (ord(c1)-ord(c2))*(ord(c1)-ord(c2))
    return sqrt(sum)

def precompute_distance(points, dist_func=string_euclidean_dist):
    output = zeros((len(points), len(points)))
    for i in range(0, len(points)):
        for j in range(0, len(points)):
            output[i][j] = dist_func(points[i], points[j])
    return output

def test():
    return precompute_distance(['aaaa', 'baaa','caaa'], dist_func=string_euclidean_dist)
