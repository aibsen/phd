import time
import numpy as np

size_of_vec = 100000

def pure_python_version():
    t1 = time.time()
    X = range(size_of_vec)
    Y = range(size_of_vec)
    Z = [X[i] + Y[i] for i in range(len(X)) ]
    return time.time() - t1

def numpy_version():
    t1 = time.time()
    X = np.arange(size_of_vec)
    Y = np.arange(size_of_vec)
    Z = X + Y
    return time.time() - t1


def pure_python_version_iterate():
    t1 = time.time()
    X = range(size_of_vec)
    for i in X:
        for j in X:
            pass
    return time.time() - t1

def numpy_version_iterate():
    t1 = time.time()
    X = np.arange(size_of_vec)
    for i in X:
        for j in X:
            pass
    return time.time() - t1


t1 = pure_python_version_iterate()
t2 = numpy_version_iterate()
print(t1, t2)
print("Numpy is in this example " + str(t1/t2) + " faster!")