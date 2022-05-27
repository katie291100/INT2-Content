import numpy as np

X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
y = np.array([[1, -1, 1, -1]])
def poly_powers(order):
    """
    Returns all powers for a polynomial of a given order.
    For example, if order is 2 (a, b), will return the powers for a and b to make [1, a, b, a^2, ab, b^2]
    """
    return [(k - i, i) for k in range(order + 1) for i in range(k + 1)]

def poly_transform_single(x, k):
    return np.array([((x[0] ** power[0]) * (x[1] ** power[1])) for power in poly_powers(k)])

def transform_polynomial_basis(X, k):

    if k not in range(5):
        raise ValueError("k can only be between 0 and 4")
  #  if X.shape[0] != 2:
    #    raise ValueError("X must be a 2 - dimensional array")
    output = np.zeros((len(poly_powers(k)), X.shape[1]), dtype = float)
    for i in range(output.shape[1]):
        output[:,i] = poly_transform_single(X[:,i], k)
    return output
np.set_printoptions(suppress=True)

print("original data set")
print(X)
print(transform_polynomial_basis(X, 3))

print("For order 0")
print(transform_polynomial_basis(X, 0))
print("For order 1")
print(transform_polynomial_basis(X, 1))
print("For order 2")
print(transform_polynomial_basis(X, 2))
print("For order 3")
print(transform_polynomial_basis(X, 3))
print("For order 4")
print(transform_polynomial_basis(X, 4))
