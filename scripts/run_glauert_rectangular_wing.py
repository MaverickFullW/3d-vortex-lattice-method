import numpy as np


AR = 6.0
r = 8
m = 2.0 * np.pi
alpha0 = 0.0

n = np.array([1, 3, 5, 7])
theta = np.arange(1, 5) * np.pi / r

matrix = np.zeros((4, 4))

for i, theta_i in enumerate(theta):
    for j, n_j in enumerate(n):
        matrix[i, j] = (
            4.0 * AR / m + n_j / np.sin(theta_i)
        ) * np.sin(n_j * theta_i)

rhs = np.ones(4)
A = np.linalg.solve(matrix, rhs)
CL_alpha = np.pi * AR * A[0]

print("collocation angles [deg]:")
print(np.rad2deg(theta))
print("matrix:")
print(matrix)
print("A1/alpha, A3/alpha, A5/alpha, A7/alpha:")
print(A)
print("CL / alpha [1/rad]:", CL_alpha)
print("McBain reference CL / alpha = 4.5273 1/rad")
