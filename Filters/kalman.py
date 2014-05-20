import numpy as np

mu_0 = np.zeros((1,6))
sigma_0 = np.eye(6)*np.array([4,4,2,2,1,1])
F = np.array(([1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0],[0,0,0,0,0,1]))
sigma_x = np.eye(6)
H = np.array(([1,0,0,0,0,0],[0,1,0,0,0,0]))
sigma_z = np.array(([2,0],[0,2]))


print(F)
print(H)

