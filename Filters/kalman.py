import numpy as np
import random


mu_0 = [0,0,0,0,0,0]
sigma_0 = np.eye(6)*np.array([4,4,2,2,1,1])
F = np.array(([1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0],[0,0,0,0,0,1]))
sigma_x = np.eye(6)
H = np.array(([1,0,0,0,0,0],[0,1,0,0,0,0]))
sigma_z = np.array(([2,0],[0,2]))
x_0 = np.random.multivariate_normal(mu_0,sigma_0)

mu_t = mu_0
sigma_t = sigma_0
x_t = x_0

for i in range(10):
    predicted_sigma_t_1 = np.dot(np.dot(F,sigma_t),F.T)+sigma_x
    x_t_1 = np.random.multivariate_normal(np.dot(F,x_t), sigma_x)
    z_t_1 = np.random.multivariate_normal(np.dot(H,x_t_1),sigma_z)
    K_t_1 = np.dot(np.dot((predicted_sigma_t_1),H.T),np.linalg.inv(np.dot(np.dot(H,predicted_sigma_t_1),H.T)+sigma_z))
    mu_t_1 = np.dot(F,mu_t)+np.dot(K_t_1,(z_t_1-np.dot(np.dot(H,F),mu_t)))
    sigma_t_1 = np.dot(np.eye(6)-np.dot(K_t_1,H),predicted_sigma_t_1)

    x_t = x_t_1
    sigma_t = sigma_t_1
    mu_t = mu_t_1
    print(x_t)
    print(mu_t)



