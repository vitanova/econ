import matplotlib.pyplot as plt
import quantecon as qe 
# all packages is installed if choose anaconda solution except quantecon
# which can be obtained by pip install commend
import numpy as np
from numba import jit
import pandas as pd
from datetime import datetime as dt

# idea: first use Aiyagari to get a guess for initial wealth distribution
# then, choose a time span and initial guess of coefficient of the regression
# simulate the path for aggregate capital K
# obtain new coefficient with OLS regression
# update until converge
beta = .96
alpha = .36
delta = .1
sigma = 2
n_y = 4
y_grid = [.1, .8, 1.2, 10]
y_trans = [[.5, .5, 0, 0],
         [.05, .9, .05, 0],
         [.02, .04, .93, .01],
         [0, 0, .8, .2]]
y_grid= np.asarray(y_grid)
y_trans = np.asarray(y_trans)
n_a = 501
a_grid = np.linspace(0, 50, n_a)

y_0 = np.ones(4)/4
y_old = y_0
for i in range(100):
    y_new = np.dot(y_old, y_trans)
    if max(abs(y_new-y_old))<1e-8:
        break
    else:
        y_old = y_new

L = np.dot(y_new, y_grid)

tol_V = 1e-6
tol_r = 1e-5
tol_phi = 1e-5 / (n_y * n_a)

v_0= np.zeros((n_y, n_a))

phi_0 = np.zeros((n_y, n_a))
phi_0[0, 0] = 0.5
phi_0[1, 0]= 0.5

r_0 = 0.75 * (1/beta - 1)
r_min = -delta
r_max = 1/beta -1

max_iter = 10000
tau = 0

@jit(nopython=True)
def VFI_monotonicity_concavity(r, v, tau):
    K = (alpha/(r+delta))**(1/(1-alpha))*L
    w = (1 - alpha)*(K/L)**alpha
    v_old=v.copy()
    iaprime_old = np.zeros((n_y, n_a))
    v_new = np.zeros((n_y, n_a))
    iaprime_new = np.zeros((n_y, n_a))
    n=0
    # start loop
    while n<=max_iter:
        error = 0
        for iy in range(n_y):
            for ia in range(n_a):
                v_max_so_far = -99999
                ia_max_so_far = 0
                # exploit policy function monotonicity
                if ia ==0:
                    iaprime_start = 0
                else:
                    iaprime_start = iaprime_old[iy, ia-1]
                # use this to reduce cells for search
                for iaprime in range(iaprime_start, n_a):
                    c = w*y_grid[iy] + (1 + (1-tau)*r)*a_grid[ia] + tau*r*K - a_grid[iaprime]
                    if c>0:
                        v_y_kprime = 0
                        for iyprime in range(n_y):
                            v_y_kprime += v_old[iyprime, iaprime] * y_trans[iy, iyprime]
                        #v_y_a = (1-beta)*c**(1-sigma)/(1-sigma) + beta*v_y_kprime
                        v_y_a = c**(1-sigma)/(1-sigma) + beta*v_y_kprime
                        if v_y_a > v_max_so_far:
                            v_max_so_far = v_y_a
                            iaprime_so_far = iaprime
                        # exploit value function concavity
                        else:
                            break
                # update value and policy function
                v_new[iy, ia] = v_max_so_far
                iaprime_new[iy, ia] = iaprime_so_far
                # now errors
                temp_err = v_old[iy, ia]-v_new[iy, ia]
                temp_err1 = v_new[iy, ia]-v_old[iy, ia]
                if temp_err1 > temp_err:
                    temp_err = temp_err1
                if temp_err > error:
                    error = temp_err
        #print(n, error)
        if error < tol_V:
            return n, v_new, iaprime_new
            break
        else:
            v_old = v_new.copy()
            iaprime_old = iaprime_new.copy()
            n = n+1

@jit(nopython=True)
def ss_phi(phi_0, iaprime_new):
    phi_old = phi_0.copy()
    n = 0
    while n < 10000:
        phi_new = np.zeros((n_y, n_a))
        for iy in range(n_y):
            for ia in range(n_a):             
                iaprime = iaprime_new[iy, ia]
                for iyprime in range(n_y):
                    phi_new[iyprime, iaprime] += phi_old[iy, ia] * y_trans[iy, iyprime]
        err = np.max(phi_new - phi_old)
        if err < tol_phi:
            return phi_new
            break
        else:
            #print(n, err)
            phi_old = phi_new.copy()
            n = n + 1

def ss_r(r_0, v_0, phi_0, tau):
    r_old = r_0
    n = 0
    r_min = -delta
    r_max = 1/beta -1
    while n < 100:
        res = VFI_monotonicity_concavity(r_old, v_0, tau)
        iaprime_new = res[2].astype('int')
        ress = ss_phi(phi_0, iaprime_new)
        Ks = np.dot(np.sum(ress, axis=0), a_grid)
        r_new = alpha*Ks**(alpha - 1)*L**(1 - alpha) - delta
        if r_new > r_old:
            r_min = max(r_min, r_old)
        else:
            r_max = min(r_max, r_old)
        r_update = max(min(0.5*r_old + 0.5*r_new, 0.5*r_old + 0.5*r_max), 0.5*r_old + 0.5*r_min)
        err = abs(r_new - r_old)
        if err < tol_r:
            return res, ress, r_old
            break
        else:
            print(n, r_new, r_old, err)
            r_old = r_update
            n = n+1
qe.util.tic()
res_all = ss_r(r_0, v_0, phi_0, tau)
print(res_all)
qe.util.toc()

# use it as the initial wealth distribution
print("   initial guess: ")
print(res_all[1])

# now the KS algorithm
K_init = np.dot(res_all[1].sum(axis=0), a_grid)
n_K = 101
K_min = 0.9*K_init
K_max = 1.1*K_init
K_grid = np.linspace(K_min, K_max, n_K)

z_grid = [-.05, 0, .05]
z_trans = [[.88, .08, .04],
          [.06, .88, .06],
          [.04, .08, .88]]
n_z = 3
z_grid = np.asarray(z_grid)
z_trans = np.asarray(z_trans)
A = np.array([[0.1, 0.9],
              [0.1, 0.9],
              [0.1, 0.9]])


v_0 = np.zeros((n_z, n_K, n_y, n_a))
@jit(nopython=True)
def VFI_KS(v, A):
    v_old = v.copy()
    v_new = np.zeros((n_z, n_K, n_y, n_a))
    iaprime_old = np.zeros((n_z, n_K, n_y, n_a))
    iaprime_new = np.zeros((n_z, n_K, n_y, n_a))
    n = 0
    # start loop
    while n <= max_iter:
        error = 0
        for iz in range(n_z):
            for iK in range(n_K):
                r = np.exp(z_grid[iz])*alpha*K_grid[iK]**(alpha - 1)*L**(1 - alpha) - delta
                w = np.exp(z_grid[iz])*(1 - alpha)*(K_grid[iK]/L)**alpha
                Kprime = np.exp(A[iz, 0]+ A[iz, 1]*K_grid[iK])
                iKprime = max(0, min(int((Kprime-K_min)/(K_max -K_min)*n_K), n_K-1))
                for iy in range(n_y):
                    for ia in range(n_a):
                        v_max_so_far = -99999
                        ia_max_so_far = 0
                        # exploit policy function monotonicity
                        if ia ==0:
                            iaprime_start = 0
                        else:
                            iaprime_start = int(iaprime_old[iz, iK, iy, ia-1])
                        # use this to reduce cells for search
                        for iaprime in range(iaprime_start, n_a):
                            c = w*y_grid[iy] + (1 + r)*a_grid[ia] - a_grid[iaprime]
                            if c > 0:
                                v_y_kprime = 0
                                for izprime in range(n_z):
                                    for iyprime in range(n_y):
                                        iaprime, iyprime = int(iaprime), int(iyprime)
                                        v_y_kprime += v_old[izprime,iKprime, iyprime, iaprime] * z_trans[iz, izprime] * y_trans[iy, iyprime]
                                v_y_a = (1-beta)*c**(1-sigma)/(1-sigma) + beta*v_y_kprime
                                if v_y_a > v_max_so_far:
                                    v_max_so_far = v_y_a
                                    iaprime_so_far = iaprime
                                # exploit value function concavity
                                else:
                                    break
                        v_new[iz, iK, iy, ia] = v_max_so_far
                        iaprime_new[iz, iK, iy, ia] = iaprime_so_far
                        # now errors
                        temp_err = v_old[iz, iK, iy, ia]-v_new[iz, iK, iy, ia]
                        temp_err1 = v_new[iz, iK, iy, ia]-v_old[iz, iK, iy, ia]
                        if temp_err1 > temp_err:
                            temp_err = temp_err1
                        if temp_err > error:
                            error = temp_err
        if n%40 ==0:
            print(n, error)
        if error < tol_V:
            return n, v_new, iaprime_new
            break
        else:
            v_old = v_new.copy()
            iaprime_old = iaprime_new.copy()
            n = n+1  

T = 1000
mc = qe.MarkovChain(z_trans)
Z_path = mc.simulate(ts_length=T)
for i in range(100):
    if Z_path[0] != 1:
        Z_path = mc.simulate(ts_length=T)
    else:
        break

@jit(nopython=True)
def find_coef(phi, iaprime_new):
    K_path = np.zeros(T+1)
    iK_path = np.zeros(T+1)
    phi_path = np.zeros((T+1, n_y, n_a))
    K_path[0] = K_init
    iK_path[0] = max(0, min(int((K_init-0.95*K_init)/(0.1*K_init)*n_K), n_K-1))
    phi_path[0] = phi
    for t in range(T):
        iz = int(Z_path[t])
        iK = int(iK_path[t])
        phi_old = phi_path[t]
        phi_new = np.zeros((n_y, n_a))
        iaprime_new_z_K = iaprime_new[iz, iK]
        for iy in range(n_y):
            for ia in range(n_a):             
                iaprime = int(iaprime_new_z_K[iy, ia])
                for iyprime in range(n_y):
                    phi_new[iyprime, iaprime] += phi_old[iy, ia] * y_trans[iy, iyprime]
        phi_path[t+1] = phi_new
        Kprime = np.dot(np.sum(phi_new, axis=0), a_grid)
        iKprime = max(0, min(int((Kprime-0.75*K_init)/(0.5*K_init)*n_K), n_K-1))
        iK_path[t+1] = iKprime
        K_path[t+1] = K_grid[iKprime]
    X = np.zeros((T, 6))
    Y = np.zeros(T)
    for t in range(T):
        X[t, 0] = 1
        X[t, 1] = (Z_path[t]==1)
        X[t, 2] = (Z_path[t]==2)
        X[t, 3] = np.log(K_path[t])
        X[t, 4] = (Z_path[t]==1) * np.log(K_path[t])
        X[t, 5] = (Z_path[t]==2) * np.log(K_path[t])
        Y[t] = np.log(K_path[t+1])
    coef = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), Y))
    A_new = np.array([[coef[0], coef[3]],
                        [coef[0]+coef[1], coef[3]+ coef[4]],
                        [coef[0]+coef[2], coef[3]+ coef[5]]])
    return A_new, K_path, phi_path

lam = 0.5
def update(A):
    A_old = A.copy()
    n = 0
    while n <= 100:
        error = 0
        res = VFI_KS(v_0, A_old)
        A_new, K_path, phi_path = find_coef(phi=res_all[1],iaprime_new = res[2])
        A_update = (1-lam)*A_old+ lam*A_new
        for iz in range(n_z):
            for i in range(2):
                temp_err = A_new[iz, i] - A_old[iz, i]
                temp_err1 = A_old[iz, i] - A_new[iz, i]
                if temp_err1 > temp_err:
                    temp_err = temp_err1
                if temp_err > error:
                    error = temp_err
        print(n, error, A_old)
        if error < 1e-5:
            return n, res, A_old, K_path
            break
        else:
            A_old = A_update.copy()
            n = n+1
res_KS = update(A)

# save the data
df = pd.DataFrame(res_KS[3], columns=['k'])
df = df[:1000]
df['z'] = Z_path
df.to_csv("yyy.csv")

K_1000 = res_KS[3]
Y_1000 = np.zeros(1000)
I_1000 = np.zeros(1000)
C_1000 = np.zeros(1000)
for i in range(1000):
    Y_1000[i] = np.exp(z_grid[Z_path[i]])*K_1000[i]**0.36*L**0.64
    I_1000[i] = K_1000[i+1] - (1-delta)*K_1000[i]
    C_1000[i] = Y_1000[i] - I_1000[i]

fig, axes = plt.subplots(3, 1, figsize=(8, 15))
LIST = [Y_1000, C_1000, I_1000]
names = ['output', 'consumption', 'investment']
for i in range(3):
    axes[i].plot(LIST[0], label=names[0])
    axes[i].plot(LIST[i], label=names[i])
    axes[i].set_xlim(0, 1000)
    axes[i].set_ylim(0, 2)
    axes[i].set_title(names[i])
    axes[i].legend()
plt.savefig('HW302.pdf', dpi=250)
plt.show()

Sim_data1 = np.asarray([Y_1000, C_1000, I_1000])
Sim_data1 = Sim_data1.T
np.savetxt('HW302'+'.txt', Sim_data1, delimiter=',')
print('now go to matlab!')