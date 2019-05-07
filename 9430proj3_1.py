import matplotlib.pyplot as plt
import quantecon as qe 
# all packages is installed if choose anaconda solution except quantecon
# which can be obtained by pip install commend
import numpy as np
from numba import jit
from datetime import datetime as dt
# parameters
beta    = .96
alpha   = .36
delta   = .1
sigma   = 2
# space and transition matrix of y
n_y     = 4
y_grid  = [.1, .8, 1.2, 10]
y_trans = [[.5, .5, 0, 0],
         [.05, .9, .05, 0],
         [.02, .04, .93, .01],
         [0, 0, .8, .2]]
# revert in numpy format         
y_grid= np.asarray(y_grid)
y_trans = np.asarray(y_trans)
# space of a
n_a = 10001
a_grid = np.linspace(0, 100, n_a)

# find ss distribution of y
y_0 = np.ones(4)/4
y_old = y_0
for i in range(100):
    y_new = np.dot(y_old, y_trans)
    if max(abs(y_new-y_old))<1e-8:
        break
    else:
        y_old = y_new
# ss labor supply
L = np.dot(y_new, y_grid)

# tolerances
tol_V = 1e-6
tol_r = 1e-5
tol_phi = 1e-5 / (n_y * n_a)
# initial guess for v
v_0= np.zeros((n_y, n_a))
# initial guess for phi
phi_0 = np.zeros((n_y, n_a))
phi_0[0, 0] = 0.5
phi_0[0, 1]= 0.5
# initial guess for r
r_0 = 0.75 * (1/beta - 1)
r_min = -delta
r_max = 1/beta -1
# maximum number of iteration
max_iter = 10000

# given r, v, and tau, get the value function
# and optimal asset choice
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
                        v_y_a = (1-beta)*c**(1-sigma)/(1-sigma) + beta*v_y_kprime
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

# given phi and iaprime_new,
# get the steady state phi
@jit(nopython=True)
def ss_phi(phi, iaprime_new):
    phi_old = phi.copy()
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

# given r_0, v_0, phi_0, and tau, 
# get the steady state r
def ss_r(r, v, phi, tau):
    r_old = r
    n = 0
    r_min = -delta
    r_max = 1/beta -1
    while n < 100:
        res = VFI_monotonicity_concavity(r_old, v, tau)
        iaprime_new = res[2].astype('int')
        ress = ss_phi(phi, iaprime_new)
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
            print(n, r_update, err)
            r_old = r_update
            n = n+1

tau = 0
qe.util.tic()
res_all = ss_r(r_0, v_0, phi_0, tau)
print(res_all)
qe.util.toc()

identifier = int(dt.timestamp(dt.today()))
# plot the value function, policy functions against z and k
dist_a = res_all[1].sum(axis=0)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(a_grid, dist_a)
ax.set_xlim(0, 100)
plt.savefig(str(identifier)+'HW301.pdf', dpi=250)
plt.show()

mean = np.dot(a_grid, dist_a)
print("       the mean is: ")
print(mean)
cdf = np.cumsum(dist_a)
median = a_grid[np.argmax(cdf>0.5)]
print("       the median is: ")
print(median)
demeaned = a_grid - mean
var = np.dot(demeaned*demeaned, dist_a)
std = var**0.5
print("       the std. dev. is: ")
print(std)

tau = 0.5
qe.util.tic()
res_all1 = ss_r(r_0, v_0, phi_0, tau)
print(res_all1)
qe.util.toc()