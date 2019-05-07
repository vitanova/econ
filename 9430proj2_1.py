from scipy.stats import norm
import matplotlib.pyplot as plt
import quantecon as qe # all packages is installed if choose anaconda solution except quantecon
# which can be obtained by pip install commend
import numpy as np
from numba import jit
from numpy.random import normal
from scipy import special
from datetime import datetime as dt


# basic parameters
ky  = 11
iy  = 0.25
rf  = 0.01
y_ss = 0.25
l_ss = 0.33
sigma = 2
beta  = 1/(1 + rf)
delta = iy/ky
alpha = ky*(rf + delta)
A     = (y_ss/l_ss)**(1 - alpha)/((alpha/(rf + delta))**alpha)
k_ss   = (alpha*A/(rf + delta))**(1/(1 - alpha))*l_ss
nu    = 1/(1 + (1 - l_ss)/l_ss*(1 - alpha)/(1 - alpha*delta/(rf + delta)))

kmin = 0.75*k_ss
kmax = 1.25*k_ss
grid = 5001
K    = np.linspace(kmin, kmax, grid)
# iteration criterions
max_iter=10000 # maximum of iteration
tol     =1e-6 # tolerance

# utility function
@jit(nopython=True)
def u(c, l):
    return (c**nu*(1-l)**(1-nu))**(1-sigma)/(1-sigma)


# (c) Tauchen method
def tauchen(mu, sigma_e, rho, lambda_z):
    # no. grid points
    N_z = 2*lambda_z+1
    # value of grid points
    Z = np.asarray([mu+lam*sigma_e/(1-rho**2)**0.5 for lam in range(-lambda_z, lambda_z+1)])
    # mid points
    M = np.asarray([(Z[i]+Z[i+1])/2 for i in range(N_z-1)])
    # transition matrix
    Pi = np.empty((N_z, N_z))
    # fill in probs
    for i in range(N_z):
        for j in range(N_z):
            if j==0:
                Pi[i, j] = special.ndtr((M[j]-(1-rho)*mu-rho*Z[i])/sigma_e)
            elif j<N_z-1:
                Pi[i, j] = special.ndtr((M[j]-(1-rho)*mu-rho*Z[i])/sigma_e) - special.ndtr((M[j-1]-(1-rho)*mu-rho*Z[i])/sigma_e)
            else:
                Pi[i, j] = 1 - special.ndtr((M[j-1]-(1-rho)*mu-rho*Z[i])/sigma_e)
    return Z, Pi
# parameters for stochastic process
lambda_z = 2
N_z      = 2*lambda_z+1
mu       = 0
sigma_e  = 0.007
rho      = 0.95
ZZ, PPi = tauchen(mu, sigma_e, rho, lambda_z)
print('The discrete states are: ')
print(ZZ)
print('Their transition matrix is: ')
print(PPi)

@jit(nopython=True)
def labor_solve(k, z, kprime,n_0=l_ss):
    n=n_0
    it = 0
    while it <= 10000:
        fval = (1 - nu)*(A*np.exp(z)*k**alpha*n**(1 - alpha) + (1 - delta)*k - kprime) - nu*(1 - alpha)*A*np.exp(z)*k**alpha*n**(-alpha)*(1 - n)
        if abs(fval) < 1e-5:
            break
        else:
            fderiv = (1 - alpha)*(1 - nu)*A*np.exp(z)*k**alpha*n**(-alpha) - nu*(1 - alpha)*A*np.exp(z)*k**alpha*(-alpha*n**(-alpha - 1) - (1 - alpha)*n**(-alpha))
            n_new  = max(1e-3,min(1 - 1e-3,n - fval/fderiv))
            if abs(n_new - n) < 1e-5:
                break
            else:
                n = n_new
                it = it + 1
    return n_new         

@jit(nopython=True)
def VFI_monotonicity_concavity(v):
    v_old=v.copy()
    ikprime_old = np.zeros((N_z, grid))
    v_new = np.ones((N_z, grid))
    ikprime_new = np.zeros((N_z, grid))
    opt_l = np.zeros((N_z, grid))
    n=0
    # start loop
    while n<=max_iter:
        error = 0
        for iz in range(N_z):
            for ik in range(grid):
                v_max_so_far = -99999
                ik_max_so_far = 0
                l_max_so_far = l_ss
                # exploit policy function monotonicity
                if ik ==0:
                    ikprime_start = 0
                else:
                    ikprime_start = ikprime_old[iz, ik-1]
                # use this to reduce cells for search
                for ikprime in range(ikprime_start, grid):
                    l = labor_solve(K[ik], ZZ[iz], K[ikprime], l_ss)
                    c = A*np.exp(ZZ[iz])*K[ik]**alpha*l**(1-alpha)+(1-delta)*K[ik]-K[ikprime]
                    if c>0:
                        v_z_kprime = 0
                        for izprime in range(N_z):
                            v_z_kprime += v_old[izprime, ikprime] * PPi[iz, izprime]
                        v_z_k = (1 - beta)*(c**nu*(1 - l)**(1 - nu))**(1-sigma)/(1 - sigma) + beta*v_z_kprime
                        if v_z_k > v_max_so_far:
                            v_max_so_far = v_z_k
                            ikprime_so_far = ikprime
                            l_max_so_far = l
                        # exploit value function concavity
                        else:
                            break
                # update value and policy function
                v_new[iz, ik] = v_max_so_far
                ikprime_new[iz, ik] = ikprime_so_far
                opt_l[iz, ik] = l_max_so_far
                # now errors
                temp_err = v_old[iz, ik]-v_new[iz, ik]
                temp_err1 = v_new[iz, ik]-v_old[iz, ik]
                if temp_err1 > temp_err:
                    temp_err = temp_err1
                if temp_err > error:
                    error = temp_err
        print(n, error)
        if error < tol:
            return n, v_new, ikprime_new, opt_l
            break
        else:
            v_old = v_new.copy()
            ikprime_old = ikprime_new.copy()
            n = n+1

v_0 = np.zeros((N_z, grid))
qe.util.tic()
res = VFI_monotonicity_concavity(v_0)
qe.util.toc()

Kprime_opt = np.zeros((N_z, grid))
for iz in range(N_z):
    for ik in range(grid):
        Kprime_opt[iz, ik] = K[int(res[2][iz][ik])]
identifier = int(dt.timestamp(dt.today()))
# plot the value function, policy functions against z and k
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
for i in range(3):
    for iz in range(5):
        if i==0:
            axes[i].plot(K, res[1][4-iz], label='z=z('+str(4-iz)+')')
            axes[i].set_title("Value Function")
        elif i==1:
            KK = [K[int(ikprime)] for ikprime in res[2][4-iz]]
            axes[i].plot(K, Kprime_opt[4-iz], label='z=z('+str(4-iz)+')')
            axes[i].set_title("Policy Function for Next Period's Capital")
        else:
            axes[i].plot(K, res[3][4-iz], label='z=z('+str(4-iz)+')')
            axes[i].set_title("Policy Function for Labor")
    axes[i].legend()
    axes[i].set_xlabel("Capital")
plt.savefig(str(identifier)+'HW101.pdf', dpi=250)
plt.show()

# Response Impulse Function
Z_path = np.ones(50)*2
Z_path[0] = 3
Z_path = Z_path.astype("int")
def simulate_rif(izpath = Z_path, ik_0 = 2500):
    IK = []
    IK.append(ik_0)
    IZ = izpath
    Z, KK, L, Y, C, I = [], [], [], [], [], []
    # sequence of all relevant variables
    for it in range(0, 49):
        # shock
        if it==0:
            shock = ZZ[IZ[it]]
            Z.append(shock)
        else:
            shock = Z[it-1]*rho
            Z.append(shock)
        # next period capital index
        ikprime = int(res[2][IZ[it], IK[it]])
        IK.append(ikprime)
        # capital
        capital = K[IK[it]]
        KK.append(capital)
        # labor
        labor = labor_solve(capital, shock, K[ikprime])
        L.append(labor)
        # output
        y = A*np.exp(shock)*capital**alpha*labor**(1-alpha)
        Y.append(y)
        # consumption
        cons = y + (1-delta)*capital-K[ikprime]
        C.append(cons)
        # investment
        invest = y - cons
        I.append(invest)
    return [[Z, KK], [L, Y], [C, I]]

Y, L, C, I= simulate_rif()[1][1], simulate_rif()[1][0], simulate_rif()[2][0], simulate_rif()[2][1]
# get the relative change
y_bar, l_bar, c_bar, i_bar = sum(Y)/len(Y), sum(L)/len(L), sum(C)/len(C), sum(I)/len(I)
yy = [(y-y_bar)/y_bar for y in Y]
ll = [(l-l_bar)/l_bar for l in L]
cc = [(c-c_bar)/c_bar for c in C]
ii = [(i-i_bar)/i_bar for i in I]

# plot the latter three sequences against output dynamics
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
LIST = [ii, cc, ll]
names = ['investment', 'consumption', 'labor']
for i in range(3):
    axes[i].plot(yy, label='output')
    axes[i].plot(LIST[i], label=names[i])
    axes[i].set_xlim(0, 50)
    axes[i].set_ylim(-0.1, 0.1)
    axes[i].set_title(names[i])
    axes[i].legend()
plt.savefig(str(identifier)+'HW202.PDF', dpi=250)
plt.show()

# SIMULATE PATH
# generate random sequence of shocks
mc = qe.MarkovChain(PPi)
Z_path = mc.simulate(ts_length=10000)
# a function takes initial capital and shock sequence as given, generate the dynamics
def simulate(izpath = Z_path, ik_0 = 0):
    IK = []
    IK.append(ik_0)
    IZ = izpath
    Z, KK, L, Y, C, I = [], [], [], [], [], []
    # sequence of all relevant variables
    for it in range(0, 10000):
        # shock
        shock = ZZ[IZ[it]]
        Z.append(shock)
        # next period capital index
        ikprime = int(res[2][IZ[it], IK[it]])
        IK.append(ikprime)
        # capital
        capital = K[IK[it]]
        KK.append(capital)
        # labor
        labor = labor_solve(capital, shock, K[ikprime])
        L.append(labor)
        # output
        y = A*np.exp(shock)*capital**alpha*labor**(1-alpha)
        Y.append(y)
        # consumption
        cons = y + (1-delta)*capital-K[ikprime]
        C.append(cons)
        # investment
        invest = y - cons
        I.append(invest)
    return [[Z, KK], [L, Y], [C, I]]

# drop the first 200 obs, and then obtain 500 obs
Y, L, C, I= simulate()[1][1][200:700], simulate()[1][0][200:700], simulate()[2][0][200:700], simulate()[2][1][200:700]
# get the relative change
y_bar, l_bar, c_bar, i_bar = sum(Y)/len(Y), sum(L)/len(L), sum(C)/len(C), sum(I)/len(I)
yy = [(y-y_bar)/y_bar for y in Y]
ll = [(l-l_bar)/l_bar for l in L]
cc = [(c-c_bar)/c_bar for c in C]
ii = [(i-i_bar)/i_bar for i in I]

# plot the latter three sequences against output dynamics
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
LIST = [ii, cc, ll]
names = ['investment', 'consumption', 'labor']
for i in range(3):
    axes[i].plot(yy, label='output')
    axes[i].plot(LIST[i], label=names[i])
    axes[i].set_xlim(0, 500)
    axes[i].set_ylim(-0.5, 0.5)
    axes[i].set_title(names[i])
    axes[i].legend()
plt.savefig(str(identifier)+'HW102.PDF', dpi=250)
plt.show()

# save the data and read it in matlab to conduct business cycle analysis
Sim_data1 = np.asarray([Y, C, I, L])
Sim_data1 = Sim_data1.T
np.savetxt(str(identifier)+'.txt', Sim_data1, delimiter=',')

print('dynamics generated!')

print('now go to matlab!')