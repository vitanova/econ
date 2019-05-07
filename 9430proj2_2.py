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
sigma = 2;
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
def chebyshev(k, order_cheby, kmin, kmax):
    # given the value of k
    # get the value of first oder_cheby functions at k
    T = np.linspace(0, order_cheby, order_cheby+1)
    x = 2*(k-kmin)/(kmax - kmin) -1
    return np.cos(T*np.arccos(x))
def cheby_approx(theta, k, order_cheby, kmin, kmax):
    # given the coefficient theta and above chebyshev
    # find the approximated value of f(k)
    res = chebyshev(k, order_cheby, kmin, kmax)
    return np.dot(theta, res)
@jit(nopython=True)
def col_points(a, b, n):
    # the collocation points
    # different from evenly distributed K
    temp = (np.linspace(n, 1, n)*2-1)/(2*n) * np.pi
    k = np.cos(temp)
    return (k+1)*(b-a)/2+a
KK = col_points(kmin, kmax, grid)
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

def residual(theta, order_cheby):
    R  = -999*np.ones((N_z, grid))
    #Kprime = np.ones((N_z, grid))
    MUprime = np.ones((N_z, grid))
    for iz in range(N_z):
        for ik in range(grid):
            l = cheby_approx(theta[iz], KK[ik], order_cheby, kmin, kmax)
            c = nu/(1-nu) * (1-alpha) *A*np.exp(ZZ[iz])*KK[ik]**alpha *l**(-alpha)*(l-1)
            kprime = A*np.exp(ZZ[iz])*KK[ik]**alpha*l**(1-alpha)+(1-delta)*KK[ik]-c
            kprime = max(kmin, min(kprime, kmax))
            #Kprime[iz, ik]=kprime
            mu = (c**nu*(1-l)**(1-nu))**(-sigma)*nu*c**(nu-1)*(1-l)**(1-nu)
            muprime = 0
            for izprime in range(N_z):
                lprime = cheby_approx(theta[izprime], kprime, order_cheby, kmin, kmax)
                cprime = nu/(1-nu) * (1-alpha) *A*np.exp(ZZ[izprime])*kprime**alpha *lprime**(-alpha)*(lprime-1)
                cprime = max(0.001, cprime)
                print(iz, ik, izprime, kprime, lprime, cprime)
                muprime_z1 = (cprime**nu*(1-lprime)**(1-nu))**(-sigma)*nu*cprime**(nu-1)*(1-lprime)**(1-nu)
                muprime_z2 = alpha*A*np.exp(ZZ[izprime])*kprime**(alpha-1)*lprime**(1-alpha) + 1 - delta
                muprime_z = muprime_z1 * muprime_z2
                muprime += beta * muprime_z * PPi[iz, izprime]
            MUprime[iz, ik] = muprime
            R[iz, ik] = mu - muprime
    return R, MUprime

@jit(nopython=True)
def get_the_policy(theta, order_cheby):
    kprime_implied = np.zeros((N_z, grid))
    for iz in range(N_z):
        for ik in range(grid):
            T = np.linspace(0, order_cheby, order_cheby+1)
            x = 2*(KK[ik]-kmin)/(kmax - kmin) -1
            res = np.cos(T*np.arccos(x))
            kprime = np.dot(theta[iz], res)
            kprime_implied[iz, ik] = kprime
    return kprime_implied

@jit(nopython=True)
def get_value_function(policy):
    v_old  = -99*np.ones((N_z, grid))
    v_new  = v_old.copy()
    n = 0
    while n<=max_iter+1:
        error = 0
        for iz in range(N_z):
            for ik in range(grid):
                v_max_so_far = -99
                kprime = policy[iz, ik]
                ikprime = np.argmax(KK>kprime)
                l = labor_solve(KK[ik], ZZ[iz], kprime)
                c = A*np.exp(ZZ[iz])*KK[ik]**alpha*l**(1-alpha)+(1-delta)*KK[ik]-kprime
                if c>0:
                    v_zprime_kprime = 0
                    for izprime in range(N_z):
                        v_zprime_kprime += v_old[izprime, ikprime]*PPi[iz, izprime]
                    v_new[iz, ik] = (1 - beta)*(c**nu*(1 - l)**(1 - nu))**(1-sigma)/(1 - sigma) + beta*v_zprime_kprime
                temp_err = v_old[iz, ik]-v_new[iz, ik]
                temp_err1 = v_new[iz, ik]-v_old[iz, ik]
                if temp_err1 > temp_err:
                    temp_err = temp_err1
                if temp_err > error:
                    error = temp_err
        # print(n, error)
        if error < tol:
            return v_new
            break
        else:
            v_old = v_new.copy()
            n = n+1   
            
@jit(nopython=True)
def get_policy_function(v):
    ikprime_new = np.zeros((N_z, grid))
    kprime_new = np.zeros((N_z, grid))
    for iz in range(N_z):
        for ik in range(grid):
            v_max_so_far = v[iz, ik]
            ik_max_so_far = 0
            if ik ==0:
                ikprime_start = 0
            else:
                ikprime_start = int(ikprime_new[iz, ik-1])
            # use this to reduce cells for search
            for ikprime in range(ikprime_start, grid):
                l = labor_solve(KK[ik], ZZ[iz], kprime)
                c = A*np.exp(ZZ[iz])*KK[ik]**alpha*l**(1-alpha)+(1-delta)*KK[ik]-kprime
                if c>0:
                    v_z_kprime = 0
                    for izprime in range(N_z):
                        v_z_kprime += v[izprime, ikprime] * PPi[iz, izprime]
                    v_z_k = (1-beta)*(c**nu*(1 - l)**(1 - nu))**(1-sigma)/(1 - sigma) + beta*v_z_kprime
                    if v_z_k > v_max_so_far:
                        v_max_so_far = v_z_k
                        ikprime_so_far = ikprime
                    # exploit value function concavity
                    else:
                        break
            ikprime_new[iz, ik] = int(ikprime_so_far)
            kprime_new[iz, ik] = KK[int(ikprime_so_far)]
    return kprime_new

def get_theta(policy, order_cheby):
    XX = np.zeros((N_z, grid, order_cheby+1))
    theta = np.zeros((N_z, order_cheby+1))
    for iz in range(N_z):
        Y = policy[iz]
        for ik in range(grid):
            T = np.linspace(0, order_cheby, order_cheby+1)
            x = 2*(KK[ik]-kmin)/(kmax - kmin) -1
            res = np.cos(T*np.arccos(x))
            XX[iz, ik] = res
        X = XX[iz]
        coef = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), Y))
        theta[iz] = coef
    return theta

def get_diff_policy(pol0, pol1):
    R = np.zeros((N_z, grid))
    for iz in range(N_z):
        for ik in range(grid):
            R[iz, ik] = abs(pol1[iz, ik] - pol0[iz, ik])
    return R.sum()

def cheby_method(theta_0, order_cheby):
    lam = 0.5
    n = 0
    theta_old = theta_0.copy()
    while n < 100:
        policy_old    = get_the_policy(theta_old, order_cheby)
        v             = get_value_function(policy_old)
        policy_new    = get_policy_function(v)
        err = get_diff_policy(policy_old, policy_new)
        if  err < 1e-5:
            print('success')
            return theta_0
            break
        else:
            theta_new = get_theta(policy_new, order_cheby)
            print(n, err, theta_old, theta_new)
            theta_update = lam*theta_new + (1-lam)*theta_old
            if 
            theta_old= theta_update.copy()
            n = n+1
order_cheby=2
theta_0 = np.zeros((N_z, order_cheby+1))
for iz in range(N_z):
    for ic in range(order_cheby+1):
        theta_0[iz, ic] = np.random.uniform(0, 1)
cheby_method(theta_0, order_cheby)