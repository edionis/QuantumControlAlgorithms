#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: GRAPE_bec_2D.py
# Author: Etienne Dionis
# Date: June 8, 2024
# Description: Optimal control of a Bose-Einstein condensate
#              in a 2D lattice with GRAPE.

#----------------------------------------------------------------------------------
#
# Modules
#
#----------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':20})
rc('text', usetex=True)

#----------------------------------------------------------------------------------
#
# Class System
#
#----------------------------------------------------------------------------------

class BEC(object):
    """
    Bose-Einstein Condensate system.

    Attributes
    ----------
    nmax : int
        N truncation order.
    mmax : int
        M truncation order.
    s : float
        Field amplitude.
    """

    def __init__(self, nmax, mmax, s):
        """Initialize the BEC class."""
        self.nmax = nmax                                                           # N truncation order
        self.mmax = mmax                                                           # M truncation order
        self.s = s                                                                 # Field amplitude 

    def get_Dimension(self) -> int:
        """Dimension of the Hilbert space."""
        return (2 * self.mmax + 1) * (2 * self.nmax + 1)

    def get_mapping(self, m: int, n: int) -> int:
        """Mapping function: map (n, m) -> k."""
        return (m + self.mmax) * (2 * self.nmax + 1) + n + self.nmax

    def get_H(self) -> np.ndarray:
        """Return the total Hamiltonian."""
        Nk = self.get_Dimension()
        H0 = np.zeros((Nk, Nk))                                                    # Diagonal of the dynamic's matrix
        H12p = np.zeros((Nk, Nk), dtype='complex128')                                # Controlled Hamiltonian 12
        H12m = np.zeros((Nk, Nk), dtype='complex128')                                # Controlled Hamiltonian -12
        H23p = np.zeros((Nk, Nk), dtype='complex128')                                # Controlled Hamiltonian 23
        H23m = np.zeros((Nk, Nk), dtype='complex128')                                # Controlled Hamiltonian -23
        H31p = np.zeros((Nk, Nk), dtype='complex128')                                # Controlled Hamiltonian 31
        H31m = np.zeros((Nk, Nk), dtype='complex128')                                # Controlled Hamiltonian -31
        coeff = -0.25 * self.s
        for m in range(-self.mmax, self.mmax + 1):
            for n in range(-self.nmax, self.nmax + 1):
                k = self.get_mapping(m, n)
                H0[k, k] = m**2 + n**2 - m * n
                if n + 1 <= self.nmax:
                    k1 = self.get_mapping(m, n + 1)
                    H12p[k, k1] = coeff
                    H12m[k1, k] = coeff
                    if m + 1 <= self.mmax:
                        k2 = self.get_mapping(m + 1, n + 1)
                        H23p[k2, k] = coeff
                        H23m[k, k2] = coeff
                if m + 1 <= self.mmax:
                    k3 = self.get_mapping(m + 1, n)
                    H31p[k, k3] = coeff
                    H31m[k3, k] = coeff
        H = np.array([H0, H12p, H12m, H23p, H23m, H31p, H31m])
        return H

#----------------------------------------------------------------------------------
#
# Class Propagation
#
#----------------------------------------------------------------------------------

class propagation(object):                                                         # Creating System class
    """
    Propagation of the quantum system.

    Attributes
    ----------
    t : numpy.ndarray
        Integration time.
    u12 : numpy.ndarray
        Control \varphi_{12}.
    u23 : numpy.ndarray
        Control \varphi_{23}.
    u31 : numpy.ndarray
        Control \varphi_{31}.
    H : numpy.ndarray
        Hamiltonian.
    Nk : int
        System's dimension.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray, optional
        Target state.
    """

    def __init__(self, t, u12, u23, u31, H, Nk, psi0, psit=None): 
        """Initialize the Propagation class."""
        self.t     = t                                                             # Integration Time
        self.u12   = u12                                                           # Control
        self.u23   = u23                                                           # Control
        self.u31   = u31                                                           # Control
        self.H     = H                                                             # Hamiltonian
        self.Nk    = Nk                                                            # System's dimension
        self.psi0  = psi0                                                          # Initial state
        self.psit  = psit                                                          # Target state

    def get_propagator(self, u12t, u23t, u31t, dt):
        """This function computes U=exp(-iHdt)."""
        H0 = self.H[0]                                                             # field-free Hamiltonian
        H12p = self.H[1]                                                    
        H12m = self.H[2]                                                    
        H23p = self.H[3]                                                    
        H23m = self.H[4]                                                    
        H31p = self.H[5]                                                    
        H31m = self.H[6]                                                    
        U = linalg.expm(-1j*(H0 + np.exp(1j*u12t)*H12p\
                                + np.exp(-1j*u12t)*H12m\
                                + np.exp(1j*u23t)*H23p\
                                + np.exp(-1j*u23t)*H23m\
                                + np.exp(1j*u31t)*H31p\
                                + np.exp(-1j*u31t)*H31m)*dt)                       # Propagator at each time step 
        return U

    def get_state(self):
        """ State at time t."""
        Nt     = self.t.size                                                       # Number of time points
        C      = np.zeros((self.Nk,Nt), dtype = 'complex128')                        # Initialization of the state
        C[:,0] = self.psi0                                                         # Initial condition
        for n in range(Nt-1):
            dt       = self.t[n+1]-self.t[n]                                       # Time step
            u12t     = self.u12[n]                                                 # Control \varphi_{12}
            u23t     = self.u23[n]                                                 # Control \varphi_{23}
            u31t     = self.u31[n]                                                 # Control \varphi_{31}
            U        = self.get_propagator(u12t, u23t, u31t, dt)                   # Get propagator
            C[:,n+1] = U @ C[:,n]                                                  # Forward propagation at each time step                 
        return C

    def get_adjoint_state(self, psif):
        """ Adjoint state at time t."""
        Nt       = self.t.size                                                     # Number of time points
        rev_time = list(reversed(range(len(self.t))))                              # Backward time
        del(rev_time[-1])                                                          # Delete the first index to compute D(0)
        D        = np.zeros((self.Nk,Nt), dtype = 'complex128')                      # Initialization of adjoint state
        D[:,-1]  = -(self.psit.conj().T @ psif) * self.psit                        # Final condition for the adjoint state
        for n in rev_time:
            dt       = self.t[n]-self.t[n-1]                                       # Time step
            u12t     = self.u12[n]                                                 # Control \varphi_{12}
            u23t     = self.u23[n]                                                 # Control \varphi_{23}
            u31t     = self.u31[n]                                                 # Control \varphi_{31}
            U        = self.get_propagator(u12t, u23t, u31t, dt)                   # Get propagator
            D[:,n-1] = U.conj().T @ D[:,n]                                         # Backward propagation at each time step     
        return D

    def get_fidelity_sm(self):
        """ Fidelity (Square Modulus) at time tf."""
        C = self.get_state()
        F = 1-abs((C[:,-1].conj().T @ self.psit))**2                               # Fidelity 
        return F.real

    def get_fidelity_derivative_sm_pmp(self):
        """ Fidelity at time tf."""
        H0   = self.H[0]                                                           # field-free Hamiltonian
        H12p = self.H[1]                                                    
        H12m = self.H[2]                                                    
        H23p = self.H[3]                                                    
        H23m = self.H[4]                                                    
        H31p = self.H[5]                                                    
        H31m = self.H[6]                                                    
        Nt   = self.t.size                                                         # Number of time points
        dF12 = np.zeros(Nt)                                                        # Initialization of dF tab
        dF23 = np.zeros(Nt)                                                        # Initialization of dF tab
        dF31 = np.zeros(Nt)                                                        # Initialization of dF tab
        C    = self.get_state()                                                    # Get state
        D    = self.get_adjoint_state(C[:,-1])                                     # Get adjoint state
        for n in range(Nt):
            u12t    = self.u12[n]                                                  # Control \varphi_{12}
            u23t    = self.u23[n]                                                  # Control \varphi_{23}
            u31t    = self.u31[n]                                                  # Control \varphi_{31}
            dM12    = 1j * (H12p*np.exp(1j*u12t) - H12m*np.exp(-1j*u12t))          # Derivative of hamiltonian by phi12
            dM23    = 1j * (H23p*np.exp(1j*u23t) - H23m*np.exp(-1j*u23t))          # Derivative of hamiltonian by phi23
            dM31    = 1j * (H31p*np.exp(1j*u31t) - H31m*np.exp(-1j*u31t))          # Derivative of Hamiltonian by phi31
            dF12[n] = (D[:,n].conj().T @ dM12 @ C[:,n]).imag                       # Compute du for each time step given by PMP
            dF23[n] = (D[:,n].conj().T @ dM23 @ C[:,n]).imag                
            dF31[n] = (D[:,n].conj().T @ dM31 @ C[:,n]).imag                
        return dF12, dF23, dF31

    def get_grape(self, maxiter=100, tol=0.999):
        """
        GRAPE (Gradient Ascent Pulse Engineering) algorithm to minimize the fidelity.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations.
        tol : float
            tolerance to stop algorithm.

        Returns
        -------
        u12 : numpy.ndarray
            Optimized control field u12.
        u23 : numpy.ndarray
            Optimized control field u23.
        u31 : numpy.ndarray
            Optimized control field u31.
        fidelity_history : list
            History of the fidelity at each iteration.
        """
        u12, u23, u31 = self.u12.copy(), self.u23.copy(), self.u31.copy()
        J = np.zeros(maxiter)
        J[0] = self.get_fidelity_sm()
        print("iteration = %i, 1-F1 = %f" % (0, 1 - J[0]))
        for i in range(1, maxiter):
            dF12, dF23, dF31 = self.get_fidelity_derivative_sm_pmp()
            
            # Line search for u12
            def Cost_u12(uu):
                self.u12 = uu
                return dyn.get_fidelity_sm()

            def dCost_u12(uu):
                self.u12 = uu
                return self.get_fidelity_derivative_sm_pmp()[0]
            
            ls_u12 = optimize.line_search(Cost_u12, dCost_u12, u12, -dF12)
            epsilon_u12 = ls_u12[0] if ls_u12[0] is not None else 0
            u12 = u12 - epsilon_u12 * dF12
            self.u12 = u12

            # Line search for u23
            def Cost_u23(uu):
                self.u23 = uu
                return self.get_fidelity_sm()

            def dCost_u23(uu):
                self.u23 = uu
                return self.get_fidelity_derivative_sm_pmp()[1]
            
            ls_u23 = optimize.line_search(Cost_u23, dCost_u23, u23, -dF23)
            epsilon_u23 = ls_u23[0] if ls_u23[0] is not None else 0
            u23 = u23 - epsilon_u23 * dF23
            self.u23 = u23

            # Line search for u31
            def Cost_u31(uu):
                self.u31 = uu
                return self.get_fidelity_sm()

            def dCost_u31(uu):
                self.u31 = uu
                return self.get_fidelity_derivative_sm_pmp()[2]
            
            ls_u31 = optimize.line_search(Cost_u31, dCost_u31, u31, -dF31)
            epsilon_u31 = ls_u31[0] if ls_u31[0] is not None else 0
            u31 = u31 - epsilon_u31 * dF31
            self.u31 = u31

            J[i] = self.get_fidelity_sm()

            print("iteration = %i, 1-F1 = %f" % (i, 1 - J[i]))

            if 1 - J[i] > tol:
                break
        return u12, u23, u31, J

#----------------------------------------------------------------------------------
#
# Parameters of the physical system
#
#----------------------------------------------------------------------------------

''' Constant '''
mrb   = 86.909180527*1.66054e-27                                                   # Mass of Rb-87 atom
wl    = 1064e-9                                                                    # Laser wavelength
d     = wl/2.0                                                                     # Spatial period
h     = 1.0545718e-34                                                              # Reduced Plank constant
k_L   = (2.0*np.pi)/d                                                              # Wave vector of the lattice
E_L   = (h*k_L)**2/(2.0*mrb)                                                       # Energy of the lattice
nu_L  = E_L/(2.0*np.pi*h)                                                          # Frequency of the lattice

''' Time '''
Nt      = 400                                                                      # Number of time steps
thold   = 250                                                                      # Final time in µs
tf      = (thold*1.0e-6*E_L)/h                                                     # Normalized final time
t       = np.linspace(0,tf,Nt)                                                     # Normalized time

''' Data '''
nmax  = 4                                                                          # Dimension of the system -> -nmax < k < nmax
mmax  = 4                                                                          # Dimension of the system -> -nmax < k < nmax
s     = 6                                                                          # Lattice depth
Nk    = (2*mmax+1)*(2*nmax+1)                                                      # Dimension of the system

''' Control Phase '''
u12 = np.pi*np.ones(Nt)                                                            # Initial guess
u23 = np.pi*np.ones(Nt)                                                            # Initial guess
u31 = 0.5*np.pi*np.ones(Nt)                                                        # Initial guess

''' Hamiltonian '''
bec = BEC(nmax, mmax, s)                                                           # Create our BEC system
H   = bec.get_H()                                                                  # BEC system's Hamiltonian

''' Initial state '''
k       = bec.get_mapping(0,0)                                                     # Index of initial state 
psi0    = np.zeros(Nk)                                                             # Initialization of initial state 
psi0[k] = 1                                                                        # Initial state 
 
''' Target state ''' 
psit    = np.zeros(Nk)         
k = bec.get_mapping(-3,-3) 
psit[k] = 1./np.sqrt(2.) 
k = bec.get_mapping(3,3) 
psit[k] = 1./np.sqrt(2.) 

''' Optimisation parameter '''
maxiter = 1                                                                       # Maximum number of iterations

#----------------------------------------------------------------------------------
#
# Solve the optimal control problem
#
#----------------------------------------------------------------------------------

''' OCT '''
dyn = propagation(t, u12, u23, u31, H, Nk, psi0, psit)                             # Initialize the propagation object
u12, u23, u31, fid = dyn.get_grape(maxiter)                                            # Run the GRAPE algorithm

''' Propagation with the optimal control '''
dyn.u12 = u12
dyn.u23 = u23
dyn.u31 = u31
C = dyn.get_state()                                                                # State associated to optimal control
print(1-dyn.get_fidelity_sm())

#----------------------------------------------------------------------------------
#
# Plot 1
#
#----------------------------------------------------------------------------------

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(1,1,1)
x_min = -mmax
x_max = mmax
y_min = -nmax
y_max = nmax
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
grid_x_ticks_minor = np.arange(x_min-1, x_max+1, 0.2)
grid_x_ticks_major = np.arange(x_min-1, x_max+1, 1)
ax.set_xticks(grid_x_ticks_minor, minor=True)
ax.set_xticks(grid_x_ticks_major)
grid_y_ticks_minor = np.arange(y_min-1, y_max+1, 0.2)
grid_y_ticks_major = np.arange(y_min-1, y_max+1, 1)
ax.set_yticks(grid_y_ticks_minor, minor=True)
ax.set_yticks(grid_y_ticks_major)
ax.grid(which='both', linestyle='--')
ax.grid(which='minor', alpha=0.2)
for m in range(-mmax, mmax + 1):
    for n in range(-nmax, nmax + 1):
        if m == 0 and n == 0:
            ax.scatter(m, n, color='C1', alpha=0.5, s=750,zorder=2)
        else:
            k = bec.get_mapping(m, n)
            pop = abs(C[k, -1])**2
            size = 750 * pop  
            test = ax.scatter(m, n, c=pop, s=size, vmin=0, vmax=1, cmap='Blues',zorder=2)
cbar = plt.colorbar(test, ax=ax)
cbar.set_label(r'$|c_{m,n}|^2$', fontsize=20)
ax.set_xlabel(r'$m$', fontsize=30)
ax.set_ylabel(r'$n$', fontsize=30)

#----------------------------------------------------------------------------------
#
# Plot 2
#
#----------------------------------------------------------------------------------

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(3,1)
ax0 = plt.subplot(gs[0,0])
ax0.plot(((t*h)/(E_L))*1.0e6, u12, color='C0')
ax0.set_xlim(0,thold)
ax0.set_xticks([])
ax0.set_ylabel(r'$\varphi_{1,2}(t)$', fontsize=30) 
ax0.set_xlabel(r'$t \ (\mu s)$', fontsize=30) 
ax0.axhline(np.pi, linestyle='--', color='black', linewidth=0.5)

#--- Second line ---
ax1 = plt.subplot(gs[1,0])
ax1.plot(((t*h)/(E_L))*1.0e6, u23, color='C1')
ax1.set_xlim(0,thold)
ax1.axhline(np.pi, linestyle='--', color='black', linewidth=0.5)
ax1.set_ylabel(r'$\varphi_{2,3}(t)$', fontsize=30) 
ax1.set_xticks([])
ax1.set_xlabel(r'$t \ (\mu s)$', fontsize=30) 

#--- Third line ---
ax2 = plt.subplot(gs[2,0])
ax2.plot(((t*h)/(E_L))*1.0e6, u31, color='C2')
ax2.set_xlim(0,thold)
ax2.axhline(np.pi/2., linestyle='--', color='black', linewidth=0.5)
ax2.set_ylabel(r'$\varphi_{3,1}(t)$', fontsize=30) 
ax2.set_xlabel(r'$t \ (\mu s)$', fontsize=30) 

plt.subplots_adjust(hspace=.0, wspace=0.)
plt.show()
