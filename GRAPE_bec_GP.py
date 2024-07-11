#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: GRAPE_bec_GP.py
# Author: Etienne Dionis
# Date: June 27, 2024
# Description: Optimal control of the Gross-Pitaevskii equation with GRAPE.

#----------------------------------------------------------------------------------
#
# Modules
#
#----------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize
from scipy import linalg
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
    dimension : int
        The system's dimension.
    dimension : float
        Dimension of the system.
    q : float
        Quasi-momentum.
    s : float
        Field amplitude.
    beta : float
        Non-linear paramter.
    """

    def __init__(self, dimension, q, s, beta):
        """Initialize the BEC class."""
        self.dimension  = dimension                                                # Dimension of the system
        self.q = q                                                                 # Quasi-momentum
        self.s = s                                                                 # Field amplitude 
        self.beta = beta                                                           # Non-linear parameter

    def get_TruncationOrder(self):
        """ Truncation order."""
        nmax = (self.dimension-1)/2                                                # Truncation order -nmax =< n => nmax
        return int(nmax)

    def get_H0(self):
        """ Return the free Hamiltonian part."""
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        H0   = np.zeros((Nk,Nk), dtype = 'complex_')                               # H0 matrix
        for k in range(-nmax,nmax+1):                                              # Create H0 matrix
            j = k+nmax
            H0[j,j] = (self.q + k)**2
        return H0                                                                  

    def get_H1(self):
        """ Return the part of H controlled by the cosine function."""
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        H1   = np.zeros((Nk,Nk), dtype = 'complex_')                               # H1 matrix
        for k in range(-nmax,nmax+1):
            j = k+nmax
            if(k < nmax):
                H1[j,j+1] = -0.25*self.s
                H1[j+1,j] = -0.25*self.s
        return H1                                                                  # Controlled part of H by cos(u)
 
    def get_H2(self):
        """ Return the part of H controlled by the sine function."""
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        H2   = np.zeros((Nk,Nk), dtype = 'complex_')                               # H2 matrix
        for k in range(-nmax,nmax+1):
            j = k+nmax
            if(k < nmax):
                H2[j,j+1] = 1j*0.25*self.s
                H2[j+1,j] = -1j*0.25*self.s
        return H2                                                                  # Controlled part of H by sin(u)

    def get_DVR_FBR(self):
        """ Return the passage matrix from DVR to FBR."""
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        R    = np.zeros((Nk,Nk), dtype = 'complex_')                               # Change-of-basis matrix
        for j in range(Nk):
            for n in range(Nk):
                norm = np.sqrt(1.0/Nk)
                coef = (2.0*np.pi)/Nk
                R[j,n] = norm*np.exp(1j*coef*(n-nmax)*j)
        return R                                                                   # Controlled part of H by sin(u)
        
    def get_H(self):
        """ Return the total Hamiltonian."""
        H0 = self.get_H0()                                                         # H0 matrix
        H1 = self.get_H1()                                                         # H1 matrix
        H2 = self.get_H2()                                                         # H2 matrix
        R  = self.get_DVR_FBR()                                                    # Matrix to go from DVR to FBR
        H  = [H0, H1, H2, self.beta, R]                                            # Hamiltonian matrix and parameters
        return H
 
    def get_ground_state(self):
        """ Return the ground state of H."""
        H0 = self.get_H0()                                                         # H0 matrix
        H1 = self.get_H1()                                                         # H1 matrix
        val,vp = np.linalg.eigh(H0 + H1)                                           # Hamiltonian matrix
        return vp[:,0]                                                             # Ground state of Hamiltonian

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
    u : numpy.ndarray
        Discrete control.
    H : numpy.ndarray
        Hamiltonian.
    Nk : int
        System's dimension.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray, optional
        Target state.
    """

    def __init__(self, t, u, H, Nk, psi0, psit=None): 
        """ Initialize the Propagation class."""
        self.t     = t                                                             # Integration Time
        self.u     = u                                                             # Control
        self.H     = H                                                             # Hamiltonian
        self.Nk    = Nk                                                            # System's dimension
        self.psi0  = psi0                                                          # Initial state
        self.psit  = psit                                                          # Target state

    def get_extended_dynamic(self, A, dA):
        """ Extended Matrix (M = [[A,0],[dA,A]])."""
        Mzero  = np.zeros((A.shape[0], A.shape[0]))                                # Dimension of zero matrix
        Aex_1L = np.concatenate((A, dA), axis=0)                                   # Fist Column
        Aex_2L = np.concatenate((Mzero, A), axis=0)                                # Second Column
        Aex    = np.concatenate((Aex_1L, Aex_2L), axis=1)                          # Full matrix
        return Aex

    def get_psi_DVR(self, C):
        """ State in DVR basis."""
        nmax   = (self.Nk-1)/2                                                     # Truncation order
        psiDVR = np.zeros((self.Nk,self.Nk), dtype = 'complex_')                   # Initialization of psi matrix in DVR basis
        for j in range(self.Nk):
            for n in range(self.Nk):
                coef = (2.0*np.pi)/self.Nk
                norm = 1.0/np.sqrt(2.0*np.pi)
                psiDVR[j,n] = norm*np.exp(1.0j*coef*(n-nmax)*j)
        res = psiDVR @ C
        res = np.diag(res)
        return res

    def get_propagator(self, ut, dt, C):
        """ This function compute U=exp(-i(H+beta*|psi|^2)dt)."""
        H0 = self.H[0]                                                             # Non controlled part of Hamiltonian
        H1 = self.H[1]                                                             # Controlled part of H by cos(u)
        H2 = self.H[2]                                                             # Controlled part of H by sin(u)
        beta = self.H[3]                                                           # Interaction paramater
        R  = self.H[4]                                                             # Matrix to go from DVR to FBR
        U  = np.zeros((self.Nk,self.Nk), dtype = 'complex_')                       # Initialisation of propagator
        # Hamiltonian
        V      =  np.cos(ut)*H1 + np.sin(ut)*H2 
        psiDVR = self.get_psi_DVR(C)                                               # State in DVR basis
        # Propagator
        U = linalg.expm(-1.0j*(H0 + V + beta*(R.conj().T@(abs(psiDVR)**2)@R))*dt)
        return U

    def get_state(self):
        """ State at time t."""
        Nt     = self.t.size                                                       # Number of time points
        C      = np.zeros((self.Nk,Nt), dtype = 'complex_')                        # Initialisation of State
        C[:,0] = self.psi0                                                         # Initial condition
        for n in range(Nt-1):
            dt = self.t[n+1]-self.t[n]                                             # Time step
            ut = self.u[n]                                                         # Discrete control
            U  = self.get_propagator(ut, dt, C[:,n])                               # Get propagator
            C[:,n+1] = U @ C[:,n]                                                  # Forward propagation at each time step                 
        return C

    def get_adjoint_state(self, psif):
        """ Adjoint state at time t."""
        H0       = self.H[0]                                                       # Non controlled part of Hamiltonian
        H1       = self.H[1]                                                       # Controlled part of H by cos(u)
        H2       = self.H[2]                                                       # Controlled part of H by sin(u)
        beta     = self.H[3]
        R        = self.H[4]
        Nt       = self.t.size                                                     # Number of time points
        D        = np.zeros((self.Nk,Nt), dtype = 'complex_')                      # Initialisation of adjoint state associated to s0
        rev_time = list(reversed(range(len(self.t))))                              # Backward time
        del(rev_time[-1])                                                          # Delete the first index to compute D(0)
        pc       = np.zeros((2*self.Nk,Nt), dtype = 'complex_')                    # Extended state
        pc[0:self.Nk,-1] = psif                                                    # State at tf
        pc[self.Nk:,-1]  = -(self.psit.conj().T @ psif)*self.psit                  # Final condition for adjoint state
        D[:,-1] = pc[self.Nk:,-1]
        for n in rev_time:
            ut        = self.u[n-1]                                                # Discrete control
            dt        = self.t[n] - self.t[n-1]                                    # Time step
            psi       = self.get_psi_DVR(pc[0:self.Nk,n])                          # State in DVR basis
            psi_mod2  = abs(psi)**2                                                # Modulus square of state in DVR basis
            psi_mod2  = R.conj().T @ psi_mod2 @ R                                  # Modulus square of state in FBR basis
            V      =  np.cos(ut)*H1 + np.sin(ut)*H2 
            H         = -1j*(H0 + V + beta*psi_mod2)*dt                            # Hamiltonian
            chit      = self.get_psi_DVR(pc[self.Nk:,n])                           # Adjoint state conjugate in DVR basis
            Im        = (chit.conj() @ psi).imag                                   # Product od adjoint state and state in DVR basis
            Im        = R.conj().T @ Im @ R                                        # Product od adjoint state and state in FBR basis
            HH        = -2.0*beta*Im*dt                                            # Interaction term in extended Hamiltonian
            Htilde    = self.get_extended_dynamic(H, HH)                           # Extended Hamiltonian
            U         = linalg.expm(Htilde)                                        # Matrix exponential of extended Hamiltonien
            pc[:,n-1] = U.conj().T @ pc[:,n]                                       # Backward propagation at each time step     
            D[:,n-1] = pc[self.Nk:,n-1]
        return D

    def get_fidelity_sm(self):
        """ Fidelity (Square Modulus) at time tf."""
        C = self.get_state()
        F  = 1.-abs((C[:,-1].conj().T @ self.psit))**2
        return F.real

    def get_fidelity_derivative_sm_pmp(self):
        """ Fidelity at time tf."""
        H0       = self.H[0]                                                       # Non controlled part of Hamiltonian
        H1       = self.H[1]                                                       # Controlled part of H by cos(u)
        H2       = self.H[2]                                                       # Controlled part of H by sin(u)
        Nt       = self.t.size                                                     # Number of time points
        dF       = np.zeros(Nt)                                                    # Initialization of dF tab
        C        = self.get_state()                                                # Get state
        D        = self.get_adjoint_state(C[:,-1])                                 # Get adjoint state
        for n in range(Nt):
            ut    = self.u[n]                                                      # Discrete control
            dM    = -np.sin(ut)*H1 + np.cos(ut)*H2                                 # Derivative of H wrt u
            dF[n] = (D[:,n].conj().T @ dM @ C[:,n]).imag                           # Compute gradient fidelity
        return dF

#----------------------------------------------------------------------------------
#
# Functions
#
#----------------------------------------------------------------------------------

def Cost(u, t, H, Nk, psi0, psit):
    """
    Compute the cost from the propagation class.

    Parameters
    ----------
    u : numpy.ndarray
        Discrete control.
    t : numpy.ndarray
        Integration time.
    H : numpy.ndarray
        Hamiltonian.
    Nk : int
        System's dimension.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray
        Target state.

    Returns
    -------
    float
        Fidelity of the system for the given control `u`.
    """
    dyn = propagation(t, u, H, Nk, psi0, psit)
    res = dyn.get_fidelity_sm()
    return res

def dCost(u, t, H, Nk, psi0, psit):
    """
    Computes the direction of the gradient algorithm for u^(i).

    This function computes the direction of the gradient algorithm
    from the propagation class.

    Parameters
    ----------
    u : numpy.ndarray
        Discrete control.
    t : numpy.ndarray
        Integration time.
    H : numpy.ndarray
        Hamiltonian.
    Nk : int
        System's dimension.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray
        Target state.

    Returns
    -------
    numpy.ndarray
        Result of the fidelity derivative computation.
    """
    dyn = propagation(t, u, H, Nk, psi0, psit)
    res = dyn.get_fidelity_derivative_sm_pmp()
    return res

def gaussian_state(x0, p0, nmax, s):
    """
    Computes a Gaussian state.

    Parameters
    ----------
    x0 : float
        Position.
    p0 : float
        Momentum.
    nmax : int
        Maximum quantum number.
    s : float
        Lattice depth.

    Returns
    -------
    numpy.ndarray
        Normalized Gaussian state.
    """
    gs = np.zeros(Nk, dtype='complex_')
    for k in range(-nmax, nmax + 1):
        gs[k + nmax] = (np.exp(1j * x0 * p0 * 0.5) * np.exp(-1j * k * x0)
                        * np.exp(-(k - p0) ** 2 / np.sqrt(s)))
    norm = gs.conj().T @ gs
    norm = np.sqrt(norm)
    gs = gs / norm
    return gs

def squeezed_state(x0, p0, nmax, s, X):
    """
    Compute a squeezed Gaussian state.

    Parameters
    ----------
    x0 : float
        Position parameter.
    p0 : float
        Momentum parameter.
    nmax : int
        Truncation order.
    s : float
        Lattice depth.
    X : float
        Squeezing factor.

    Returns
    -------
    numpy.ndarray
        The squeezed Gaussian state as a complex numpy array.
    """
    gs = np.zeros(Nk, dtype='complex_')
    for k in range(-nmax, nmax + 1):
        gs[k + nmax] = (np.exp(1j * x0 * p0 * 0.5) *
                        np.exp(-1j * k * x0) *
                        np.exp(-X**2 * (k - p0)**2 / np.sqrt(s)))
    norm = gs.conj().T @ gs
    norm = np.sqrt(norm)
    gs = gs / norm
    return gs

def callback(xk):
    """
    Prints the progress of the algorithm at each step.

    Parameters
    ----------
    xk : numpy.ndarray
        Current state of the system.

    Returns
    -------
    None
    """
    dyn = propagation(t, xk, H, Nk, psi0, psit)
    F = 1 - dyn.get_fidelity_sm()
    print('Iteration: %d, 1-F1=%.15f' % (callback.iter, F))
    callback.iter += 1

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
thold   = 150                                                                      # Final time in µs
tf      = (thold*1.0e-6*E_L)/h                                                     # Normalized final time
t       = np.linspace(0,tf,Nt)                                                     # Normalized time

''' Data '''
nmax  = 10                                                                         # Dimension of the system -> -nmax < k < nmax
q     = 0                                                                          # -0.5<q<0.5 (normalized quasi-momentum)
s     = 5                                                                          # Lattice depth
Nk    = 2*nmax+1                                                                   # Dimension of the system
beta  = 0.7                                                                        # Non-linear parameter

''' Control Phase '''
u0 = np.cos(0.5*t)                                                                 # Initial guess

''' Hamiltonian '''
bec = BEC(Nk, q, s, beta)                                                          # Create our BEC system
H   = bec.get_H()                                                                  # BEC system's Hamiltonian

''' Initial state '''
psi0       = np.zeros(Nk)                                                          # Initialization of the initial state
psi0[nmax] = 1                                                                     # Initial state
pop0       = abs(psi0)**2                                                          # Initial Population

''' Target state '''
psit = squeezed_state(0,0,nmax,s,X=1.5)
popt       = abs(psit)**2                                                          # Initial Population

''' Optimisation parameter '''
maxiter = 10                                                                       # Maximum number of iterations
method = 'BFGS'                                                                    # Method of the optimization algorithm (here BFGS)

#----------------------------------------------------------------------------------
#
# Solve the optimal control problem
#
#----------------------------------------------------------------------------------

''' OCT '''
callback.iter = 0
sol   = minimize(Cost, u0, method=method, jac=dCost,\
                 args=(t, H, Nk, psi0, psit), callback=callback,\
                 tol=1e-17, options={'maxiter': maxiter})                          # Minimization algorithm with a specified gradient
uopt = sol.x                                                                       # save the optimal control into another variable
dyn   = propagation(t, uopt, H, Nk, psi0, psit)                                    # Create propagator class for u_optimal
psif  = dyn.get_state()                                                            # State associated to optimal control

''' Population at final time '''
popf = abs(psif[:,-1])**2                                                          # Final population

#----------------------------------------------------------------------------------
#
# Plot 1
#
#----------------------------------------------------------------------------------

fig = plt.figure(1,figsize=(14.4,7.49))
gs = gridspec.GridSpec(1,2)                                                        

#--- First column ---
ax1 = plt.subplot(gs[0,0]) 
tt = (t*h)/(1.e-6*E_L)
ax1.plot(tt, uopt, color='C0')
ax1.set_xlim(0,tt[-1])
ax1.set_ylim(-np.pi-0.5, np.pi+0.5)
ax1.axhline(-np.pi, linestyle='--', color='black', linewidth=0.5)
ax1.axhline(0, linestyle='--', color='black', linewidth=0.5)
ax1.set_ylabel(r'$\varphi(t)$', fontsize=30) 
ax1.axhline(np.pi, linestyle='--', color='black', linewidth=0.5)
ax1.set_yticks([-np.pi, 0, np.pi], [r'$-\pi$', '',r'$\pi$'])
ax1.set_xticks([0, tt[-1]/2, tt[-1]], [r'$0$', '',r'$150$'])
ax1.set_xlabel(r'$t \ (\mu s)$', fontsize=30) 
ax1.set_box_aspect(1)

#--- Second column ---
ax2 = plt.subplot(gs[0,1])
width = 0.8
qq = np.arange(-nmax, nmax+1) 
ax2.bar(qq, popf, width, color='C0', edgecolor='black') 
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_ylim(0,1)
ax2.set_xlim(-5.5,5.5)
ax2.grid()
ax2.set_axisbelow(True)
ax2.set_box_aspect(1)

plt.show()
