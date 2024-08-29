#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: GRAPE_bec_porte.py
# Author: Etienne Dionis
# Date: June 8, 2024
# Description: Optimal control of a Bose-Einstein condensate system with GRAPE
#              to generate gates.

#----------------------------------------------------------------------------------
#
# Modules
#
#----------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize
from scipy import linalg
import cmath
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import pandas as pd
from matplotlib.collections import PatchCollection
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
    Nf : int
        Gate's dimension.
    """

    def __init__(self, dimension, q, s, Nf):
        """Initialize the BEC class."""
        self.dimension  = dimension                                                # Dimension of the system
        self.q  = q                                                                # Quasi-momentum
        self.s  = s                                                                # Gate's dimension 
        self.Nf = Nf

    def get_TruncationOrder(self):
        """ Truncation order."""
        nmax = (self.dimension-1)/2                                                # Truncation order -nmax =< n => nmax
        return int(nmax)

    def get_H0(self):
        """ Return the free Hamiltonian part."""
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        H0   = np.zeros((Nk,Nk), dtype = 'complex128')                               # H0 matrix
        for k in range(-nmax,nmax+1):                                              # Create H0 matrix
            j = k+nmax
            H0[j,j] = (self.q + k)**2
        return H0                                                                  

    def get_H1(self):
        """ Return the part of H controlled by the cosine function."""
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        H1   = np.zeros((Nk,Nk), dtype = 'complex128')                               # H1 matrix
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
        H2   = np.zeros((Nk,Nk), dtype = 'complex128')                               # H2 matrix
        for k in range(-nmax,nmax+1):
            j = k+nmax
            if(k < nmax):
                H2[j,j+1] = 1j*0.25*self.s
                H2[j+1,j] = -1j*0.25*self.s
        return H2                                                                  # Controlled part of H by sin(u)
 
    def get_H(self):
        """ Return the total Hamiltonian."""
        H0 = self.get_H0()                                                         # H0 matrix
        H1 = self.get_H1()                                                         # H1 matrix
        H2 = self.get_H2()                                                         # H2 matrix
        H  = [H0, H1, H2, self.Nf]                                                 # Hamiltonian matrix
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
        """Initialize the Propagation class."""
        self.t     = t                                                             # Integration Time
        self.u     = u                                                             # Control
        self.H     = H                                                             # Hamiltonian
        self.Nk    = Nk                                                            # System's dimension
        self.psi0  = psi0                                                          # Initial state
        self.psit  = psit                                                          # Target state

    def get_propagator(self, ut, dt):
        """This function computes U=exp(-iHdt)."""
        H0 = self.H[0]                                                             # field-free Hamiltonian
        H1 = self.H[1]                                                             # Controlled part of H by cos(u)
        H2 = self.H[2]                                                             # Controlled part of H by sin(u)
        U  = np.zeros((self.Nk,self.Nk), dtype = 'complex128')                       # Initialization of the propagator
        U = linalg.expm(-1j*(H0 + np.cos(ut)*H1 + np.sin(ut)*H2)*dt)               # Propagator at each time step
        return U

    def get_state(self):
        """ State at time t."""
        Nt       = self.t.size                                                     # Number of time points
        C        = np.zeros((Nt,self.Nk,self.Nk), dtype = 'complex128')              # Initialization of the state
        C[0,:,:] = self.psi0                                                       # Initial condition
        for n in range(Nt-1):
            dt         = self.t[n+1]-self.t[n]                                     # Time step
            ut         = self.u[n]                                                 # Discrete control
            U          = self.get_propagator(ut, dt)                               # Get propagator
            C[n+1,:,:] = U @ C[n,:,:]                                              # Forward propagation at each time step                 
        return C

    def get_adjoint_state(self, psif):
        """ Adjoint state at time t."""
        Nt        = self.t.size                                                    # Number of time points
        rev_time  = list(reversed(range(len(self.t))))                             # Backward time
        del(rev_time[-1])                                                          # Delete the first index to compute D(0)
        D         = np.zeros((Nt,self.Nk,self.Nk), dtype = 'complex128')             # Initialization of adjoint state
        D[-1,:,:] = -sum(np.diag(self.psit.conj().T @ psif))*self.psit             # Final condition for the adjoint state
        for n in rev_time:
            dt         = self.t[n]-self.t[n-1]                                     # Time step
            ut         = self.u[n-1]                                               # Discrete control
            U          = self.get_propagator(ut, dt)                               # Get propagator
            D[n-1,:,:] = U.conj().T @ D[n,:,:]                                     # Backward propagation at each time step     
        return D

    def get_fidelity_gate(self):
        """ Fidelity (Square Modulus of gate transfer) at time tf."""
        Nf = self.H[3]
        C = self.get_state()
        F = 1.0 - abs(sum(np.diag(self.psit.conj().T @ C[-1,:,:])))**2/Nf**2
        return F.real

    def get_fidelity_derivative_gate_pmp(self):
        """ Fidelity at time tf."""
        H0       = self.H[0]                                                       # field-free part of H
        H1       = self.H[1]                                                       # Controlled part of H by cos(u)
        H2       = self.H[2]                                                       # Controlled part of H by sin(u)
        Nt       = self.t.size                                                     # Number of time points
        dF       = np.zeros(Nt)                                                    # Initialization of dF tab
        C        = self.get_state()                                                # Get state
        D        = self.get_adjoint_state(C[-1,:,:])                               # Get adjoint state
        for n in range(Nt):
            ut    = self.u[n]                                                      # Discrete control
            dM0   = -np.sin(ut)*H1 + np.cos(ut)*H2                                 # Derivative of H wrt u
            dF[n] = (sum(np.diag(D[n,:,:].conj().T @ dM0 @ C[n,:,:]))).imag
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
    res = dyn.get_fidelity_gate()
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
    res = dyn.get_fidelity_derivative_gate_pmp()
    return res

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
    F = 1 - dyn.get_fidelity_gate()
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
Nt      = 1000                                                                     # Number of time steps
thold   = 400                                                                      # Final time in µs
tf      = (thold*1.0e-6*E_L)/h                                                     # Normalized final time
t       = np.linspace(0,tf,Nt)                                                     # Normalized time

''' Data '''
nmax  = 10                                                                         # Dimension of the system -> -nmax < k < nmax
q     = 0                                                                          # -0.5<q<0.5 (normalized quasi-momentum)
s     = 5                                                                          # Lattice depth
Nk    = 2*nmax+1                                                                   # Dimension of the system
Nf    = 3                                                                          # Gate's dimension

''' Control Phase '''
ui = np.pi/3.0                                                                     #Initial constant control
u0 = ui*np.ones(Nt)                                                                #Initial u control (for all n steps)

''' Hamiltonian '''
bec = BEC(Nk, q, s, Nf)                                                            # Create our BEC system
H   = bec.get_H()                                                                  # BEC system's Hamiltonian

''' Initial state '''
psi0 = np.zeros((Nk,Nk))                                                           # Initialization of the initial state
psi0[nmax-1,nmax-1] = 1                                                            # Initial state
psi0[nmax,nmax] = 1
psi0[nmax+1,nmax+1] = 1
pop0 = abs(psi0)**2                                                                # Initial Population

''' Target state '''
psit = np.zeros((Nk,Nk))                                                           # Initialization of the target state
psit[nmax-1,nmax] = 1                                                              # Target state
psit[nmax,nmax-1] = 1
psit[nmax+1,nmax+1] = 1
popt = abs(psit)**2                                                                # Target Population

''' Optimisation parameter '''
maxiter = 100                                                                      # Maximum number of iterations
method = 'L-BFGS-B'                                                                # Method of the optimization algorithm (here BFGS)
bds = []                                                                           # Bound sequence
for n in range(Nt):
    bds.append((-np.pi,np.pi))                                                     # Control Phase between -\pi and \pi (None or comment if no bound is required)

#----------------------------------------------------------------------------------
#
# Solve the optimal control problem
#
#----------------------------------------------------------------------------------

''' OCT '''
callback.iter = 0
sol   = minimize(Cost, u0, method=method, jac=dCost,\
                 args=(t, H, Nk, psi0, psit), callback=callback,\
                 bounds=bds, tol=1e-17, options={'maxiter': maxiter})              # Minimization algorithm with a specified gradient
u     = sol.x                                                                      # save the optimal control into another variable
dyn   = propagation(t, u, H, Nk, psi0, psit)                                       # Create propagator class for u_optimal
C     = dyn.get_state()                                                            # State associated to optimal control

#----------------------------------------------------------------------------------
#
# Plot 1
#
#----------------------------------------------------------------------------------

fig = plt.figure(1,figsize=(8,7))
gs = gridspec.GridSpec(1,1)
ax = plt.subplot(gs[0,0])
ax.plot((t*h*1.0e6)/E_L, u, color='C0',linewidth=4)
ax.set_xlim(0,(t[-1]*h*1.0e6)/E_L)
ax.set_ylabel(r'$\varphi(t)$',fontsize=30)
ax.set_xlabel(r'$t \ (\mu s)$',fontsize=30)

#----------------------------------------------------------------------------------
#
# Plot 2
#
#----------------------------------------------------------------------------------

fig = plt.figure(2,figsize=(8,7))
gs = gridspec.GridSpec(1,1)
ax = plt.subplot(gs[0,0])
ylabels = np.arange(-nmax,nmax+1)
x, y = np.meshgrid(np.arange(Nk), np.arange(Nk), indexing='ij')
mat_phase = np.zeros((Nk,Nk))
for i in range(Nk):
    for j in range(Nk):
        mat_phase[i,j] = cmath.phase(C[-1,i,j])
mat_phase = mat_phase.T
x1 = np.arange(0, Nk)
y1 = np.arange(0, Nk)
s = abs(C[-1,:,:])**2
R = s/s.max()/2
circles = [plt.Circle((i,j), radius=R[i,j]) for i in range(Nk) for j in range(Nk)]
col = PatchCollection(circles, array=mat_phase.flatten(), cmap="Blues_r")
ax.add_collection(col)
ax.set(xticks=x1, yticks=y1,\
       xticklabels=ylabels, yticklabels=ylabels)
ax.set_xticks(x1+0.5, minor=True)
ax.set_yticks(y1+0.5, minor=True)
ax.grid(which='minor')
ax.set_xlabel(r'$n$',fontsize=30)
ax.set_ylabel(r'$m$',fontsize=30)
cbar = fig.colorbar(col, label=r'$arg(U_{m,n})$', ticks=[-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'$-3\pi/4$', r'$-\pi/2$' , r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$' , r'$3\pi/4$', r'$\pi$'])
plt.gca().invert_yaxis()
plt.show()
