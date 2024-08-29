#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: GRAPE_bec.py
# Author: Etienne Dionis
# Date: June 7, 2024
# Description: Optimal control of a Bose-Einstein condensate system with GRAPE.

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
    """

    def __init__(self, dimension, q, s):
        """Initialize the BEC class."""
        self.dimension  = dimension                                                # Dimension of the system
        self.q = q                                                                 # Quasi-momentum
        self.s = s                                                                 # Field amplitude 

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
        H  = np.array([H0, H1, H2])                                                # Hamiltonian matrix
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
        Nt     = self.t.size                                                       # Number of time points
        C      = np.zeros((self.Nk,Nt), dtype = 'complex128')                        # Initialization of the state
        C[:,0] = self.psi0                                                         # Initial condition
        for n in range(Nt-1):
            dt       = self.t[n+1]-self.t[n]                                       # Time step
            ut       = self.u[n]                                                   # Discrete control
            U        = self.get_propagator(ut, dt)                                 # Get propagator
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
            ut       = self.u[n-1]                                                 # Discrete control
            U        = self.get_propagator(ut, dt)                                 # Get propagator
            D[:,n-1] = U.conj().T @ D[:,n]                                         # Backward propagation at each time step     
        return D

    def get_fidelity_sm(self):
        """ Fidelity (Square Modulus) at time tf."""
        C = self.get_state()
        F = 1-abs((C[:,-1].conj().T @ self.psit))**2                               # Fidelity 
        return F.real

    def get_fidelity_derivative_sm_pmp(self):
        """ Fidelity at time tf."""
        H0       = self.H[0]                                                       # field-free part of H
        H1       = self.H[1]                                                       # Controlled part of H by cos(u)
        H2       = self.H[2]                                                       # Controlled part of H by sin(u)
        Nt       = self.t.size                                                     # Number of time points
        dF       = np.zeros(Nt)                                                    # Initialization of dF tab
        C        = self.get_state()                                                # Get state
        D        = self.get_adjoint_state(C[:,-1])                                 # Get adjoint state
        for n in range(Nt):
            ut    = self.u[n]                                                      # Discrete control
            dM    = -np.sin(ut)*H1 + np.cos(ut)*H2                                 # Derivative of H wrt u
            dF[n] = (D[:,n].conj().T @ dM @ C[:,n]).imag                           # Compute du for each time step given by PMP
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
    gs = np.zeros(Nk, dtype='complex128')
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
    gs = np.zeros(Nk, dtype='complex128')
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

''' Control Phase '''
u0 = np.cos(2*t)                                                                   # Initial guess

''' Hamiltonian '''
bec = BEC(Nk, q, s)                                                                # Create our BEC system
H   = bec.get_H()                                                                  # BEC system's Hamiltonian

''' Initial state '''
psi0       = np.zeros(Nk)                                                          # Initialization of the initial state
psi0[nmax] = 1                                                                     # Initial state
pop0       = abs(psi0)**2                                                          # Initial Population

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

#---- First Control ----#
''' Target state '''
psit         = np.zeros(Nk)                                                        # Initialization of the initial state
psit[nmax+2] = 1                                                                   # Target state
popt1        = abs(psit)**2                                                        # Target Population

''' OCT '''
callback.iter = 0
sol   = minimize(Cost, u0, method=method, jac=dCost,\
                 args=(t, H, Nk, psi0, psit), callback=callback,\
                 bounds=bds, tol=1e-17, options={'maxiter': maxiter})              # Minimization algorithm with a specified gradient
uopt1 = sol.x                                                                      # save the optimal control into another variable
dyn   = propagation(t, uopt1, H, Nk, psi0, psit)                                   # Create propagator class for u_optimal
psif  = dyn.get_state()                                                            # State associated to optimal control

''' Wave function '''
Nx = 200
phix = np.zeros((Nt,Nx), dtype='complex128')
x_tab = np.linspace(-np.pi,np.pi,Nx)
for tt, ti in enumerate(t):
    for i,xx in enumerate(x_tab):
        for n in range(Nk):
            coef = 1.0/np.sqrt(2.0*np.pi)
            phix[tt,i] = phix[tt,i] + coef*psif[n,tt]*np.exp(1j*(n-nmax)*xx)       # Compute the wave function associated to the final state

''' Husini representation of a squeezed state '''
nb = 100
husimi1 = np.zeros((nb,nb))
utab = np.linspace(-np.pi, np.pi, nb)
vtab = np.linspace(-np.pi, np.pi, nb)
for i,q in enumerate(utab):
    for j,p in enumerate(vtab):
        vec = gaussian_state(q,p,nmax,s)
        res1 = abs(vec.conj().T @ psif[:,-1])**2
        husimi1[i,j] = res1

''' Population at final time '''
popf1 = abs(psif[:,-1])**2                                                         # Final population

#---- Second Control ----#
''' Target state '''
psit  = gaussian_state(0,0,nmax,s)
popt2 = abs(psit)**2                                                               #Target Population

''' OCT '''
callback.iter = 0
sol   = minimize(Cost, u0, method=method, jac=dCost,\
                 args=(t, H, Nk, psi0, psit), callback=callback,\
                 bounds=bds, tol=1e-17, options={'maxiter': maxiter})              # Minimization algorithm with a specified gradient
uopt2 = sol.x                                                                      # save the optimal control into another variable
dyn   = propagation(t, uopt2, H, Nk, psi0, psit)                                   # Create propagator class for u_optimal
psif  = dyn.get_state()                                                            # State associated to optimal control

''' Wave function '''
Nx = 200
phix2 = np.zeros((Nt,Nx), dtype='complex128')
x_tab = np.linspace(-np.pi,np.pi,Nx)
for tt, ti in enumerate(t):
    for i,xx in enumerate(x_tab):
        for n in range(Nk):
            coef = 1.0/np.sqrt(2.0*np.pi)
            phix2[tt,i] = phix2[tt,i] + coef*psif[n,tt]*np.exp(1j*(n-nmax)*xx)     # Compute the wave function associated to the final state

''' Husini representation of a squeezed state '''
nb = 100
husimi2 = np.zeros((nb,nb))
utab = np.linspace(-np.pi, np.pi, nb)
vtab = np.linspace(-np.pi, np.pi, nb)
for i,q in enumerate(utab):
    for j,p in enumerate(vtab):
        vec = gaussian_state(q,p,nmax,s)
        res1 = abs(vec.conj().T @ psif[:,-1])**2
        husimi2[i,j] = res1

''' Population at final time '''
popf2 = abs(psif[:,-1])**2                                                         # Final population

#---- Third Control ----#
''' Target state '''
psit  = squeezed_state(0,0,nmax,s,1./3.)
popt3 = abs(psit)**2                                                               #Target Population

''' OCT '''
callback.iter = 0
sol   = minimize(Cost, u0, method=method, jac=dCost,\
                 args=(t, H, Nk, psi0, psit), callback=callback,\
                 bounds=bds, tol=1e-17, options={'maxiter': maxiter})              # Minimization algorithm with a specified gradient
uopt3 = sol.x                                                                      # save the optimal control into another variable
dyn   = propagation(t, uopt3, H, Nk, psi0, psit)                                   # Create propagator class for u_optimal
psif  = dyn.get_state()                                                            # State associated to optimal control

''' Wave function '''
Nx = 200
phix3 = np.zeros((Nt,Nx), dtype='complex128')
x_tab = np.linspace(-np.pi,np.pi,Nx)
for tt, ti in enumerate(t):
    for i,xx in enumerate(x_tab):
        for n in range(Nk):
            coef = 1.0/np.sqrt(2.0*np.pi)
            phix3[tt,i] = phix3[tt,i] + coef*psif[n,tt]*np.exp(1j*(n-nmax)*xx)     # Compute the wave function associated to the final state

''' Husini representation of a squeezed state '''
nb = 100
husimi3 = np.zeros((nb,nb))
utab = np.linspace(-np.pi, np.pi, nb)
vtab = np.linspace(-np.pi, np.pi, nb)
for i,q in enumerate(utab):
    for j,p in enumerate(vtab):
        vec = gaussian_state(q,p,nmax,s)
        res1 = abs(vec.conj().T @ psif[:,-1])**2
        husimi3[i,j] = res1

''' Population at final time '''
popf3 = abs(psif[:,-1])**2                                                         # Final population

#----------------------------------------------------------------------------------
#
# Plot 1
#
#----------------------------------------------------------------------------------

fig = plt.figure(1,figsize=(14.4,7.49))
gs = gridspec.GridSpec(3, 3)                                                        

# FIRST COLUMN ####################################################################
#--- First line ---
ax0 = plt.subplot(gs[0,0])
tt = (t*h)/(1.e-6*E_L)
line0, = ax0.plot(tt, uopt1, color='r')
ax0.xaxis.set_ticks([])
ax0.set_xlim(0,tt[-1])
ax0.set_ylim(-np.pi-0.5, np.pi+0.5)
ax0.text(0.05, 0.95, r'\bf{(a)}',transform=ax0.transAxes,fontsize=30,\
         fontweight='bold',va='top')
ax0.axhline(-np.pi, linestyle='--', color='black', linewidth=0.5)
ax0.axhline(0, linestyle='--', color='black', linewidth=0.5)
ax0.axhline(np.pi, linestyle='--', color='black', linewidth=0.5)
ax0.set_yticks([-np.pi, 0, np.pi], [r'$-\pi$', '',r'$\pi$'])

#--- Second line ---
ax1 = plt.subplot(gs[1,0]) 
line1, = ax1.plot(tt, uopt2, color='green')
ax1.set_xlim(0,tt[-1])
ax1.set_ylim(-np.pi-0.5, np.pi+0.5)
ax1.axhline(-np.pi, linestyle='--', color='black', linewidth=0.5)
ax1.axhline(0, linestyle='--', color='black', linewidth=0.5)
ax1.axhline(np.pi, linestyle='--', color='black', linewidth=0.5)
ax1.set_yticks([-np.pi, 0, np.pi], [r'$-\pi$', '',r'$\pi$'])
ax1.xaxis.set_ticks([])
ax1.set_ylabel(r'$\varphi(t)$', fontsize=30) 

#--- Third line ---
ax11 = plt.subplot(gs[2,0]) 
line11, = ax11.plot(tt, uopt3, color='C0')
ax11.set_xlim(0,tt[-1])
ax11.set_ylim(-np.pi-0.5, np.pi+0.5)
ax11.axhline(-np.pi, linestyle='--', color='black', linewidth=0.5)
ax11.axhline(0, linestyle='--', color='black', linewidth=0.5)
ax11.axhline(np.pi, linestyle='--', color='black', linewidth=0.5)
ax11.set_yticks([-np.pi, 0, np.pi], [r'$-\pi$', '',r'$\pi$'])
ax11.set_xticks([0, tt[-1]/2, tt[-1]], [r'$0$', '',r'$150$'])
ax11.set_xlabel(r'$t \ (\mu s)$', fontsize=30) 

# SECOND COLUMN ###################################################################
#--- First line ---
ax2 = plt.subplot(gs[0,1])
width = 0.8
qq = np.arange(-nmax, nmax+1) 
ax2.bar(qq, popf1, width, color='red', edgecolor='black',\
        label=r'$Final \ population$') 
ax2.text(0.05, 0.95, r'\bf{(b)}',transform=ax2.transAxes,\
         fontsize=30,fontweight='bold',va='top')
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_ylim(0,1)
ax2.set_xlim(-5.5,5.5)
ax2.grid()
ax2.set_axisbelow(True)

#--- Second subplot ---
ax3 = plt.subplot(gs[1,1])
width = 0.8 
qq = np.arange(-nmax, nmax+1) 
ax3.bar(qq, popf2, width, color='green', edgecolor ='black',\
        label=r'$Final \ population$') 
ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8]) 
ax3.set_ylim(0,1)
ax3.set_xlim(-5.5,5.5)
ax3.grid()
ax3.set_ylabel(r'$| c_{0,n} |^2$', fontsize=30) 
ax3.set_axisbelow(True)

#--- Third subplot ---
ax33 = plt.subplot(gs[2,1])
width = 0.8 
qq = np.arange(-nmax, nmax+1) 
ax33.bar(qq, popf3, width, color='C0', edgecolor ='black',\
         label=r'$Final \ population$') 
ax33.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax33.set_ylim(0,1)
ax33.set_xlim(-5.5,5.5)
ax33.set_xlabel(r'$p$', fontsize=30) 
ax33.grid()
ax33.set_axisbelow(True)

# THIRD COLUMN ####################################################################
#--- First line ---
ax4 = plt.subplot(gs[0,2])
ax4.text(0.05, 0.95, r'\bf{(c)}',transform=ax4.transAxes,fontsize=30,\
         fontweight=1000,va='top')
phixx = phix[-1,:]
phixx = abs(phixx)**2
ax4.plot(x_tab, phixx, color='red')
ax4.fill_between(x_tab, phixx, color='red', alpha=.25)
ax4.xaxis.set_ticks([])
ax4.set_ylim(0, 1.8)
ax4.set_xlim(-np.pi, np.pi)

#--- Second subplot ---
ax5 = plt.subplot(gs[1,2])#, sharex = True)
phixx2 = phix2[-1,:]
phixx2 = abs(phixx2)**2
ax5.plot(x_tab, phixx2, color='green')
ax5.xaxis.set_ticks([])
ax5.fill_between(x_tab, phixx2, color='green', alpha=.25)
ax5.set_xlim(-np.pi, np.pi)
ax5.set_ylim(0, 1.8)
#ax5.set_yticks([0])
ax5.set_ylabel(r'$| \psi(x,t) |^2$', fontsize=30) 

#--- Third subplot ---
ax55 = plt.subplot(gs[2,2])
phixx3 = phix3[-1,:]
phixx3 = abs(phixx3)**2
ax55.plot(x_tab, phixx3, color='C0')
ax55.set_xticks([-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$\pi$'])
ax55.fill_between(x_tab, phixx3, color='C0', alpha=.25)
ax55.set_xlim(-np.pi, np.pi)
ax55.set_ylim(0, 1.8)
ax55.set_xlabel(r'$x$', fontsize=30) 

plt.subplots_adjust(hspace=.0, wspace=0.4)

#----------------------------------------------------------------------------------
#
# Plot 2
#
#----------------------------------------------------------------------------------

fig = plt.figure(2,figsize=(14.4,7.49))
gs = gridspec.GridSpec(1,1)
ax1 = plt.subplot(gs[0,0])
X, Y = np.meshgrid(utab,vtab)
Z = husimi1.T
ax1.axhline(0, color='black', linestyle='--', linewidth=0.7)
ax1.axvline(0, color='black', linestyle='--', linewidth=0.7)
cp = ax1.contourf(X, Y, Z,cmap='Reds')
ax1.contour(X, Y, Z, colors='black')
ax1.set_xticks([-np.pi, -np.pi*0.5, 0, np.pi*0.5, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax1.set_xlabel(r'$x$', fontsize=30)
ax1.set_ylabel(r'$p$', fontsize=30)
ax1.set_box_aspect(1)

#----------------------------------------------------------------------------------
#
# Plot 3
#
#----------------------------------------------------------------------------------

fig = plt.figure(3,figsize=(14.4,7.49))
gs = gridspec.GridSpec(1,1)
ax2 = plt.subplot(gs[0,0])
X, Y = np.meshgrid(utab,vtab)
Z = husimi2.T
ax2.axhline(0, color='black', linestyle='--', linewidth=0.7)
ax2.axvline(0, color='black', linestyle='--', linewidth=0.7)
cp = ax2.contourf(X, Y, Z,cmap='Greens')
ax2.contour(X, Y, Z, colors='black')
ax2.set_xticks([-np.pi, -np.pi*0.5, 0, np.pi*0.5, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax2.set_xlabel(r'$x$', fontsize=30)
ax2.set_ylabel(r'$p$', fontsize=30)
ax2.set_box_aspect(1)

#----------------------------------------------------------------------------------
#
# Plot 4
#
#----------------------------------------------------------------------------------

fig = plt.figure(4,figsize=(14.4,7.49))
gs = gridspec.GridSpec(1,1)
ax3 = plt.subplot(gs[0,0])
X, Y = np.meshgrid(utab,vtab)
Z = husimi3.T
ax3.axhline(0, color='black', linestyle='--', linewidth=0.7)
ax3.axvline(0, color='black', linestyle='--', linewidth=0.7)
cp = ax3.contourf(X, Y, Z,cmap='Blues')
ax3.contour(X, Y, Z, colors='black')
ax3.set_xticks([-np.pi, -np.pi*0.5, 0, np.pi*0.5, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax3.set_xlabel(r'$x$', fontsize=30)
ax3.set_ylabel(r'$p$', fontsize=30)
ax3.set_box_aspect(1)
plt.show()
