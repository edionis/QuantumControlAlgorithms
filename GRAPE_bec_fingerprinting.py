#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: GRAPE_bec_fingerprinting.py
# Author: Etienne Dionis
# Date: June 28, 2024
# Description: GRAPE to fingerprint the value of a magnetic force in a BEC system.

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
    t : float
        Time of the system.
    q : float
        Quasi-momentum.
    s : float
        Field amplitude.
    lamb : float
        Magnetic force parameter.
    """

    def __init__(self, dimension, t, q, s, lamb):
        """Initialize the BEC class."""
        self.dimension  = dimension                                                # Dimension of the system
        self.t = t                                                                 # Time
        self.q = q                                                                 # Quasi-momentum
        self.s = s                                                                 # Field amplitude 
        self.lamb = lamb                                                           # Magnetic force parameter 

    def get_TruncationOrder(self):
        """ Truncation order."""
        nmax = (self.dimension-1)/2                                                # Truncation order -nmax =< n => nmax
        return int(nmax)

    def get_H0(self):
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        q    = self.q                                                          # Quasi-momentum
        H0   = np.zeros((self.lamb.size, self.t.size,Nk,Nk), dtype = 'complex128')   # H0 matrix
        for i in range(self.lamb.size):
            for l in range(self.t.size):
                for k in range(-nmax,nmax+1):                                              
                    j = k+nmax
                    H0[i,l,j,j] = (q + k + self.lamb[i]*self.t[l])**2
        return H0                                                                  # Not controlled part of H 
        
    def get_H1(self):
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        H1   = np.zeros((Nk,Nk), dtype = 'complex128')                               # H1 matrix
        for k in range(-nmax,nmax+1):                                              # Create Hm matrix      
            j = k+nmax
            if(k < nmax):
                H1[j,j+1] = -0.25*self.s
                H1[j+1,j] = -0.25*self.s
        return H1                                                                  # Controlled part of H by cos(u)

    def get_H2(self):
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        H2   = np.zeros((Nk,Nk), dtype = 'complex128')                               # H2 matrix
        for k in range(-nmax,nmax+1):                                              # Create H2 matrix      
            j = k+nmax
            if(k < nmax):
                H2[j,j+1] = 1j*0.25*self.s
                H2[j+1,j] = -1j*0.25*self.s
        return H2                                                                  # Controlled part of H by sin(u)
        
    def get_dH(self):
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                      # System's dimension
        q    = self.q                                                          # Quasi-momentum
        dH   = np.zeros((self.lamb.size, self.t.size,Nk,Nk), dtype = 'complex128')   # dH matrix
        for i in range(self.lamb.size):
            for l in range(self.t.size):
                for k in range(-nmax,nmax+1):                                              
                    j = k+nmax
                    dH[i,l,j,j] = 2.0*self.t[l]*(q + k + self.lamb[i]*self.t[l])
        return dH                                                                  # Derivative of H by lambda 
        
    def get_H(self):
        H0 = self.get_H0()                                                         # H0 matrix
        H1 = self.get_H1()                                                         # H1 matrix
        H2 = self.get_H2()                                                         # H2 matrix
        H  = [H0, H1, H2]                                                          # Hamiltonian matrix and parameters
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
    u : numpy.ndarray
        Discrete control.
    H : numpy.ndarray
        Hamiltonian.
    Nk : int
        System's dimension.
    g : float
        Ponderation coefficient.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray, optional
        Target state.
    dH : numpy.ndarray, optional
        Derivative of Hamiltonian by lambda.
    """

    def __init__(self, t, u, H, Nk, g, psi0, psit=None, dH=None): 
        """Initialize the Propagation class."""
        self.t     = t                                                             # Integration Time
        self.u     = u                                                             # Control
        self.H     = H                                                             # Hamiltonian
        self.Nk    = Nk                                                            # System's dimension
        self.g     = g                                                             # Ponderation coefficient 
        self.psi0  = psi0                                                          # Initial state
        self.psit  = psit                                                          # Target state
        self.dH    = dH                                                            # Hamiltonian derivative by lambda

    def get_extended_dynamic(self, A, dA):
        """ Extended Matrix."""
        Mzero  = np.zeros((A.shape[0], A.shape[0]))                                #Dimension of zero matrix
        Aex_1L = np.concatenate((A, dA), axis=0)                                   #Fist Column
        Aex_2L = np.concatenate((Mzero, A), axis=0)                                #Second Column
        Aex    = np.concatenate((Aex_1L, Aex_2L), axis=1)                          #Full matrix
        return Aex

    def get_propagator(self, ut, dt, i, n):
        """ This function compute U=exp(-iHdt)."""
        H0 = self.H[0]                                                             # Non controlled part of Hamiltonian
        H1 = self.H[1]                                                             # Controlled part of H by cos(u)
        H2 = self.H[2]                                                             # Controlled part of H by sin(u)
        U  = np.zeros((self.Nk,self.Nk), dtype = 'complex128')                       # Initialisation of propagator
        U = linalg.expm(-1j*(H0[i,n,:,:] + np.cos(ut)*H1 + np.sin(ut)*H2)*dt)      # Propagator at each time step
        return U

    def get_propagator_extended(self, ut, dt, n):
        """ This function compute U=exp(-iHdt) for H extended."""
        H0  = self.H[0]                                                            # Non controlled part of Hamiltonian
        H1  = self.H[1]                                                            # Controlled part of H by cos(u)
        H2  = self.H[2]                                                            # Controlled part of H by sin(u)
        dH  = self.dH[0,n,:,:]                                                     # Derivative of H1 by s
        U   = np.zeros((self.Nk*2,self.Nk*2), dtype = 'complex128')                  # Initialisation of propagator
        H        = H0[0,n,:,:] + np.cos(ut)*H1 + np.sin(ut)*H2                     # Controlled Hamiltonian
        Htilde   = self.get_extended_dynamic(H, dH)                                # Auxiliary matrix
        U        = linalg.expm(-1j*dt*Htilde)                                      # Evolution operator
        return U

    def get_state(self, i):
        """ State at time t."""
        Nt     = self.t.size                                                       # Number of time points
        C      = np.zeros((self.Nk,Nt), dtype = 'complex128')                        # Initialisation of State
        C[:,0] = self.psi0                                                         # Initial condition
        for n in range(Nt-1):
            dt       = self.t[n+1]-self.t[n]                                       # Time step
            ut       = self.u[n]                                                   # Discrete control
            U        = self.get_propagator(ut, dt, i, n)                           # Get propagator
            C[:,n+1] = U @ C[:,n]                                                  # Forward propagation at each time step                 
        return C

    def get_state_extended(self):
        """ Extended state."""
        Nt        = self.t.size                                                    # Number of time points
        C         = np.zeros((2*self.Nk,Nt), dtype = 'complex128')                   # Initialisation of State
        C[0:self.Nk,0] = self.psi0                                                 # Initial conditions
        for n in range(Nt-1):
            dt       = self.t[n+1]-self.t[n]                                       # Time step
            ut       = self.u[n]                                                   # Discrete control
            U        = self.get_propagator_extended(ut, dt, n)                     # Get propagator
            C[:,n+1] = U @ C[:,n]                                                  # Forward propagation at each time step                 
        return C

    def get_adjoint_state(self, chif, i):
        """ Adjoint state at time t."""
        Nt       = self.t.size                                                     # Number of time points
        rev_time = list(reversed(range(len(self.t))))                              # Backward time
        del(rev_time[-1])                                                          # Delete the first index to compute D(0)
        D        = np.zeros((self.Nk,Nt), dtype = 'complex128')                      # Initialization of adjoint state
        D[:,-1]  = chif                                                            # Final condition for the adjoint state
        for n in rev_time:
            dt       = self.t[n]-self.t[n-1]                                       # Time step
            ut       = self.u[n-1]                                                 # Discrete control
            U        = self.get_propagator(ut, dt, i, n)                           # Get propagator
            D[:,n-1] = U.conj().T @ D[:,n]                                         # Backward propagation at each time step     
        return D

    def get_fidelity_sm(self):
        """ Fidelity (Fingerprinting) at time tf."""
        C0 = self.get_state(0)
        C1 = self.get_state(1)
        F  = abs((C0[:,-1].conj().T @ C1[:,-1]))**2\
             + self.g*(1.0 - abs(C0[:,-1].conj().T @ self.psit)**2)
        return F.real

    def get_fidelity_derivative_sm_pmp(self):
        """ Fidelity at time tf."""
        H0       = self.H[0]                                                       # Non controlled part of Hamiltonian
        H1       = self.H[1]                                                       # Controlled part of H by cos(u)
        H2       = self.H[2]                                                       # Controlled part of H by sin(u)
        Nt       = self.t.size                                                     # Number of time points
        dF       = np.zeros(Nt)                                                    # Initialization of dF tab
        C0       = self.get_state(0)                                               # Get state associated to lambda_1
        C1       = self.get_state(1)                                               # Get state associated to lambda_2
        chif0    = -self.g*(self.psit.conj().T @ C0[:,-1])*self.psit\
                    + (C1[:,-1].conj().T @ C0[:,-1])*C1[:,-1]                      # Final condition for adjoint state
        chif1    = (C0[:,-1].conj().T @ C1[:,-1]) * C0[:,-1]                       # Final condition for adjoint state
        D0       = self.get_adjoint_state(chif0, 0)                                # Get state associated to lambda_1
        D1       = self.get_adjoint_state(chif1, 1)                                # Get state associated to lambda_2
        for n in range(Nt):
            ut    = self.u[n]                                                      # Discrete control
            dM0   = -np.sin(ut)*H1 + np.cos(ut)*H2                                 # Derivative of H wrt u
            dM1   = -np.sin(ut)*H1 + np.cos(ut)*H2                                 # Derivative of H wrt u
            dF[n] = (D0[:,n].conj().T @ dM0 @ C0[:,n]\
                    + D1[:,n].conj().T @ dM1 @ C1[:,n]).imag                       # Gradient of Fidelity
        return dF

#----------------------------------------------------------------------------------
#
# Functions
#
#----------------------------------------------------------------------------------

def Cost(u, t, H, Nk, g, psi0, psit):
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
    g : float
        Ponderation coefficient.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray
        Target state.

    Returns
    -------
    float
        Fidelity of the system for the given control `u`.
    """
    dyn = propagation(t, u, H, Nk, g, psi0, psit)
    res = dyn.get_fidelity_sm()
    return res

def dCost(u, t, H, Nk, g, psi0, psit):
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
    g : float
        Ponderation coefficient.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray
        Target state.

    Returns
    -------
    numpy.ndarray
        Result of the fidelity derivative computation.
    """
    dyn = propagation(t, u, H, Nk, g, psi0, psit)
    res = dyn.get_fidelity_derivative_sm_pmp()
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
    dyn = propagation(t, xk, H, Nk, g, psi0, psit)
    F = 1 - dyn.get_fidelity_sm()
    print('Iteration: %d, 1-F1=%.15f' % (callback.iter, F))
    callback.iter += 1

def QFI_func(psi, dpsi):
    """
    Compute the Quantum Fisher Information (QFI).

    Parameters
    ----------
    psi : ndarray
        State vector |ψ⟩.
    dpsi : ndarray
        Derivative of the state vector with respect to some parameter.

    Returns
    -------
    QFI : float
        The computed Quantum Fisher Information.
    """
    dpsi_dpsi = dpsi.conj().T @ dpsi                                               # Dψ scalar product
    psi_dpsi = abs(psi.conj().T @ dpsi)**2                                         # |⟨ψ|dψ⟩|²
    QFI = 4.0 * (dpsi_dpsi - psi_dpsi)                                             # QFI at time t
    return QFI

def CFI_func(psi, dpsi, Nk):
    """
    Compute the Classical Fisher Information (CFI).

    Parameters
    ----------
    psi : ndarray
        State vector |ψ⟩.
    dpsi : ndarray
        Derivative of the state vector with respect to some parameter.
    Nk : int
        Dimension of the system.

    Returns
    -------
    CFI : float
        The computed Classical Fisher Information.
    """
    CFI = 0                                                                        # Initial value of the sum
    for jj in range(Nk):                                                           # Iterate sum over system's dimension
        vec_n = np.zeros(Nk)                                                       # Define ket |n⟩
        vec_n[jj] = 1
        inter = (1.0 / ((vec_n.conj().T @ psi) * (psi.conj().T @ vec_n))) * \
                ((vec_n.conj().T @ dpsi) * (psi.conj().T @ vec_n) +
                 (vec_n.conj().T @ psi) * (dpsi.conj().T @ vec_n))**2              # Term of the sum
        CFI += inter                                                               # Update CFI sum
    return CFI

#----------------------------------------------------------------------------------
#
# Parameters of the physical system
#
#----------------------------------------------------------------------------------

''' Constant '''
mrb   = 86.909180527*1.66054e-27                                                   # Mass of Rb-87 atom
wl    = 1064e-9                                                                    # Wavelenght of lasers
d     = wl/2.0                                                                     # Spatial period
h     = 1.0545718e-34                                                              # Reduced Plank constant
k_L   = (2.0*np.pi)/d                                                              # Wave vector of the lattice
E_L   = (h*k_L)**2/(2.0*mrb)                                                       # Energy of the lattice
nu_L  = E_L/(2.0*np.pi*h)                                                          # Frequency of the lattice
g     = 1                                                                          # Ponderation parameter

''' Time '''
Nt      = 1000                                                                     # Number of time steps
thold   = 600                                                                      # Final time in µs
tf      = (thold*1.0e-6*E_L)/h                                                     # Normalized final time
t       = np.linspace(0,tf,Nt)                                                     # Normalized time

''' Data '''
nmax  = 15                                                                         # Dimension of the system -> -nmax < k < nmax
q     = 0                                                                          # Quasi-momentum
s     = 5.                                                                         # Lattice depth
lamb  = np.array([0.001, 0.002])                                                   # Magnetic force parameter
Nk    = 2*nmax+1                                                                   # Dimension of the system

''' Control ux '''
u = np.cos(2*t)                                                                    # Initial control

''' Hamiltonian '''
bec = BEC(Nk, t, q, s, lamb)                                                       # Create our BEC system
H   = bec.get_H()                                                                  # BEC system's Hamiltonian
dH  = bec.get_dH()                                                                 # Extended dynamics

''' Initial state '''
psi0       = np.zeros(Nk)                                                          # Initialization of initial state
psi0[nmax] = 1                                                                     # Initial state
pop0       = abs(psi0)**2                                                          # Initial Population
dpsi0 = np.zeros(Nk)                                                               # Initial state extended state

''' Target state '''
psit       = np.zeros(Nk)                                                          # Initialization of the initial state
psit[nmax] = 1                                                                     # Target state

''' Optimisation parameter '''
maxiter = 30                                                                       # Maximum number of iterations
method = 'BFGS'                                                                    # Method of the optimization algorithm (here BFGS)

#----------------------------------------------------------------------------------
#
# Solve the optimal control problem
#
#----------------------------------------------------------------------------------

''' OCT '''
callback.iter = 0
sol = minimize(Cost, u, method="BFGS", jac=dCost,\
               args=(t, H, Nk, g, psi0, psit), callback=callback,\
               options={'maxiter': maxiter})                                       # Minimization algorithm with a specified gradient
u     = sol.x                                                                      # save the optimal control field into another variable
dyn   = propagation(t, u, H, Nk, g, psi0, psit)                                    # Create propagator class for u_optimal
psi1  = dyn.get_state(0)                                                           # State associated to optimal control
psi2  = dyn.get_state(1)                                                           # State associated to optimal control

''' Compute QFI and CFI '''
CFI = np.zeros(Nt, dtype='complex128')
QFI = np.zeros(Nt, dtype='complex128')
propa = propagation(t, u, H, Nk, g, psi0, psit, dH)
C = propa.get_state_extended()
psi = C[0:Nk,:]                                                     
dpsi = C[Nk:2*Nk,:]
for n in range(Nt):
    CFI[n] = CFI_func(psi[:,n], dpsi[:,n], Nk)
    QFI[n] = QFI_func(psi[:,n], dpsi[:,n])

#----------------------------------------------------------------------------------
#
# Plot 
#
#----------------------------------------------------------------------------------

fig = plt.figure(figsize=(10.4,7.49))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])
ax0.plot((t*h*1.0e6)/E_L, QFI, color='C0', linewidth=4)
ax0.plot((t*h*1.0e6)/E_L, CFI, color='C1', linewidth=4)
ax0.set_xlim(0,thold)
ax0.set_ylabel(r'$F_{C,Q}(t)$',fontsize=30)
ax0.set_xlabel(r'$t \ (\mu s)$',fontsize=30)
ax0.set_ylim(bottom=0)
plt.subplots_adjust(hspace=.0, wspace=0.3)
plt.show()
