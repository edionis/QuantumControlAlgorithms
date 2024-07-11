#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: GRAPE_bec_robust.py
# Author: Etienne Dionis
# Date: June 29, 2024
# Description: Robust control of a Bose-Einstein condensate system with GRAPE, 
#              in phase and amplitude.

#----------------------------------------------------------------------------------
#
# Modules
#
#----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
from scipy import optimize
from scipy import linalg
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':'20'})
rc('text', usetex=True)

#----------------------------------------------------------------------------------
#
# Class System
#
#----------------------------------------------------------------------------------

class BEC(object):
    """
    Bose-Einstein Condensate system.

    Parameters
    ----------
    dimension : int
        The system's dimension.
    p0 : float
        Initial momentum.
    dphi : float
        Pulsation.
    """

    def __init__(self, dimension, p0, dphi):
        """ Initialize the Bose-Einstein Condensate system."""
        self.dimension = dimension                                                 # Dimension of the system
        self.p0 = p0                                                               # Initial momentum
        self.dphi = dphi                                                           # Pulsation

        
    def get_TruncationOrder(self):
        """ Troncation order."""
        nmax = (self.dimension-1)/2
        return int(nmax)
        
    def get_H0(self):
        """ Return the free Hamiltonian part."""
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension
        p0   = self.p0
        H0   = np.zeros((p0.size,Nk,Nk), dtype = 'complex_')
        for pp in range(p0.size):
            for k in range(-nmax,nmax+1):
                j = k+nmax
                H0[pp,j,j] = (k + (p0[pp]+self.dphi)*0.5)**2
        return H0

    def get_H0c(self):
        """ Return the free Hamiltonian part."""
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                    
        p0   = self.p0
        H0c  = np.zeros((p0.size,Nk,Nk), dtype = 'complex_')                     
        for pp in range(p0.size):
            for k in range(-nmax,nmax+1):                                        
                j = k+nmax
                H0c[pp,j,j] = (k + p0[pp]*0.5)
        return H0c                                                                
        
    def get_Hc(self):
        """ Return the cosine controlled part of H."""
        nmax = self.get_TruncationOrder()
        Nk   = self.dimension                                                    
        Hc   = np.zeros((Nk,Nk), dtype = 'complex_')                             
        for k in range(-nmax,nmax+1):                                            
            j = k+nmax
            if(k < nmax):
                Hc[j,j+1] = -1.0
                Hc[j+1,j] = -1.0
        return Hc

    def get_H(self):
        """ Return the total Hamiltonian."""
        H0  = self.get_H0()                                                      
        H0c = self.get_H0c()                                                     
        Hc  = self.get_Hc()                                                      
        H   = [H0, H0c, Hc]
        return H

#----------------------------------------------------------------------------------
#
# Class Propagation
#
#----------------------------------------------------------------------------------

class propagation(object):
    """
    Propagation of the quantum system.

    Parameters
    ----------
    t : numpy.ndarray
        Integration time.
    up : numpy.ndarray
        Discrete control.
    ua : numpy.ndarray
        Discrete control.
    H : numpy.ndarray
        Hamiltonian of the system.
    Nk : int
        System's dimension.
    coeff : float
        Coefficient for the control.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray, optional
        Target state (default is None).
    """

    def __init__(self, t, up, ua, H, Nk, coeff, psi0, psit=None):
        """ Initialize the propagation of the quantum system."""
        self.t = t                                                                 # Integration time
        self.up = up                                                               # Phase control
        self.ua = ua                                                               # Amplitude control
        self.H = H                                                                 # Hamiltonian
        self.Nk = Nk                                                               # System's dimension
        self.coeff = coeff                                                         # Coefficient for the control
        self.psi0 = psi0                                                           # Initial state
        self.psit = psit                                                           # Target state (optional)

    def get_propagator(self, upt, uat, dt, i):
        """This function compute U=exp(-iHdt)."""
        H0 = self.H[0]                                                             # H0 matrix
        H0c = self.H[1]                                                            # Non controlled part of Hamiltonian
        Hc = self.H[2]                                                             # Controlled part of H
        U  = np.zeros((self.Nk,self.Nk), dtype = 'complex_')                       # Initialisation of propagator
        U = linalg.expm(-1j*(H0[i,:,:] + upt*H0c[i,:,:]  + uat*Hc)*dt)             # Propagator at each time step
        return U

    def get_state(self, i):
        """ State at time t."""
        Nt     = self.t.size                                                       # Number of time points
        C      = np.zeros((self.Nk,Nt), dtype = 'complex_')                        # Initialisation of State
        C[:,0] = self.psi0                                                         # Initial condition
        for n in range(Nt-1):
            dt       = self.t[n+1]-self.t[n]                                       # Time step
            upt       = self.up[n]                                                 # Discrete control
            uat       = self.ua[n]                                                 # Discrete control
            U        = self.get_propagator(upt, uat, dt, i)                        # Get propagator
            C[:,n+1] = U @ C[:,n]                                                  # Forward propagation at each time step                 
        return C

    def get_adjoint_state(self, chif, i):
        """ Adjoint state at time t."""
        Nt       = self.t.size                                                     # Number of time points
        rev_time = list(reversed(range(len(self.t))))                              # Backward time
        del(rev_time[-1])                                                          # Delete the first index to compute D(0)
        D        = np.zeros((self.Nk,Nt), dtype = 'complex_')                      # Initialization of adjoint state
        D[:,-1]  = chif                                                            # Final condition for the adjoint state
        for n in rev_time:
            dt       = self.t[n]-self.t[n-1]                                       # Time step
            upt       = self.up[n-1]                                               # Discrete control
            uat       = self.ua[n-1]                                               # Discrete control
            U        = self.get_propagator(upt, uat, dt, i)                        # Get propagator
            D[:,n-1] = U.conj().T @ D[:,n]                                         # Backward propagation at each time step     
        return D

    def get_fidelity_sm(self, i):
        """ Fidelity (Square Modulus) at time tf."""
        C = self.get_state(i)
        F = 1-abs(C[:,-1].conj().T @ self.psit)**2
        return F.real

    def get_fidelity_derivative_sm_pmp(self, i):
        """ Fidelity at time tf."""
        H0       = self.H[0]                                                       # H0 matrix 
        H0c      = self.H[1]                                                       # Non controlled part of Hamiltonian
        Hc       = self.H[2]                                                       # Controlled part of H by amplitude
        Nt       = self.t.size                                                     # Number of time points
        D        = np.zeros((self.Nk,Nt), dtype = 'complex_')                      # Initialisation of adjoint state
        dFp       = np.zeros(Nt)                                                   # Initialization of dF tab
        dFa       = np.zeros(Nt)                                                   # Initialization of dF tab
        C        = self.get_state(i)                                               # Get state
        chif  = -(self.psit.conj().T @ C[:,-1]) * self.psit                        # Final condition for adjoint state
        D        = self.get_adjoint_state(chif,i)                                  # Get state
        for n in range(Nt):
            dMp    = H0c[i,:,:]                                                    # Derivative of H wrt u
            dMa    = Hc                                                            # Derivative of H wrt u
            dFp[n] = (D[:,n].conj().T @ dMp @ C[:,n]).imag                         # Compute du for each time step given by PMP
            dFa[n] = (D[:,n].conj().T @ dMa @ C[:,n]).imag                         # Compute du for each time step given by PMP
        return dFp, dFa

    def get_fidelity_rob(self):
        """ Robust Fidelity (Square Modulus) at time tf."""
        coeff = self.coeff
        F     = np.zeros(self.coeff.size)
        for i in range(self.coeff.size):
            F[i] = self.get_fidelity_sm(i) 
        Frob = 0
        for i in range(self.coeff.size):
            Frob = Frob + self.coeff[i]*F[i]                           
        return Frob

    def get_fidelity_derivative_rob(self):
        """ Robust Fidelity Derivative at time tf."""
        coeff = self.coeff
        dFp = np.zeros((coeff.size,self.t.size))
        dFa = np.zeros((coeff.size,self.t.size))
        for i in range(coeff.size):
            resp, resa = self.get_fidelity_derivative_sm_pmp(i)
            dFp[i,:] = resp
            dFa[i,:] = resa
        dFrobp = np.zeros(self.t.size)
        dFroba = np.zeros(self.t.size)
        for i in range(coeff.size):
            dFrobp = dFrobp + coeff[i]*dFp[i,:] 
            dFroba = dFroba + coeff[i]*dFa[i,:] 
        return dFrobp, dFroba

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
        up : numpy.ndarray
             Optimized control phase.
        ua : numpy.ndarray
            Optimized control amplitude.
        fidelity_history : list
            History of the fidelity at each iteration.
        """
        up, ua = self.up.copy(), self.ua.copy()
        J = np.zeros(maxiter)
        J[0] = self.get_fidelity_rob()
        print("iteration = %i, 1-F1 = %f" % (0, 1 - J[0]))
        for i in range(1, maxiter):
            dFp, dFa = self.get_fidelity_derivative_rob()
            
            # Line search for up
            def Cost_up(uu):
                self.up = uu
                return self.get_fidelity_rob()

            def dCost_up(uu):
                self.up = uu
                return self.get_fidelity_derivative_rob()[0]
            
            ls_up = optimize.line_search(Cost_up, dCost_up, up, -dFp)
            epsilon_up = ls_up[0] if ls_up[0] is not None else 0
            u12 = up - epsilon_up * dFp
            self.up = up

            # Line search for ua
            def Cost_ua(uu):
                self.ua = uu
                return self.get_fidelity_rob()

            def dCost_ua(uu):
                self.ua = uu
                return self.get_fidelity_derivative_rob()[1]
            
            ls_ua = optimize.line_search(Cost_ua, dCost_ua, ua, -dFa)
            epsilon_ua = ls_ua[0] if ls_ua[0] is not None else 0
            ua = ua - epsilon_ua * dFa
            self.ua = ua

            J[i] = self.get_fidelity_rob()

            print("iteration = %i, 1-F1 = %f" % (i, 1 - J[i]))

            if 1 - J[i] > tol:
                break
        return up, ua, J

#----------------------------------------------------------------------------------
#
# Functions
#
#----------------------------------------------------------------------------------

def gaussian_state(u, v, nmax, s):
    """
    Compute a Gaussian state.

    Parameters
    ----------
    u : float
        Position parameter.
    v : float
        Momentum parameter.
    nmax : int
        Truncation order.
    s : float
        Lattice depth.

    Returns
    -------
    numpy.ndarray
        The Gaussian state as a complex numpy array.
    """
    gs = np.zeros(2 * nmax + 1, dtype='complex_')
    for k in range(-nmax, nmax + 1):
        gs[k + nmax] = (np.exp(1j * u * v * 0.5) *
                        np.exp(-2.j * k * u) *
                        np.exp(-(2. * k - v)**2 / (8. * np.sqrt(s))))
    norm = np.sqrt(gs.conj().T @ gs)
    gs = gs / norm
    return gs


def squeezed_state(u, v, nmax, s, X):
    """
    Compute a squeezed state.

    Parameters
    ----------
    u : float
        Position parameter.
    v : float
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
        The squeezed state as a complex numpy array.
    """
    gs = np.zeros(2 * nmax + 1, dtype='complex_')
    for k in range(-nmax, nmax + 1):
        gs[k + nmax] = (np.exp(1j * u * v * 0.5) *
                        np.exp(-2.j * k * u) *
                        np.exp(-X**2 * (2. * k - v)**2 / (8. * np.sqrt(s))))
    norm = np.sqrt(gs.conj().T @ gs)
    gs = gs / norm
    return gs

#----------------------------------------------------------------------------------
#
# Parameters of the physical system
#
#----------------------------------------------------------------------------------

''' Constant '''
mrb   = 1.44e-25                                                                   # Mass of Rb-87 atom
wl    = 780e-9                                                                     # Wavelenght of lasers
h     = 1.0545718e-34                                                              # Reduced Plank constant
kcst  = (2.0*np.pi)/wl                                                             # Wave vector of the lattice
wr    = (h*kcst**2)/(2.0*mrb)                                                      # Constant used to normalized the system
p0    = 0                                                                          # Initial momentum - in units of hbar*k 
nmax  = 10                                                                         # Truncation order
Nk    = 2*nmax+1                                                                   # Dimension of the system

''' Robust parameters '''
sigmap = 0.3                                                                       # Dispertion of p0
p0      = np.array([-0.9, -0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 0.9])                # Value of p0
coeff = np.exp(-p0**2/(2*sigmap**2))                                               # Discrete Gaussian function
coeff    = coeff/sum(coeff)                                                        # Coefficient on the p0 value 

''' Time '''
Nt      = 400                                                                      # Number of time steps
thold   = 100e-6                                                                   # Final time in µs                                                           
tf      = thold*4.0*wr                                                             # Normalized final time
t       = np.linspace(0,tf,Nt)                                                     # Normalized time
dt      = t[1]-t[0]                                                                # Time step

''' Amplitude Control '''
Omax = 0.2                                                                         # Amplitude
ua = Omax*np.tanh(8*(t/tf))*np.tanh(8*(1-(t/tf)))                                  # Initial guess for amplitude control
ua : Omax*np.cos(t)

''' Phase Control '''
up = 0.2*np.cos(t)                                                                 # Initial guess for phase control

''' Phase Control '''
dphi = 0.5

''' Initial state '''
psi0       = np.zeros(Nk)                                                          # Initialization of initial state
psi0[nmax] = 1                                                                     # Etat initial
pop0       = abs(psi0)**2                                                          # Initial Population

''' Final state '''
psit = squeezed_state(0,0,nmax,Omax,1.5)                                           # Target state 

''' Hamiltonian '''
bec = BEC(Nk, p0, dphi)
H   = bec.get_H()

''' Iteration parameters '''
maxiter = 30                                                                       # Maximum of iteration

#----------------------------------------------------------------------------------
#
# Solve the optimal control problem
#
#----------------------------------------------------------------------------------

''' GRAPE '''
gp = propagation(t, up, ua, H, Nk, coeff, psi0, psit)
up, ua, J = gp.get_grape(maxiter=maxiter)

''' Robustness '''
NN = 101
p0_tab = np.linspace(-0.9, 0.9, NN) 
F1 = np.zeros(NN)
for i, p0 in enumerate(p0_tab):
    bec.p0 = np.array([p0])
    H = bec.get_H()
    coeff = 1.
    dyn = propagation(t, up, ua, H, Nk, coeff, psi0, psit)
    F1[i] = 1.0 - dyn.get_fidelity_sm(0)

#----------------------------------------------------------------------------------
#
# Plot
#
#----------------------------------------------------------------------------------

fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(1,3)

#--- First column ---
ax0 = plt.subplot(gs[0,0])
t = (t/(4.0*wr))*1.0e6
line0, = ax0.plot(t, up, color='C0')
ax0.set_xlim(0,t[-1])
ax0.axhline(0, linestyle='--', color='black', linewidth=0.5)
ax0.set_xlabel(r'$t \ (\mu s)$', fontsize=30)
ax0.set_ylabel(r'$\omega(t)$', fontsize=30)
ax0.set_box_aspect(1)

#--- Third column ---
ax0 = plt.subplot(gs[0,1])
t = (t/(4.0*wr))*1.0e6
line0, = ax0.plot(t, ua, color='C2')
ax0.set_xlim(0,t[-1])
ax0.axhline(0, linestyle='--', color='black', linewidth=0.5)
ax0.set_xlabel(r'$t \ (\mu s)$', fontsize=30)
ax0.set_ylabel(r'$\gamma(t)$', fontsize=30)
ax0.set_box_aspect(1)

#--- Second column ---
ax2 = plt.subplot(gs[0,2])
ax2.plot(p0_tab, F1)
ax2.set_ylim(0,1)
ax2.set_xlim(p0_tab[0],p0_tab[-1])
ax2.set_xlabel(r'$p_0$', fontsize=30)
ax2.set_ylabel(r'$1-F_1$', fontsize=30)
ax2.set_box_aspect(1)
plt.show()
