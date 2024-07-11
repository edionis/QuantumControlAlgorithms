#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: GRAPE_fluxonium_distorsion.py
# Author: Etienne Dionis
# Date: June 28, 2024
# Description: Optimal control of a fluxonium system with a delay on the control
#              paramater.
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

class FLUX(object):
    """
    Fluxonium system.

    Parameters
    ----------
    dimension : int
        The system's dimension.
    Ec : float
        Charging energy of the fluxonium.
    Ej : float
        Josephson energy of the fluxonium.
    El : float
        Inductive energy of the fluxonium.
    """

    def __init__(self, dimension, Ec, Ej, El):
        """ Initialize the fluxonium system."""
        self.dimension = dimension 
        self.Ec = Ec
        self.Ej = Ej
        self.El = El
       
    def get_H0(self):
        """ Return the free Hamiltonian part."""
        Nk   = self.dimension                                                      # System's dimension
        H0   = np.zeros((Nk,Nk), dtype = 'complex_')                               # H0 matrix
        for k in range(Nk):                                                        # Create H0 matrix
            H0[k,k] = np.sqrt(8.0*self.Ec*self.El)*(k+0.5)
        return H0                                                                  # Not controlled part of H 

    def get_H1(self):
        """ Return the free Hamiltonian part."""
        Nk   = self.dimension                                                      # System's dimension
        H1   = np.zeros((Nk,Nk), dtype = 'complex_')                               # H1 matrix
        lamb = (self.El/(8.0*self.Ec))**(-0.25)
        coef = (lamb*self.El)/np.sqrt(2.0)
        for k in range(Nk):                                                        # Create H1 matrix
            if(k < Nk-1):
                H1[k,k+1] = coef*np.sqrt(k+1)
                H1[k+1,k] = coef*np.sqrt(k+1)
        return H1                                                                  # Controlled part of H 

    def get_cosX(self):
        """ Return the free Hamiltonian part."""
        Nk   = self.dimension                                                      # System's dimension
        X    = np.zeros((Nk,Nk), dtype = 'complex_')                               # X matrix
        cosX = np.zeros((Nk,Nk), dtype = 'complex_')                               # cosX matrix
        lamb = (self.El/(8.0*self.Ec))**(-0.25)
        coef = 1.0/np.sqrt(2.0)
        for k in range(Nk):
            if(k < Nk-1):
                X[k,k+1] = coef*np.sqrt(k+1)
                X[k+1,k] = coef*np.sqrt(k+1)
        cosX = 0.5*self.Ej*(linalg.expm(1j*lamb*X) + linalg.expm(-1j*lamb*X))
        return cosX
       
    def get_H(self):
        """ Return the total Hamiltonian."""
        H0   = self.get_H0()                                                       # H0 matrix
        H1   = self.get_H1()                                                       # H1 matrix
        cosX = self.get_cosX()                                                     # cosX matrix
        H    = [H0, H1, cosX, self.El]                                             # Hamiltonian matrix
        return H

    def get_ground_state(self):
        """ Return the ground state of H."""
        H0 = self.get_H0()                                                         # H0 matrix
        cosX = self.get_cosX()                                                     # cosX matrix
        val,vp = np.linalg.eigh(2.0*np.pi*(H0 + cosX))                             # Hamiltonian matrix for u=0
        return val,vp                                                              # Ground state of Hamiltonian

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
    ub : numpy.ndarray
        Discrete control.
    H : numpy.ndarray
        Hamiltonian of the system.
    Nk : int
        System's dimension.
    tau : float
        Time step for integration.
    T : float
        Total time for propagation.
    tb : numpy.ndarray
        Time array for control.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray, optional
        Target state (default is None).
    TrF : bool, optional
        Whether to use the transfer function (default is False).
    """

    def __init__(self, t, ub, H, Nk, tau, T, tb, psi0, psit=None, TrF=False):
        """ Initialize the propagation of the quantum system."""
        self.t = t                                                                 # Integration time
        self.ub = ub                                                               # Control
        self.H = H                                                                 # Hamiltonian
        self.Nk = Nk                                                               # System's dimension
        self.tau = tau                                                             # Time step for integration
        self.T = T                                                                 # Total time for propagation
        self.tb = tb                                                               # Time array for control
        self.psi0 = psi0                                                           # Initial state
        self.psit = psit                                                           # Target state
        self.TrF = TrF                                                             # Transfer function

    def get_fk_pk(self, k, n):
        """This function compute the fk(t) or pk(t) function."""
        if(self.TrF == True):
            if(self.t[n] < self.tb[k]):
                res = 0
            if(self.t[n] >= self.tb[k] and self.t[n] <= self.tb[k+1]):
                res = (1.0 - np.exp(-(self.t[n]-self.tb[k])/self.tau)) 
            if(self.t[n] > self.tb[k+1]):
                res = np.exp(-(self.t[n]-self.tb[k+1])/self.tau)\
                      *(1.0 - np.exp(-self.T/self.tau)) 
        if(self.TrF == False):
            if(self.t[n] >= self.tb[k] and self.t[n] < self.tb[k+1]):
                res = 1.0
            else:
                res = 0.0
        return res

    def get_u(self):
        """This function compute the control see by the Fluxonium."""
        u = np.zeros(self.t.size)
        for n in range(self.t.size):
            res = 0
            for k in range(self.tb.size-1):
                res = res + self.ub[k]*self.get_fk_pk(k,n)
            u[n] = res
        return u

    def get_propagator(self, ut, dt, n):
        """This function compute U=exp(-iHdt)."""
        H0   = self.H[0]                                                           # Non controlled part of Hamiltonian
        H1   = self.H[1]                                                           # Controlled part of Hamiltonian
        cosX = self.H[2]                                                           # cosX matrix
        U    = np.zeros((self.Nk,self.Nk), dtype = 'complex_')                     # Initialisation of propagator
        H = H0 + cosX + ut*H1                                                      # Hamiltonian
        U = linalg.expm(-1j*2.0*np.pi*H*dt)                                        # Propagator at each time step
        return U

    def get_state(self):
        """ State at time t."""
        Nt     = self.t.size                                                       # Number of time points
        C      = np.zeros((self.Nk,Nt), dtype = 'complex_')                        # Initialisation of State
        C[:,0] = self.psi0                                                         # Initial condition
        u      = self.get_u()                                                      # Get time continuous control
        for n in range(Nt-1):
            dt       = self.t[n+1]-self.t[n]                                       # Time step
            ut       = u[n]                                                        # Discrete control
            U        = self.get_propagator(ut, dt, n)                              # Get propagator
            C[:,n+1] = U @ C[:,n]                                                  # Forward propagation at each time step                 
        return C

    def get_adjoint_state(self, psif):
        """ Adjoint state at time t."""
        Nt       = self.t.size                                                     # Number of time points
        u      = self.get_u()                                                      # Get time continuous control
        rev_time = list(reversed(range(len(self.t))))                              # Backward time
        del(rev_time[-1])                                                          # Delete the first index to compute D(0)
        D        = np.zeros((self.Nk,Nt), dtype = 'complex_')                      # Initialization of adjoint state
        D[:,-1]  = -(self.psit.conj().T @ psif) * self.psit                        # Final condition for the adjoint state
        for n in rev_time:
            dt       = self.t[n]-self.t[n-1]                                       # Time step
            ut       = u[n-1]                                                      # Discrete control
            U        = self.get_propagator(ut, dt, n)                              # Get propagator
            D[:,n-1] = U.conj().T @ D[:,n]                                         # Backward propagation at each time step     
        return D

    def get_fidelity_sm(self):
        """ Fidelity (Square Modulus) at time tf."""
        C = self.get_state()
        F = 1-abs(C[:,-1].conj().T @ self.psit)**2
        return F.real

    def get_fidelity_derivative_sm_pmp(self):
        """ Fidelity at time tf."""
        H0       = self.H[0]                                                       # Non controlled part of Hamiltonian
        H1       = self.H[1]                                                       # Controlled part of Hamiltonian
        cosX     = self.H[2]                                                       # cosX Matrix
        El       = self.H[3]                                                       # Inductive energy of the fluxonium
        Nt       = self.t.size                                                     # Number of time points
        u        = self.get_u()                                                    # Get time continuous control
        dF       = np.zeros(Nt)                                                    # Initialization of dF tab
        dFk      = np.zeros(self.tb.size)                                          # Initialization of dF tab
        C        = self.get_state()                                                # Get state
        D        = self.get_adjoint_state(C[:,-1])                                 # Get adjoint state
        for n in range(Nt):
            ut    = u[n]                                                           # Discrete control
            dM    = H1                                                             # Derivative of H wrt u
            dF[n] = (D[:,n].conj().T @ dM @ C[:,n]).imag                           # Compute du for each time step given by PMP
        for k in range(self.tb.size):
            if(k==self.tb.size-1):
                dFk[k] = 0
            else:
                for n in range(Nt):
                    dFk[k] = dFk[k] + dF[n]*self.get_fk_pk(k,n)
        return dFk

#----------------------------------------------------------------------------------
#
# Functions
#
#----------------------------------------------------------------------------------

def Cost(ub, t, H, Nk, tau, T, tb, psi0, psit, TrF):
    """
    Compute the cost from the propagation class.

    Parameters
    ----------
    ub : numpy.ndarray
        Discrete control.
    t : numpy.ndarray
        Integration time.
    H : numpy.ndarray
        Hamiltonian of the system.
    Nk : int
        System's dimension.
    tau : float
        Time step for integration.
    T : float
        Total time for propagation.
    tb : numpy.ndarray
        Time array for control.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray
        Target state.
    TrF : bool
        Whether to use the transfer function.

    Returns
    -------
    float
        The computed fidelity.
    """
    dyn = propagation(t, ub, H, Nk, tau, T, tb, psi0, psit, TrF)
    res = dyn.get_fidelity_sm()
    return res

def dCost(ub, t, H, Nk, tau, T, tb, psi0, psit, TrF):
    """
    Compute the direction of the gradient algorithm from the propagation class.

    Parameters
    ----------
    ub : numpy.ndarray
        Discrete control.
    t : numpy.ndarray
        Integration time.
    H : numpy.ndarray
        Hamiltonian of the system.
    Nk : int
        System's dimension.
    tau : float
        Time step for integration.
    T : float
        Total time for propagation.
    tb : numpy.ndarray
        Time array for control.
    psi0 : numpy.ndarray
        Initial condition for the state.
    psit : numpy.ndarray
        Target state.
    TrF : bool
        Whether to use the transfer function.

    Returns
    -------
    numpy.ndarray
        The computed gradient direction.
    """
    dyn = propagation(t, ub, H, Nk, tau, T, tb, psi0, psit, TrF)
    res = dyn.get_fidelity_derivative_sm_pmp()
    return res

def phin(n, x_tab, lamb):
    """
    Compute the wave function.

    Parameters
    ----------
    n : int
        Quantum number.
    x_tab : numpy.ndarray
        Position array.
    lamb : float
        Scaling parameter.

    Returns
    -------
    numpy.ndarray
        The computed wave function.
    """
    phi_n = np.zeros(x_tab.size, dtype='complex_')
    for ix, x in enumerate(x_tab):
        coef1 = 1.0 / (np.sqrt(2.0 ** n * math.factorial(n)))
        coef2 = np.sqrt(lamb) * (1.0 / np.pi) ** 0.25
        coef3 = np.exp(-0.5 * lamb ** 2 * x ** 2)
        phi_n[ix] = coef1 * coef2 * coef3
    Herm = special.hermite(n)
    phi_n *= Herm(lamb * x_tab)
    return phi_n

def callback(xk):
    """
    Print the progress of the algorithm at each step.

    Parameters
    ----------
    xk : numpy.ndarray
        Current parameter values.
    """
    dyn = propagation(t, xk, H, Nk, tau, T, tb, psi0, psit, TrF)
    F = 1 - dyn.get_fidelity_sm()
    print('Iteration: %d, 1-F1=%.15f' % (callback.iter, F))
    callback.iter += 1

#----------------------------------------------------------------------------------
#
# Parameters of the physical system
#
#----------------------------------------------------------------------------------

''' Constant '''
El = 1.00                                                                          # Parameter E_L
Ej = 1.25                                                                          # Parameter E_J
Ec = 0.70                                                                          # Parameter E_C
h  = 1.0545718e-34                                                                 # Reduced Plank constant
lamb = (El/(8.0*Ec))**(0.25)                                                       # Change of variable constant

''' Data '''
Nk = 50                                                                            # Dimension of the system

''' Hamiltonian '''
flux = FLUX(Nk, Ec, Ej, El)                                                        # Create our Fluxonium system
H   = flux.get_H()                                                                 # Fluxonium system's Hamiltonian

''' Caracteristic Time '''
val,vp = flux.get_ground_state()                                                   # Eigendecomposition of H for u=0
DE = val[1] - val[0]
tau = (2.0*np.pi)/DE

''' Time parameters '''
tau = 0.1                                                                          # Time step for integration
T = 0.5                                                                            # Time interval for control updates
tf = 10.0                                                                          # Final time
tb = np.arange(0, tf + T, T)                                                       # Time array for control updates
dt = 0.01                                                                          # Time step for state propagation
t = np.arange(0, tf + dt, dt)                                                      # Time array for state propagation

''' Initial state '''
psi0 = vp[:, 0]                                                                    # Initial state vector
pop0 = abs(vp.conj().T @ psi0)**2                                                  # Population of the initial state

''' Target state '''
psit = vp[:, 1]                                                                    # Target state vector

'''  Define control seen by the Fluxonium '''
u0 = 0.3 * np.ones(tb.size)                                                        # Initial control values
TrF = True                                                                         # Flag to indicate the use of the transfer function


''' Optimisation parameter '''
maxt = 20                                                                          # Maximum number of iterations
meth = 'L-BFGS-B'                                                                  # Method of the optimization algorithm (here BFGS)
bds = []                                                                           # Bound sequence
for n in range(tb.size):
    bds.append((-0.5,0.5))                                                         # Control Phase between -\pi and \pi (None or comment if no bound is required)

#----------------------------------------------------------------------------------
#
# Solve the optimal control problem
#
#----------------------------------------------------------------------------------

''' OCT '''
callback.iter = 0
sol   = minimize(Cost, u0, method=meth, jac=dCost,\
                 args=(t, H, Nk,tau, T, tb, psi0, psit,TrF),\
                 callback=callback,\
                 bounds=bds, tol=1e-17, options={'maxiter': maxt})                 # Minimization algorithm with a specified gradient
ub    = sol.x                                                                      # save the optimal control field into another variable
dyn   = propagation(t, ub, H, Nk, tau, T, tb, psi0, psit, TrF)                     # Create propagator class for u_optimal
C     = dyn.get_state()                                                            # State associated to optimal control
u     = dyn.get_u()
print(1-dyn.get_fidelity_sm())

''' Results '''
Cp = vp.conj().T @ C[:,-1]
popf1 = abs(Cp)**2                                                                 # Final population

#----------------------------------------------------------------------------------
#
# Plot 
#
#----------------------------------------------------------------------------------

fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(1,2)

#--- First column ---
ax0 = plt.subplot(gs[0,0])
ax0.plot(t, u, color='C0')#, linestyle='dotted')
ax0.set_xlim(0,t[-1])
ax0.text(0.05, 0.95, r'\bf{(a)}',transform=ax0.transAxes,fontsize=15,fontweight='bold',va='top')
ax0.axhline(0, linestyle='--', color='black', linewidth=0.5)
#ax0.set_xticks([0, 2.5, tf]) 
ax0.set_xlabel(r'$t \ (ns)$',fontsize=30) 
ax0.set_ylabel(r'$\theta(t)$',fontsize=30) 

#--- Second column ---
ax2 = plt.subplot(gs[0,1])
width = 0.8
qq = np.arange(0, 5) 
ax2.bar(qq, popf1[0:5], width, color='C0', edgecolor='black') 
ax2.text(0.05, 0.95, r'\bf{(b)}',transform=ax2.transAxes,fontsize=15,fontweight='bold',va='top')
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_xticks([0, 1, 2, 3, 4, 5]) 
ax2.set_ylim(0,1)
ax2.grid()
ax2.set_axisbelow(True)
ax2.set_xlabel(r'$|\zeta_n\rangle$',fontsize=30) 
ax2.set_ylabel(r'$| b_{n} |^2$',fontsize=30) 
ax2.set_axisbelow(True)

ax0.set_box_aspect(1)
ax2.set_box_aspect(1)
plt.subplots_adjust(hspace=.0, wspace=0.3)
plt.show()
