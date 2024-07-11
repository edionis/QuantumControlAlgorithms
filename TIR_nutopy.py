#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: TIR_nutopy.py
# Author: Etienne Dionis
# Date: June 8, 2024
# Description: Shooting method for continuous PMP of a two level systemes with
#              two controls parameters (solve with nutopy package).

#------------------------------------------------------------------------------
#
# Modules 
#
#------------------------------------------------------------------------------

import numpy as np
import nutopy as nt
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

#------------------------------------------------------------------------------
#
# Functions
#
#------------------------------------------------------------------------------

def dynamics(t, x):
    """
    Compute the dynamics of the system dx/dt = f(t, x).

    This function computes the time derivatives of the state and adjoint state 
    variables based on the current state and time.

    Parameters
    ----------
    t : float
        Current time.
    x : array-like, shape (6,)
        Current state and adjoint state vector, where:
        x[0] : float
            State variable y.
        x[1] : float
            State variable y.
        x[2] : float
            State variable y.
        x[3] : float
            Adjoint state variable p.
        x[4] : float
            Adjoint state variable p.
        x[5] : float
            Adjoint state variable p.

    Returns
    -------
    dx : ndarray, shape (6,)
        Time derivatives of the state and adjoint state variables, where:
        dx[0] : float
            Time derivative of state variable y.
        dx[1] : float
            Time derivative of state variable y.
        dx[2] : float
            Time derivative of state variable y.
        dx[3] : float
            Time derivative of adjoint state variable p.
        dx[4] : float
            Time derivative of adjoint state variable p.
        dx[5] : float
            Time derivative of adjoint state variable p.
    """
    # Initialization
    dx = np.zeros(x.size)
    
    # Dynamics
    dx[0] = ((x[3] * x[2] - x[5] * x[0]) / np.sqrt((x[5] * x[1] - x[4] * x[2]) ** 2 +
             (x[3] * x[2] - x[5] * x[0]) ** 2)) * x[2]
    dx[1] = -((x[5] * x[1] - x[4] * x[2]) / np.sqrt((x[5] * x[1] - x[4] * x[2]) ** 2 +
             (x[3] * x[2] - x[5] * x[0]) ** 2)) * x[2]
    dx[2] = ((x[5] * x[1] - x[4] * x[2]) / np.sqrt((x[5] * x[1] - x[4] * x[2]) ** 2 +
             (x[3] * x[2] - x[5] * x[0]) ** 2)) * x[1] - ((x[3] * x[2] - x[5] * x[0]) /
             np.sqrt((x[5] * x[1] - x[4] * x[2]) ** 2 +
             (x[3] * x[2] - x[5] * x[0]) ** 2)) * x[0]
    dx[3] = ((x[3] * x[2] - x[5] * x[0]) / np.sqrt((x[5] * x[1] - x[4] * x[2]) ** 2 +
             (x[3] * x[2] - x[5] * x[0]) ** 2)) * x[5]
    dx[4] = -((x[5] * x[1] - x[4] * x[2]) / np.sqrt((x[5] * x[1] - x[4] * x[2]) ** 2 +
             (x[3] * x[2] - x[5] * x[0]) ** 2)) * x[5]
    dx[5] = ((x[5] * x[1] - x[4] * x[2]) / np.sqrt((x[5] * x[1] - x[4] * x[2]) ** 2 +
             (x[3] * x[2] - x[5] * x[0]) ** 2)) * x[4] - ((x[3] * x[2] - x[5] * x[0]) /
             np.sqrt((x[5] * x[1] - x[4] * x[2]) ** 2 +
             (x[3] * x[2] - x[5] * x[0]) ** 2)) * x[3]
    return dx

def shooting_function(x, x0):
    """
    Compute the shooting function to find the initial adjoint condition and final time 
    that allow the state to reach the target.

    Parameters
    ----------
    x : array-like
        Initial adjoint state and final time to be determined by Newton's method.
        - x[0] : float
            Final time tf.
        - x[1] : float
            Initial adjoint state px.
        - x[2] : float
            Initial adjoint state py.
        - x[3] : float
            Initial adjoint state pz.
    x0 : array-like
        Shooting function target, where:
        - x0[0] : float
            Target state x(tf).
        - x0[1] : float
            Target state y(tf) - 1.
        - x0[2] : float
            Target state z(tf).
        - x0[3] : float
            Hamiltonian H(tf).

    Returns
    -------
    out : ndarray
        Objective of the shooting function, where:
        - out[0] : float
            Difference between computed and target state x(tf).
        - out[1] : float
            Difference between computed and target state y(tf) - 1.
        - out[2] : float
            Difference between computed and target state z(tf).
        - out[3] : float
            Difference between computed and target Hamiltonian H(tf).
    """
    # Initialization of the shooting parameters
    out   = np.zeros(x.size)                                                       # Output of shooting function
    tf    = x[0]                                                                   # Final time
    px    = x[1]                                                                   # Initial guess of theta
    py    = x[2]                                                                   # Initial guess of phi
    pz    = x[3]                                                                   # Initial guess of phi

    # Compute dynamics 
    y0  = np.array([x0[0], x0[1], x0[2], px, py, pz])                              # Initial condition
    sol = nt.ivp.exp(dynamics, tf, 0, y0)                                          # System's integration 
    xt  = sol.xf[0]                                                                # Final x coordinates
    yt  = sol.xf[1]                                                                # Final y coordinates
    zt  = sol.xf[2]                                                                # Final z coordinates
    pxt = sol.xf[3]                                                                # Final px coordinates
    pyt = sol.xf[4]                                                                # Final py coordinates
    pzt = sol.xf[5]                                                                # Final pz coordinates

    # Hamiltonian at final time 
    den = (pzt*yt - pyt*zt)**2 + (pxt*zt - pzt*xt)**2                              # Normalisation factor of optimal control
    den = np.sqrt(den)                                                             # Square root of normalisation factor of optimal control
    ux = (pzt*yt - pyt*zt)/den                                                     # Final Ux optimal control
    uy = (pxt*zt - pzt*xt)/den                                                     # Final Uy optimal control
    Hp = ux*(pzt*yt - pyt*zt) + uy*(pxt*zt - pzt*xt) - 1.0                         # Hamiltonian at final time  

    # Target
    out[0] = xt                                                                    # X=0                                                               
    out[1] = yt-1                                                                  # Y=1
    out[2] = zt                                                                    # Z=0
    out[3] = Hp                                                                    # Hp=0
    return out                                                                     # Shooting objective

#----------------------------------------------------------------------------------
#
# Parameters of the physical system
#
#----------------------------------------------------------------------------------

x0 = np.array([1, 0, 0])                                                           # Initial state
t0 = 0                                                                             # Initial time

#----------------------------------------------------------------------------------
#
# Solve the optimal control problem
#
#----------------------------------------------------------------------------------

#Shooting method
p0  = np.array([1,1/np.sqrt(3.0),1,-1])                                            # Initial guess (tf, px(0), py(0), pz(0))
res = nt.nle.solve(shooting_function, p0, args=(x0))                               # Find the roots of the shooting function
print(res)
tf  = res.x[0]                                                                     # Final time
px  = res.x[1]                                                                     # Initial theta
py  = res.x[2]                                                                     # Initial phi
pz  = res.x[3]                                                                     # Initial phi

#Solving the system
#--- Dynamics integrations ---
y0  = np.array([x0[0], x0[1], x0[2], px, py, pz])                                  # Initial condition
sol = nt.ivp.exp(dynamics, tf, 0, y0)                                              # System's integration  
x   = sol.xout[:,0]                                                                # x coordinate
y   = sol.xout[:,1]                                                                # y coordinate
z   = sol.xout[:,2]                                                                # z coordinate
px  = sol.xout[:,3]                                                                # px coordinate
py  = sol.xout[:,4]                                                                # py coordinate
pz  = sol.xout[:,5]                                                                # pz coordinate
t   = sol.tout                                                                     # Time
#--- Optimal control ---
den = (pz*y - py*z)**2 + (px*z - pz*x)**2                                          # Normalisation factor of optimal control
den = np.sqrt(den)                                                                 # Square root of normalisation factor of optimal control
ux = (pz*y - py*z)/den                                                             # Final Ux optimal control
uy = (px*z - pz*x)/den                                                             # Final Uy optimal control

# Distance to target
dist = np.sqrt(x[-1]**2 + (y[-1]-1)**2 + z[-1]**2)
print('Distance to target : %.3e' %(dist))

#----------------------------------------------------------------------------------
#
# Plot 
#
#----------------------------------------------------------------------------------

fig = plt.figure(figsize=(10,7))
gs = gridspec.GridSpec(1, 2)                                                        

###################################################################################
ax0 = plt.subplot(gs[0,0])
ax0.plot(t, x, color='C0', label=r'$x$')
ax0.plot(t, y, color='C1', label=r'$y$')
ax0.plot(t, z, color='C2', label=r'$z$')
ax0.set_xlim(0,t[-1])
ax0.set_ylim(-1,1)
ax0.text(0.05, 0.95, r'\bf{(a)}',transform=ax0.transAxes,fontsize=15,\
         fontweight='bold',va='top')
ax0.set_xlabel(r'$t$')
ax0.set_ylabel(r'$x,y,z$')
ax0.legend()

###################################################################################
ax1 = plt.subplot(gs[0,1])
ax1.text(0.05, 0.95, r'\bf{(b)}',transform=ax1.transAxes,\
         fontsize=15,fontweight='bold',va='top')
ax1.set_xlim(0,t[-1])
ax1.set_ylim(0,1)
ax1.plot(t, ux, color='C0', label=r'$u_x$')
ax1.plot(t, uy, color='C3', label=r'$u_y$')
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$u_x, u_y$')
ax1.legend()


ax0.set_box_aspect(1)
ax1.set_box_aspect(1)
plt.subplots_adjust(hspace=.0, wspace=0.3)
plt.show()
