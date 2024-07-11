#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: TIR_discret.py
# Author: Etienne Dionis
# Date: June 8, 2024
# Description: Shooting method for discrete PMP of a two level systemes with
#              two controls parameters.

#------------------------------------------------------------------------------
#
# Modules 
#
#------------------------------------------------------------------------------

import numpy as np
import nutopy as nt
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import optimize
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':20})
rc('text', usetex=True)

#------------------------------------------------------------------------------
#
# Functions
#
#------------------------------------------------------------------------------

def control(t, Ix, Iy, Iz, T):
    """
    Compute optimal constant control.

    Parameters
    ----------
    t : float
        Unknown parameter to find.
    Ix : float
        Moment of inertia around the x-axis.
    Iy : float
        Moment of inertia around the y-axis.
    Iz : float
        Moment of inertia around the z-axis.
    T : float
        Time or angle parameter.

    Returns
    -------
    float
        Optimal control equation result.

    Notes
    -----
    The optimal control `u` is given by:
    
        u = 2 * arctan(t)
    
    The function computes the equation to find the roots:

        eq = Iy * (t**2 - 1) * sin(T) + Ix * 2 * t * sin(T) + Iz * (t**2 + 1) * (1 - cos(T))
    """
    eq = (Iy * (t**2 - 1) * np.sin(T) + 
          Ix * 2 * t * np.sin(T) + 
          Iz * (t**2 + 1) * (1 - np.cos(T)))
    return eq

def fonction(t, x, u):
    """
    Compute dx/dt = f(t, x) system.

    Parameters
    ----------
    t : float
        Time interval.
    x : ndarray
        State and adjoint state, x = (y, p).
    u : float
        Optimal control, piecewise constant control.

    Returns
    -------
    dx : ndarray
        Dynamics of the extremal system, dx/dt = f(t, x).
    """
    # --- State and adjoint state ---
    dx = np.zeros(x.size)                                                          # Initialization of state and adjoint state 

    # --- Dynamic ---
    dx[0] = np.sin(u) * x[2]                                                       # Time derivative of x
    dx[1] = -np.cos(u) * x[2]                                                      # Time derivative of y
    dx[2] = np.cos(u) * x[1] - np.sin(u) * x[0]                                    # Time derivative of z
    dx[3] = np.sin(u) * x[5]                                                       # Time derivative of px
    dx[4] = -np.cos(u) * x[5]                                                      # Time derivative of py
    dx[5] = np.cos(u) * x[4] - np.sin(u) * x[3]                                    # Time derivative of pz
    return dx

def objectif(q, x_init, y_init, z_init, t0, N):
    """
    Find initial adjoint condition and final time, that state reach the target.

    This function uses a shooting method to determine the initial adjoint state
    and the final time such that the state reaches a specified target. The process 
    involves solving a set of differential equations iteratively.

    Parameters
    ----------
    q : array-like, shape (4,)
        Initial guesses for the adjoint state and final time.
        q[0] : float
            Initial guess for the final time (tf).
        q[1] : float
            Initial guess for px.
        q[2] : float
            Initial guess for py.
        q[3] : float
            Initial guess for pz.
    x_init : float
        Initial x position.
    y_init : float
        Initial y position.
    z_init : float
        Initial z position.
    t0 : float
        Initial time.
    N : int
        Number of time intervals for the integration.

    Returns
    -------
    out : ndarray, shape (4,)
        Output of the shooting function, representing the target conditions:
        out[0] : float
            Final x position, should be 0.
        out[1] : float
            Final y position, should be 1.
        out[2] : float
            Final z position, should be 0.
        out[3] : float
            Hamiltonian at the final time, should be 0.
    """
    #--- Initialisation ---
    out  = np.zeros(q.size)                                                        # Output of shooting function
    tf   = q[0]                                                                    # Initial guess of tf
    px   = q[1]                                                                    # Initial guess of der
    py   = q[2]                                                                    # Initial guess of dei
    pz   = q[3]                                                                    # Initial guess of dgr
    time = np.linspace(t0, tf, N+1)                                                # Time initialization
    T    = time[1]-time[0]                                                         # Sampling Period
    
    #--- Solve dynamics ---
    for i in range(0,N):                                                           # Loop on time interval subdivision
        # First iteration
        if(i==0):  
            dt = time[i+1] - time[i]                                               # Sampling period
            Ix = -py*z_init + pz*y_init                                            # Ix(t)
            Iy = px*z_init - pz*x_init                                             # Iy(t)
            Iz = -px*y_init + py*x_init                                            # Iz(t)           
            res = optimize.root_scalar(control,\
                                       args=(Ix,Iy,Iz,dt),\
                                       method='secant', x0=np.pi, x1=np.pi/2)      # Find optimal control
            root = res.root                                                        # Root of the control function
            u = 2.0*np.arctan(root)                                                # Optimal control
            x0 = np.array([x_init, y_init, z_init, px, py, pz])                    # Initial condition
            sol = nt.ivp.exp(fonction, time[i+1], time[i], x0, args=(u))           # RK4 resolution
            x  = sol.xf[0]                                                         # Final x
            y  = sol.xf[1]                                                         # Final y
            z  = sol.xf[2]                                                         # Final z
            px = sol.xf[3]                                                         # Final px
            py = sol.xf[4]                                                         # Final py
            pz = sol.xf[5]                                                         # Final pz
            
        # Others iterations
        else:
            dt = time[i+1] - time[i]                                               # Sampling period
            Ix = -py*z + pz*y                                                      # Ix(t)
            Iy = px*z - pz*x                                                       # Iy(t)
            Iz = -px*y + py*x                                                      # Iz(t) 
            res = optimize.root_scalar(control,\
                                       args=(Ix,Iy,Iz,dt),\
                                       method='secant', x0=np.pi, x1=np.pi/2)      # Find optimal control
            root = res.root                                                        # Root of the control function
            u = 2.0*np.arctan(root)                                                # Optimal control
            x0 = np.array([x, y, z, px, py, pz])                                   # Initial condition
            sol = nt.ivp.exp(fonction, time[i+1], time[i], x0, args=(u))           # RK4 method
            x  = sol.xf[0]                                                         # Final x
            y  = sol.xf[1]                                                         # Final y
            z  = sol.xf[2]                                                         # Final z
            px = sol.xf[3]                                                         # Final px
            py = sol.xf[4]                                                         # Final py
            pz = sol.xf[5]                                                         # Final pz

    #--- Hamiltonian at final time ---
    Hp = np.cos(u)*(pz*y - py*z) + np.sin(u)*(px*z - pz*x) - 1.0                   # Hamiltonian at final time
 
    #--- Target ---
    out[0] = x                                                                     # X=0                                                               
    out[1] = y-1                                                                   # Y=1
    out[2] = z                                                                     # Z=0
    out[3] = Hp                                                                    # Hp=0
    return out                                                                     # Objective of shooting method

#----------------------------------------------------------------------------------
#
# Parameters of the physical system
#
#----------------------------------------------------------------------------------

''' Initial Condition '''
x0 = np.array([1., 0, 0])                                                          # Initial state

''' Time '''
tf_opti  = (np.sqrt(3.0)*np.pi)/2.0                                                # Final time (optimal solution)
t0       = 0                                                                       # Initial time
N        = 3                                                                       # N number of controls
t_tot    = []                                                                      # Total time
time     = np.linspace(0, tf_opti, N+1)                                            # Time initialization
T        = time[1]-time[0]                                                         # Sampling Period

''' Trajectory '''
COS_U = []                                                                         # Total cosinus(u)
SIN_U = []                                                                         # Total sinus(u)
X     = []                                                                         # Total x
Y     = []                                                                         # Total y 
Z     = []                                                                         # Total z

#----------------------------------------------------------------------------------
#
# Solve the optimal control problem
#
#----------------------------------------------------------------------------------

#--- Shooting method ---
q0 = np.array([2.7,1,1/np.sqrt(3.),-1])                                            # Initial guess
racine = nt.nle.solve(objectif,q0,args=(x0[0],x0[1],x0[2],t0,N))                   # Newton Method
tf   = racine.x[0]                                                                 # Final time
px   = racine.x[1]                                                                 # Initial theta
py   = racine.x[2]                                                                 # Initial phi
pz   = racine.x[3]                                                                 # Initial phi

#--- Solving the system ---
time = np.linspace(0, tf, N+1)                                                     # Time initialization
T    = time[1]-time[0]                                                             # Sampling Period
for i in range(0,N):                                                               # Loop on time interval subdivision
    # First iteration
    if(i==0):                                                                      
        dt = time[i+1] - time[i]                                                   # Sampling period
        Ix = -py*x0[2] + pz*x0[1]                                                  # Ix(t)
        Iy = px*x0[2] - pz*x0[0]                                                   # Iy(t)
        Iz = -px*x0[1] + py*x0[0]                                                  # Iz(t)           
        res = optimize.root_scalar(control,\
                                   args=(Ix,Iy,Iz,dt),\
                                   method='secant', x0=np.pi, x1=np.pi/2)          # Find optimal control
        root = res.root                                                            # Root of the control function
        u = 2.0*np.arctan(root)                                                    # Optimal control
        y0 = np.array([x0[0], x0[1], x0[2], px, py, pz])                           # Initial condition
        sol = nt.ivp.exp(fonction, time[i+1], time[i], y0, args=(u))               # RK4 method
        x  = sol.xout[:,0]                                                         # Final x
        y  = sol.xout[:,1]                                                         # Final y
        z  = sol.xout[:,2]                                                         # Final z
        px = sol.xout[:,3]                                                         # Final px
        py = sol.xout[:,4]                                                         # Final py
        pz = sol.xout[:,5]                                                         # Final pz
        t  = sol.tout                                                              # Time tab
        cos_u = np.cos(u)                                                          # Cos(u)
        sin_u = np.sin(u)                                                          # Sin(u)
        COS_U = np.append(COS_U, cos_u)                                            # Add cos(u) to total cos(u) control
        SIN_U = np.append(SIN_U, sin_u)                                            # Add sin(u) to total sin(u) control
        X = np.append(X,x)                                                         # Add to total Bloch coordinates
        Y = np.append(Y,y)                                                         # Add to total Bloch coordinates
        Z = np.append(Z,z)                                                         # Add to total Bloch coordinates
        t_tot = np.append(t_tot, t)                                                # Add to total time
        
    # Others iterations
    else:
        dt  = time[i+1] - time[i]                                                  # Sampling period
        Ix  = -py[-1]*z[-1] + pz[-1]*y[-1]                                         # Ix(t)
        Iy  = px[-1]*z[-1] - pz[-1]*x[-1]                                          # Iy(t)
        Iz  = -px[-1]*y[-1] + py[-1]*x[-1]                                         # Iz(t) 
        res = optimize.root_scalar(control,\
                                   args=(Ix,Iy,Iz,dt),\
                                   method='secant', x0=np.pi, x1=np.pi/2)          # Find optimal control
        root = res.root                                                            # Root of the control function
        u = 2.0*np.arctan(root)                                                    # Optimal control
        x0 = np.array([x[-1], y[-1], z[-1], px[-1], py[-1], pz[-1]])               # Initial condition
        sol = nt.ivp.exp(fonction, time[i+1], time[i], x0, args=(u))               # RK4 method
        x  = sol.xout[:,0]                                                         # Final x
        y  = sol.xout[:,1]                                                         # Final y
        z  = sol.xout[:,2]                                                         # Final z
        px = sol.xout[:,3]                                                         # Final px
        py = sol.xout[:,4]                                                         # Final py
        pz = sol.xout[:,5]                                                         # Final pz
        t  = sol.tout                                                              # Time tab
        cos_u = np.cos(u)                                                          # Cos(u)
        sin_u = np.sin(u)                                                          # Sin(u)
        COS_U = np.append(COS_U, cos_u)                                            # Add cos(u) to total cos(u) control
        SIN_U = np.append(SIN_U, sin_u)                                            # Add sin(u) to total sin(u) control
        X = np.append(X,x)                                                         # Add to total Bloch coordinates
        Y = np.append(Y,y)                                                         # Add to total Bloch coordinates
        Z = np.append(Z,z)                                                         # Add to total Bloch coordinates
        t_tot = np.append(t_tot, t)                                                # Add to total time
        
#--- Trajectory and optimal control ---
ux = COS_U                                                                         # Ux control
uy = SIN_U                                                                         # Uy control
x = X                                                                              # X Bloch coordinates
y = Y                                                                              # Y Bloch coordinates
z = Z                                                                              # Z Bloch coordinates
t = t_tot                                                                          # Total time

#----------------------------------------------------------------------------------
#
# Plot 1
#
#----------------------------------------------------------------------------------

fig = plt.figure(1,figsize=(14.4,7.49))
gs = gridspec.GridSpec(1,1)
ax = plt.subplot(gs[0,0])                                                         

ax.axvline(tf_opti, color='black',linewidth=3)
for i in range(0,N):
    if(i==0):
        ax.plot(np.linspace(time[i],time[i+1],100),ux[i]*np.ones(100), color='C0',linewidth=4)
        ax.plot(np.linspace(time[i],time[i+1],100),uy[i]*np.ones(100), color='C1',linewidth=4)
    else:
        ax.plot(np.linspace(time[i],time[i+1],100),ux[i]*np.ones(100), color='C0',linewidth=4)
        ax.plot(np.linspace(time[i],time[i+1],100),uy[i]*np.ones(100), color='C1',linewidth=4)
ax.scatter(time[:-1], ux, color='C0',s=100)
ax.scatter(time[:-1], uy, color='C1',s=100)
ax.set_ylim(-0.03, 1.03)
ax.set_xlim(0, tf_opti+0.05)
ax.set_ylabel(r'$u_x(t), \ u_y(t)$',fontsize=30)
ax.set_xlabel(r'$t$',fontsize=30)
ax.set_box_aspect(1)

#----------------------------------------------------------------------------------
#
# Plot 2
#
#----------------------------------------------------------------------------------

fig = plt.figure(2,figsize=(14.4,7.49))
gs = gridspec.GridSpec(1,1)
ax = plt.subplot(gs[0,0])                                                         

ax.plot(t, x, color='C0',linewidth=2)
ax.plot(t, y, color='C1',linewidth=2)
ax.plot(t, z, color='C2',linewidth=2)
ax.set_ylim(-1, 1.)
ax.set_xlim(0, tf_opti+0.05)
ax.set_ylabel(r'$x(t), \ y(t), \ z(t)$',fontsize=30)
ax.set_xlabel(r'$t$',fontsize=30)
ax.set_box_aspect(1)
plt.show()
