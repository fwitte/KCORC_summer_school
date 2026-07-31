"""
This is a Python design tool for the design of an axial impulse single stage small scale ORC turbine.
Accompanies the PhD thesis of J.Spale, 2024. CTU in Prague, supervisor: prof.Ing.M.Kolovratnik, CSc.
for questions, send an email to: Jan.Spale@cvut.cz
"""
from __future__ import print_function

# Importing the libraries
import json
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import timeit
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import CoolProp.CoolProp as CP
from tabulate import tabulate
from units import convert
from scipy.interpolate import griddata
from deap import base, creator, tools, algorithms
from multiprocessing import Pool
import random

# Set the REFPROP path
RP_path = "c:\\Program Files (x86)\\REFPROP" # Change this to your REFPROP installation path

# cycle boundary conditions, fixed
p_inlet = convert(2500,"kPa","Pa") # kPa
T_inlet = convert(150,"C","K") # C
p_outlet = convert(475,"kPa","Pa") # kPa
m_dot = 5.025  # kg/s
fluid = "R1233ZDE" # HFO
z = [1.0] # [-] molar fraction of the fluid

# Design parameters, optimization variables
n_design = 3000  # rpm
D_mid = 0.5  # m
alpha_stator = 11.5 # deg
beta_rotor = 23  # deg
beta_rotor_2 = 25 # deg
eta_guess = 0.65  # [-]
e = 0.305 # [-] partial admission factor
sigma = 1 # [-] solidity
chord_opt = 0.025 # m

RUN_GA = False #True to run the genetic algorithm optimization, False to run a single design point
PLOTTING = False #true to load best res from file and plot
ENABLE_GAMMA_PLOTS = False  #true to enable gamma plots
USE_CAPACITY_NOZZLE = False  #true to use capacity based nozzle design

def meanline_design(D_mid, n_design, alpha_stator, eta_guess, p_inlet, T_inlet, beta_rotor, m_dot, e, sigma, beta_rotor_2,chord):
    #initialize REFPROP
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX']) # Instantiate the REFPROP function library
    RP.SETPATHdll(os.environ['RPPREFIX']) # Set the path to the folder containing the REFPROP shared library
    RP.SETFLUIDSdll(fluid)  # Set the fluids
    baseSI = RP.GETENUMdll(0, "MASSBASESI").iEnum # Get the base SI units
    iMass = 0; iFlag = 0 # Set the mass and flag to 0

    def check_err(r): # Function to check the error code from REFPROP
        if r.ierr > 0:
            raise ValueError(r.herr)

    #initialize state variables
    h,s,T,p,rho,a,Z,Ma,Pr = np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5)

    # turbine inlet ([0])
    p[0] = p_inlet # Pa
    T[0] = T_inlet # K
    r = RP.REFPROPdll(fluid,"PT","H;S;D;W;Pr",baseSI,iMass,iFlag,p[0],T[0],z); check_err(r)
    h[0],s[0],rho[0],a[0],Pr[0] = r.Output[0:5] # J/kg, J/kgK, kg/m3, m/s
    Ma[0] = 0 # [-], Mach number at the inlet
    

    # stator isentropic expansion ([1])
    p[1] = p_outlet # Pa
    s[1] = s[0] # J/kgK, isentropic entropy
    r = RP.REFPROPdll(fluid,"PS","H;T;D;W;Z",baseSI,iMass,iFlag,p[1],s[1],z); check_err(r) # isentropic expansion
    h[1],T[1],rho[1],a[1] = r.Output[0:4] # J/kg, K, kg/m3, m/s

    #velocities
    dh_stage = h[0]-h[1] # J/kg, isentropic enthalpy drop
    c_is = np.sqrt(2*dh_stage) # m/s, isentropic velocity
    U = n_design*np.pi*D_mid/60 # m/s, mean blade speed midspan
    U_over_c_is = U/c_is # [-], ratio of mean blade speed to isentropic velocity

    Ma[1] = c_is/a[1]  # [-], Mach number at the stator outlet (isentropic)
    phi_stator = np.sqrt(1-(0.0029*Ma[1]**3-0.0502*Ma[1]**2+0.2241*Ma[1]-0.0877)) # [-], velocity loss coefficient at the nozzles

    #calculate isentropic expansion quasi-steady 1D, 1000 steps in pressure
    throat_trigger = False
    p_is = np.linspace(p[0],p[1],1000)
    T_is,rho_is,h_is,a_is,c_is,Ma_is,PR_is,v_is = np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is))
    T_act,rho_act,h_act,a_act,c_act,Ma_act,PR_act,s_act = np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is)),np.zeros(len(p_is))
    for i in range(len(p_is)):
        r = RP.REFPROPdll(fluid,"PS","T;D;H;W",baseSI,iMass,iFlag,p_is[i],s[0],z); check_err(r)
        T_is[i],rho_is[i],h_is[i],a_is[i] = r.Output[0:4] # K, kg/m3, J/kg, m/s
        if h_is[i] < h[0]:
            c_is[i] = np.sqrt(2*(h[0]-h_is[i])) # m/s
        else:
            c_is[i] = 0 # m/s
        Ma_is[i] = c_is[i]/a_is[i] # [-]
        PR_is[i] = p_is[i]/p_outlet # [-]
        v_is[i] = 1/rho_is[i] # m3/kg
        #print(f"Ma_is: {Ma_is[i]}")
        #as soon as Ma_is[i] > 1, save the index of the throat as throat_index = i
        if Ma_is[i] > 1 and not throat_trigger:
            throat_index = i
            throat_trigger = True
        if Ma_is[i] < 1:
            h_act[i] = h_is[i] # J/kg
        else:
            h_act[i] = h_is[throat_index] - ((phi_stator**2)*(h_is[throat_index]-h_is[i])) # J/kg
        r = RP.REFPROPdll(fluid,"PH","T;D;S;W",baseSI,iMass,iFlag,p_is[i],h_act[i],z); check_err(r)
        T_act[i],rho_act[i],s_act[i],a_act[i] = r.Output[0:4] # K, kg/m3, J/kg, m/s
        if h_act[i] < h[0]:
            c_act[i] = np.sqrt(2*(h[0]-h_act[i])) # m/s
        else:
            c_act[i] = 0 # m/s
        Ma_act[i] = c_act[i]/a_act[i] # [-]
        PR_act[i] = p_is[i]/p_outlet # [-]

    #calculate the actual expansion quasi-steady 1D, 1000 steps in pressure
    #nozzle out enthalpy
    h_nozzle_out = h[0] - ((phi_stator**2)*(h[0]-h_is[-1]))# J/kg

    #get nozzle out T, rho, a,s, Ma,c
    r = RP.REFPROPdll(fluid,"HP","T;D;W;S",baseSI,iMass,iFlag,h_nozzle_out,p_outlet,z); check_err(r)
    T_nozzle_out,rho_nozzle_out,a_nozzle_out,s_nozzle_out = r.Output[0:4] # K, kg/m3, m/s
    c_nozzle_out = phi_stator*c_is[-1] # m/s
    Ma_nozzle_out = c_nozzle_out/a_nozzle_out # [-]

    #calculate nozzle area, throat, outlet and between
    A_outlet = m_dot/(rho_nozzle_out*max(c_nozzle_out,1e-6)) # m2
    A_outlet_id = m_dot/(rho_is[-1]*c_is[-1]) # m2
    A_throat = m_dot/(rho_is[throat_index]*c_is[throat_index]) # m2
    A_ratio = A_outlet/A_throat # [-]
    A_ratio_id = A_outlet_id/A_throat # [-]

    #velocity triangles
    c1a = c_nozzle_out*np.sin(np.radians(alpha_stator)) # m/s; axial absolute velocity stator outlet
    c1u = c_nozzle_out*np.cos(np.radians(alpha_stator)) # m/s; tangential absolute velocity stator outlet
    #nozzle design
    eps = max(1e-6, min(1.0, e))   # clamp e to (0,1]
    ht_nozzle = m_dot/(rho_nozzle_out*c1a*eps*np.pi*D_mid) # m
    b_nozzles_out = A_outlet/ht_nozzle # m
    b_nozzles_throat = A_throat/ht_nozzle # m

    #appendix
    # Discharge & blockage
    Cd_exit, Ka_exit = 1, 1
    Cd_star, Ka_star = 1, 1

    # Effective total areas 
    A_outlet_eff_total = A_outlet
    A_throat_eff_total = A_throat

    # Convert to geometric totals via A_eff = Cd*Ka*A_geo
    A_outlet_geo_total = A_outlet_eff_total / max(Cd_exit*Ka_exit, 1e-6)
    A_throat_geo_total = A_throat_eff_total / max(Cd_star*Ka_star, 1e-6)

    # Admission discretization via stator pitch
    circumference_s = np.pi*D_mid
    chord_s_ax_loc  = 1.5*chord
    sigma_s_loc     = sigma
    pitch_s_loc     = chord_s_ax_loc / max(sigma_s_loc, 1e-9)

    # Integer number of fed passages that fits requested e
    eps_nom = float(np.clip(e, 1e-6, 1.0))
    no_nozzles_cap = max(1, int(round(eps_nom * circumference_s / max(pitch_s_loc,1e-9))))
    f_active_cap   = no_nozzles_cap * pitch_s_loc
    eps_cap        = f_active_cap / circumference_s

    # Height must satisfy BOTH outlet and throat capacity over admitted arc
    ht_from_out_cap = A_outlet_geo_total / max(f_active_cap, 1e-12)
    ht_from_thr_cap = A_throat_geo_total / max(f_active_cap, 1e-12)
    ht_nozzle_cap   = max(ht_from_out_cap, ht_from_thr_cap)
    ht_rotor_cap    = ht_nozzle_cap + 0.0003

    # Implied circumferential widths (totals) for reporting
    b_nozzles_throat_cap = A_throat_geo_total / max(ht_nozzle_cap, 1e-12)
    b_nozzles_out_cap    = A_outlet_geo_total  / max(ht_nozzle_cap, 1e-12)

    # Per-nozzle widths
    b_nozzle_throat_cap = b_nozzles_throat_cap / max(no_nozzles_cap,1)
    b_nozzle_out_cap    = b_nozzles_out_cap    / max(no_nozzles_cap,1)

    if USE_CAPACITY_NOZZLE:
        # Replace the runtime values used downstream
        eps        = eps_cap
        ht_nozzle  = ht_nozzle_cap
        ht_rotor   = ht_rotor_cap

        # Keep your original "totals" variables but add CAP variants for clarity in res{}
        b_nozzles_throat = b_nozzles_throat_cap
        b_nozzles_out    = b_nozzles_out_cap

        # And per-nozzle splits used later for reporting
        b_nozzle_throat  = b_nozzle_throat_cap
        b_nozzle_out     = b_nozzle_out_cap

        # If you reference number of nozzles later, align it too
        no_nozzles = no_nozzles_cap

    c1 = np.sqrt(c1a**2+c1u**2) # m/s; absolute velocity stator outlet
    w1a = c1a # m/s; axial relative velocity stator outlet
    w1u = c1u - U # m/s; tangential relative velocity stator outlet
    w1 = np.sqrt(w1a**2+w1u**2) # m/s; relative velocity stator outlet
    Ma1r = w1/a_nozzle_out # [-]; relative Mach number stator outlet

    beta1 = beta_rotor # deg; rotor inlet flow angle                          
    beta2 = 180-beta_rotor_2 # deg; rotor outlet flow angle (impulse turbine)

    theta = beta2 - beta1 # deg; rotor total deflection angle
    
    ht_rotor = ht_nozzle + 0.0003 # m; rotor height
    P_guess = m_dot*dh_stage*eta_guess # W; guess for power output
    chord = chord_opt # m; rotor chord for optimization

    #legacy MM chord script
    #chords = [0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15] #list of chord values
    #chord_guess = 1.6*np.sqrt((P_guess/m_dot/1000+25))/1000 # m; guess for rotor chord
    #chord = min(chords, key=lambda x:abs(x-chord_guess)) #pick the closest chord value from the list
    #chord = 0.0125 # m; rotor chord for GA fixed

    phi_it = 0.957 - 0.000362 * theta - 0.0258 * Ma1r + 0.00000639 * theta**2 + 0.0674 * Ma1r**2 - 0.0000000753 * theta**3 - 0.043 * Ma1r**3 - 0.000238 * theta * Ma1r + 0.00000145 * theta**2 * Ma1r + 0.0000425 * theta * Ma1r**2 # [-]; impulse turbine loss factor, correlation
    def get_phi(phi_it,ht_rotor,chord):
        #need phi_it loop here to get w2 
        current_phi = phi_it
        k1 = 2 # [-]; phi_it_coeff m
        k2 = 0.65 # [-]; phi_it_coeff n
        phi_corrected = np.sqrt(1-((1-current_phi**2)/k1)*(1+(k1-1)*(ht_rotor/chord)**(-k2))) # [-]; corrected impulse turbine loss factor
        phi_diff = abs(phi_corrected-current_phi) # [-]; difference between corrected and initial impulse turbine loss factor
        return phi_corrected

    phi_rotor = np.real(get_phi(phi_it,ht_rotor,chord)) # [-]; corrected impulse turbine loss factor

    sigma_r = sigma # solidity
    pitch_r = chord/sigma_r # m,
    no_blades = round(np.pi*D_mid/pitch_r) # [-]; number of rotor blades
    circumference_r = np.pi*D_mid # m
    Dtip = D_mid + ht_rotor # m; tip diameter
    Dhub = D_mid - ht_rotor # m; hub diameter
    f_active = max(eps * np.pi * D_mid, 1e-9) # m

    #Sector (filling/emptying) loss
    #    Glassman shows it as a velocity (momentum) reduction with a coefficient Ks that
    #    depends on the ratio blade pitch / active arc length (eqs. 8-24, 8-27, 8-31).
    #    A practical engineering form is: Ks = 1 - (pitch / f_active), clipped to [0,1].
    Ks = 1.0 - (pitch_r / f_active)
    Ks = np.clip(Ks, 0.0, 1.0)

    w2 = -w1 * phi_rotor * Ks # m/s; relative velocity rotor outlet
    w2a = w2*(-np.sin(np.radians(float(beta2)))) # m/s; axial relative velocity rotor outlet
    w2u = w2*(-np.cos(np.radians(float(beta2)))) # m/s; tangential relative velocity rotor outlet
    #print("w2a:", w2a, "w2u:", w2u)
    if w2.imag != 0:
        print("Warning: complex number for w2")
    c2a = w2a # m/s; axial absolute velocity rotor outlet
    c2u = w2u + U # m/s; tangential absolute velocity rotor outlet
    c2 = np.sqrt(c2a**2+c2u**2) # m/s; absolute velocity rotor outlet
    #print("c2a:", c2a, "c2u:", c2u)
    #if c2 is a complex number, print a warning
    if c2.imag != 0:
        print("Warning: complex number for c2")
    alpha_rotor = np.degrees(np.arctan(float(c2a) / float(c2u))) # deg; rotor outlet absolute angle
    dh_rotor = (w1**2-w2**2)/2 # J/kg; enthalpy drop rotor
    h2 = h_act[-1] + dh_rotor # J/kg; enthalpy rotor outlet
    r = RP.REFPROPdll(fluid,"HP","T;D;W;S",baseSI,iMass,iFlag,h2,p_outlet,z); check_err(r)
    T2,rho2,a2,s2 = r.Output[0:4] # K, kg/m3, m/s, J/kgK
    Ma2 = c2/a2 # [-]; Mach number rotor outlet
    Ma2r = w2/a2 # [-]; relative Mach number rotor outlet
    h2t = h2 + (c2**2)/2 # J/kg; total enthalpy rotor outlet
    #check if the state is superheated vapor
    try:
        q_tot_out = RP.REFPROPdll(fluid,"HS","QMASS",baseSI,iMass,iFlag,h2t,s2,z); check_err(q_tot_out)
        #print(f"q_tot_out: {q_tot_out.Output[0]}")
        if q_tot_out.Output[0] < 0:
            r = RP.REFPROPdll(fluid,"HS","T;D;W;P",baseSI,iMass,iFlag,h2t,s2 ,z); check_err(r)
            T2t,rho2t,a2t,p_outlet_total = r.Output[0:4] # K, kg/m3, m/s, J/kgK
    except:
        T2t,rho2t,a2t,p_outlet_total = T2,rho2,a2,p_outlet # K, kg/m3, m/s, J/kgK
    #print(f"dh_rotor: {dh_rotor}, dh_stage: {dh_stage}, DOR: {DOR}")

    #lopatky
    
    chord_s_ax = 1.25 * chord
    sigma_s = 1.2 * sigma # solidity
    pitch_s = chord_s_ax/sigma_s # m
    circumference_s = np.pi*D_mid # m
    no_nozzles = round(eps_nom * circumference_s/pitch_s) # [-]; number of nozzles
    b_nozzle_out = b_nozzles_out/no_nozzles # m
    b_nozzle_throat = b_nozzles_throat/no_nozzles # m

    P_turb_aero = m_dot*U*(c1u-c2u) # W; aerodynamic power
    #print (f"p_outlet_total: {p_outlet_total}")
    #friction loss correlation
    P_fric = 0.01*(n_design/60)**3*D_mid**5*rho2 # W; friction loss
    
    P_partial_admission_v1 = (1-eps)*rho2*(n_design/60)**3*D_mid**4*3.8*ht_rotor # W; partial admission and ventilation loss
    Kp_unenclosed = 3.63
    enclosure_factor = 0.4   # choose 0.25–0.5 for enclosed rotors; expose as a knob if you like
    Kp = Kp_unenclosed * enclosure_factor
    P_pump = Kp * rho2 * U**3 * ht_rotor * D_mid**4 * (1.0 - eps)
    P_partial_admission = max(P_partial_admission_v1, P_pump) # W; partial admission and ventilation loss

    P_mech = P_turb_aero - P_fric - P_partial_admission # W; mechanical power
    eta_turb = (P_mech)/(m_dot*dh_stage) # [-]; turbine efficiency
    h_out_eta = h[0] - eta_turb*dh_stage # J/kg; outlet enthalpy with efficiency
    r=RP.REFPROPdll(fluid,"HP","S",baseSI,iMass,iFlag,h_out_eta,p_outlet,z); check_err(r)
    s_out_eta = r.Output[0] # J/kgK; outlet entropy with efficiency
    DOR = dh_rotor/dh_stage # [-]; degree of reaction

    #package the results in a dictionary
    res = {"c1u":c1u,"c1a":c1a,"w1u":w1u,"w1a":w1a,"U":U,"c2u":c2u,"c2a":c2a,"w2u":w2u,"w2a":w2a,"alpha_stator":alpha_stator,
           "beta2":beta2,"alpha_rotor":alpha_rotor,"s":s,"h":h,"s_nozzle_out":s_nozzle_out,"h_nozzle_out":h_nozzle_out,"s2":s2,
           "h2":h2,"h2t":h2t,"s_out_eta":s_out_eta,"h_out_eta":h_out_eta,"p_inlet":p_inlet,"p_outlet":p_outlet,
           "p_outlet_total":p_outlet_total,"fluid":fluid,"z":z,"PR_is":PR_is,"Ma_is": Ma_is,"Ma_is_out": Ma_is[-1],"p_is":p_is,"throat_index":throat_index,
           "PR_act":PR_act,"Ma_act":Ma_act,"eta_turb":eta_turb,"P_turb_aero":P_turb_aero,"P_mech":P_mech,"U_over_c_is":U_over_c_is,"Ma1r":Ma1r, "u_over_c":U/c_nozzle_out,
           "dh_stage":dh_stage,"phi_rotor":phi_rotor,"phi_stator":phi_stator,"P_fric":P_fric,"P_partial_admission":P_partial_admission, "A_ratio": A_ratio, "b_nozzle_out":b_nozzle_out,
           "b_nozzle_throat":b_nozzle_throat, "no_nozzles":no_nozzles, "chord":chord, "chord_s_ax": chord_s_ax, "ht_nozzle": ht_nozzle, "rho_nozzle_out": rho_nozzle_out, "Ma_nozzle_out": Ma_nozzle_out, "c_nozzle_out": c_nozzle_out,
           "p_throat":p_is[throat_index], "T_throat":T_is[throat_index], "Ma_throat":Ma_is[throat_index], "T_outlet":T2, "ht_rotor": ht_rotor, "P_pump": P_pump, "Ks": Ks, "pitch_r": pitch_r, "f_active": f_active, "A_throat": A_throat, "A_outlet": A_outlet,
           "eps_eff": eps_cap if USE_CAPACITY_NOZZLE else eps,
            "no_nozzles_cap": no_nozzles_cap if USE_CAPACITY_NOZZLE else no_nozzles,
            "ht_nozzle_cap": ht_nozzle_cap if USE_CAPACITY_NOZZLE else ht_nozzle,
            "ht_rotor_cap":  ht_rotor_cap  if USE_CAPACITY_NOZZLE else ht_rotor,
            "b_nozzles_throat_cap": b_nozzles_throat_cap if USE_CAPACITY_NOZZLE else b_nozzles_throat,
            "b_nozzles_out_cap":    b_nozzles_out_cap    if USE_CAPACITY_NOZZLE else b_nozzles_out,
            "b_nozzle_throat_cap":  b_nozzle_throat_cap  if USE_CAPACITY_NOZZLE else b_nozzle_throat,
            "b_nozzle_out_cap":     b_nozzle_out_cap     if USE_CAPACITY_NOZZLE else b_nozzle_out,
            "A_outlet_geo_total":   A_outlet_geo_total,
            "A_throat_geo_total":   A_throat_geo_total,
            "A_ratio_id": A_ratio_id, 
            "no_blades": no_blades,
            "circumference_r": circumference_r,
            "Dtip": Dtip,
            "Dhub": Dhub,
            "w1": w1,
    }

    return res

def draw_velocity_triangles(c1u,c1a,w1u,w1a,U,c2u,c2a,w2u,w2a,alpha_stator,beta2,alpha_rotor):
    plt.figure()
    plt.arrow(0,0,c1u,c1a, head_width=5, head_length=5, fc='red', ec='red', length_includes_head=True)
    plt.arrow(0,0,w1u,w1a, head_width=5, head_length=5, fc='blue', ec='blue', length_includes_head=True)
    plt.arrow(w1u,w1a,U,0, head_width=5, head_length=5, fc='green', ec='green', length_includes_head=True)
    plt.arrow(0,0,c2u,c2a, head_width=5, head_length=5, fc='red', ec='red', length_includes_head=True)
    plt.arrow(0,0,w2u,w2a, head_width=5, head_length=5, fc='blue', ec='blue', length_includes_head=True)
    plt.arrow(w2u,w2a,U,0, head_width=5, head_length=5, fc='green', ec='green', length_includes_head=True)
    plt.ylabel(r'Axial velocity [m.s$^{-1}$]')
    plt.xlabel(r'Tangential velocity [m.s$^{-1}$]')
    plt.xlim(-150,300)
    plt.ylim(0,100)
    #move x axis to y=0
    plt.axvline(0, color='black',linewidth=0.5)
    #invert y axis
    plt.gca().invert_yaxis()
    #show grid
    plt.grid(True, linestyle='--')
    #equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('velocity_triangles.png', dpi = 1000)
    plt.show()
    plt.close()

def draw_expansion_line(s,h,s_nozzle_out,h_nozzle_out,s2,h2,h2t,s_out_eta,h_out_eta,p_inlet,p_outlet,p_outlet_total,fluid,z):
    #initialize REFPROP
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX']) # Instantiate the REFPROP function library
    RP.SETPATHdll(os.environ['RPPREFIX']) # Set the path to the folder containing the REFPROP shared library
    RP.SETFLUIDSdll(fluid)  # Set the fluids
    baseSI = RP.GETENUMdll(0, "MASSBASESI").iEnum # Get the base SI units
    iMass = 0; iFlag = 0 # Set the mass and flag to 0

    def check_err(r): # Function to check the error code from REFPROP
        if r.ierr > 0:
            raise ValueError(r.herr)
    s_range = np.linspace(s[0]-20,s_nozzle_out+30,1000)
    isobar_inlet = []
    isobar_outlet = []
    isobar_rotor_out_total = []
    for ss in s_range:
        r = RP.REFPROPdll(fluid,"PS","H",baseSI,iMass,iFlag,p_inlet,ss,z); check_err(r)
        hh = r.Output[0] # J/kg
        isobar_inlet.append(hh)
        r = RP.REFPROPdll(fluid,"PS","H",baseSI,iMass,iFlag,p_outlet,ss,z); check_err(r)
        hh = r.Output[0] # J/kg
        isobar_outlet.append(hh)
        r = RP.REFPROPdll(fluid,"PS","H",baseSI,iMass,iFlag,p_outlet_total,ss,z); check_err(r)
        hh = r.Output[0] # J/kg
        isobar_rotor_out_total.append(hh)

    fig, ax = plt.subplots()
    ax.plot(s_range,isobar_inlet, label='Isobar inlet', color='black', linestyle='--', linewidth = 0.7)
    ax.plot(s_range,isobar_outlet, label='Isobar outlet', color='black', linestyle='--', linewidth = 0.7)
    ax.plot(s_range,isobar_rotor_out_total, label='Isobar rotor outlet total', color='black', linestyle='--', linewidth = 0.7)
    ax.scatter(s[0], h[0], color='red', marker='x')
    ax.scatter(s_nozzle_out, h_nozzle_out, color='red', marker='x')
    #connect s0 and snozzle out with a dashed line
    ax.plot([s[0], s_nozzle_out], [h[0], h_nozzle_out],linewidth=1,color='black')
    ax.scatter(s[1], h[1], color='red', marker='x')
    ax.scatter(s2, h2t, color='red', marker='x')
    ax.plot([s_nozzle_out, s2], [h_nozzle_out, h2],linewidth=1,color='black')
    ax.scatter(s2, h2, color='red', marker='x')
    ax.plot([s2,s2], [h2,h2t],linewidth=1,color='black')
    ax.scatter(s_out_eta,h_out_eta, color='red', marker='x')
    ax.plot([s[0],s_out_eta], [h[0],h_out_eta],"k--",linewidth=1,color='black')
    #connect s0 and s1 with a dashed line
    ax.plot([s[0], s[1]], [h[0], h[1]], 'k--',linewidth=0.5,color='black')
    ax.grid(True, linestyle='--')
    ax.set(xlabel=r'Specific entropy $s$ [J.(kg.K)$^{-1}$]', ylabel=r'Specific enthalpy $h$ [J.kg$^{-1}$]')
    #change y axis to scientific notation
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #add text to the isobars
    ax.text(s[0]-10, h[0]-2000, r'$p_{evap}$', ha='right')
    ax.text(s_nozzle_out-10, h_nozzle_out-9000, r'$p_{cond}$', ha='right')
    ax.text(s_nozzle_out-10, h_nozzle_out-2000, r'$p_{cond,tot}$', ha='right')
    #add text to the state points
    ax.text(s[0]-1, h[0]+500, '1', ha='right')
    ax.text(s_nozzle_out-1, h_nozzle_out-1000, '2', ha='right')
    ax.text(s[1]-1, h[1]+500, r'2$_{is}$=3$_{is}$', ha='right')
    ax.text(s2-1, h2-3500, '3', ha='right')
    ax.text(s2-1, h2t, r'3$_{t}$', ha='right')
    ax.text(s_out_eta+1, h_out_eta-2000, r'2$_{\eta}$=3$_{\eta}$', ha='left')
    plt.savefig('expansion_line_h_s.png', dpi = 600)
    plt.show()
    plt.close()

def _safe_gamma_sT(s_val, T_val, fluid_name="R1233ZDE", p_cap=1e8):
    """
    Returns Gamma at (s,T) or NaN if invalid/out-of-range/two-phase/extreme P.
    Uses REFPROP backend through CoolProp.
    """
    AS = CP.AbstractState("REFPROP", fluid_name)
    AS.set_mole_fractions([1.0])
    try:
        AS.update(CP.SmassT_INPUTS, float(s_val), float(T_val))
    except Exception:
        return np.nan
    # Skip two-phase points
    try:
        q = AS.Q()
        if 0.0 <= q <= 1.0:
            return np.nan
    except Exception:
        pass
    # Bound very high pressures to avoid out-of-range
    try:
        if AS.p() > p_cap:
            return np.nan
    except Exception:
        pass
    try:
        g = AS.fundamental_derivative_of_gas_dynamics()
    except Exception:
        return np.nan
    # Optional: hide Gamma>1 region as in your original code
    if g > 1.0:
        return np.nan
    return g


def plot_gamma_Ts(fluid_name="R1233ZDE",
                  T_min=300.0, T_max=450.0,
                  s_min=900.0,  s_max=2000.0,
                  steps=600, outfile="Gamma_T_s_diagram.png"):
    """
    Robust Γ(s,T) colormap on a T–s plane, with saturation lines.
    """
    # Make sure CP can find REFPROP
    CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, os.getenv('RPPREFIX'))

    T_vals = np.linspace(T_min, T_max, steps)
    s_vals = np.linspace(s_min, s_max, steps)
    TT, SS = np.meshgrid(T_vals, s_vals)

    Gamma = np.full_like(TT, np.nan, dtype=float)
    Edge  = np.zeros_like(TT, dtype=float)

    # Fill Gamma safely
    for i in range(TT.shape[0]):
        for j in range(TT.shape[1]):
            Gamma[i, j] = _safe_gamma_sT(SS[i, j], TT[i, j], fluid_name=fluid_name)

    # Simple edge outline
    for i in range(1, Gamma.shape[0]):
        for j in range(Gamma.shape[1]):
            if (np.isnan(Gamma[i, j]) and not np.isnan(Gamma[i-1, j])) or \
               (not np.isnan(Gamma[i, j]) and np.isnan(Gamma[i-1, j])):
                Edge[i, j] = 1.0

    # Saturation curves
    s_vaps, s_liqs, T_crits = [], [], []
    for Ts in T_vals:
        try:
            AS = CP.AbstractState("REFPROP", fluid_name); AS.set_mole_fractions([1.0])
            AS.update(CP.QT_INPUTS, 1, Ts); s_vaps.append(AS.smass()); T_crits.append(Ts)
            AS.update(CP.QT_INPUTS, 0, Ts); s_liqs.append(AS.smass())
        except Exception:
            # skip Ts outside validity
            continue

    plt.figure()
    cmap = plt.contourf(SS, TT, Gamma, 15, cmap='viridis')
    plt.contour(SS, TT, Edge, 1, colors='black', linewidths=0.2)
    if s_vaps and s_liqs:
        plt.plot(s_vaps, T_crits, 'k', linewidth=0.75)
        plt.plot(s_liqs, T_crits, 'k', linewidth=0.75)
    plt.xlabel(r'Specific entropy [J/(kg·K)]')
    plt.ylabel(r'Temperature [K]')
    cbar = plt.colorbar(cmap); cbar.set_label('Gamma [-]')
    plt.xlim(s_min, s_max); plt.ylim(T_min, T_max)
    plt.savefig(outfile, dpi=600)
    plt.show()
    plt.close()


def plot_gamma_Pv(fluid_name="R1233ZDE",
                  P_min=1e4, P_max=4e6,
                  v_min=1.5e-3, v_max=5e-1,
                  steps=600, outfile="Gamma_P_v_diagram.png"):
    """
    Robust Γ(p,v) colormap on a p–v plane, with saturation lines drawn safely.
    """
    CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, os.getenv('RPPREFIX'))

    P_vals = np.linspace(P_min, P_max, steps)
    v_vals = np.linspace(v_min, v_max, steps)
    PP, VV = np.meshgrid(P_vals, v_vals)

    # Compute Gamma safely
    Gamma = np.full_like(PP, np.nan, dtype=float)
    for i in range(PP.shape[0]):
        for j in range(PP.shape[1]):
            AS = CP.AbstractState("REFPROP", fluid_name)
            AS.set_mole_fractions([1.0])
            try:
                AS.update(CP.DmassP_INPUTS, 1.0/float(VV[i, j]), float(PP[i, j]))
            except Exception:
                continue
            # Skip two-phase
            try:
                q = AS.Q()
                if 0.0 <= q <= 1.0:
                    continue
            except Exception:
                pass
            try:
                g = AS.fundamental_derivative_of_gas_dynamics()
            except Exception:
                continue
            if g > 1.0:   # keep your original mask
                continue
            Gamma[i, j] = g

    # Build saturation curves independently to avoid length mismatches
    vap_P, vap_v = [], []
    liq_P, liq_v = [], []
    for Ps in P_vals:
        AS = CP.AbstractState("REFPROP", fluid_name)
        AS.set_mole_fractions([1.0])
        # saturated vapor (Q=1)
        try:
            AS.update(CP.PQ_INPUTS, Ps, 1)
            vap_P.append(Ps)
            vap_v.append(1.0/AS.rhomass())
        except Exception:
            pass
        # saturated liquid (Q=0)
        try:
            AS.update(CP.PQ_INPUTS, Ps, 0)
            liq_P.append(Ps)
            liq_v.append(1.0/AS.rhomass())
        except Exception:
            pass

    # Plot
    fig, ax = plt.subplots()
    cmap = ax.contourf(VV, PP, Gamma, 15, cmap='viridis')
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label('Gamma [-]')

    # Plot saturation lines only if we have matching x/y lengths
    if len(vap_v) == len(vap_P) and len(vap_v) > 1:
        ax.plot(vap_v, vap_P, 'k', linewidth=0.75)
    if len(liq_v) == len(liq_P) and len(liq_v) > 1:
        ax.plot(liq_v, liq_P, 'k', linewidth=0.75)

    ax.set_xlabel(r'Specific Volume (m$^{3}$/kg)')
    ax.set_ylabel('Pressure [Pa]')
    ax.set_xscale('log')
    ax.set_ylim(P_min, P_max)
    ax.set_xlim(v_min, v_max)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(outfile, dpi=600)
    plt.show()
    plt.close()

def draw_nozzle_expansion(PR_is,Ma_is,p_is,throat_index,PR_act,Ma_act):
    #draw isentropic expansion
    fig, ax = plt.subplots()
    ax.plot(PR_is,Ma_is, label='Isentropic expansion', color='black')
    ax.set(xlabel=r'Pressure ratio $\Pi$ [-]', ylabel=r'Mach number $Ma$ [-]')
    ax.scatter(p_is[throat_index]/p_outlet,1, color='red', marker='x')
    #add text to the throat
    ax.text(p_is[throat_index]/p_outlet+0.05, 1.05, 'Throat', ha='right')
    ax.plot(PR_act,Ma_act, label='Real expansion with losses', color='black', linestyle='--')
    #add scatter to the nozzle outlet for both and add text with the Ma value, use x markers in red
    ax.scatter(p_is[-1]/p_outlet,Ma_act[-1], color='red', marker='x')
    ax.text(p_is[-1]/p_outlet, Ma_act[-1], r'$Ma_2$ = '+str(round(Ma_act[-1],3)), ha='right')
    ax.scatter(p_is[-1]/p_outlet,Ma_is[-1], color='red', marker='x')
    ax.text(p_is[-1]/p_outlet, Ma_is[-1], r'$Ma_{2,is}$ = '+str(round(Ma_is[-1],3)), ha='right')
    #add a tick at x axis for the inlet, throat and for the outlet of the nozzle
    ax.axvline(p_is[0]/p_outlet, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(p_is[throat_index]/p_outlet, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(p_is[-1]/p_outlet, color='black', linestyle='--', linewidth=0.5)
    #add labels to the ticks, rotate the text by 90°, offset the text by 0.5 in x axis
    ax.text(p_is[0]/p_outlet+0.05, 0.5, 'Inlet', ha='right', rotation=90)
    ax.text(p_is[throat_index]/p_outlet+0.05, 0.5, 'Throat', ha='right', rotation=90)
    ax.text(p_is[-1]/p_outlet+0.05, 0.5, 'Outlet', ha='right', rotation=90)
    #write the x axis value next to the tick text, rotated by 90°
    ax.text(p_is[0]/p_outlet-0.05, 0.5, r"$\Pi$ = "+str(round(p_is[0]/p_outlet,2)), ha='left', rotation=90)
    ax.text(p_is[throat_index]/p_outlet-0.05, 0.5, r"$\Pi$ = "+str(round(p_is[throat_index]/p_outlet,2)), ha='left', rotation=90)
    ax.text(p_is[-1]/p_outlet-0.05, 0.5, r"$\Pi$= "+str(round(p_is[-1]/p_outlet,2)), ha='left', rotation=90)
    ax.legend()
    ax.grid(True, linestyle='--')
    #invert x axis
    ax.invert_xaxis()
    plt.savefig('isentropic_expansion_nozzle_losses.png', dpi = 600)
    plt.show()
    plt.close()

def legacy_sensitivity_analysis_rpm(n_design,D_mid,alpha_stator,eta_guess,p_inlet,T_inlet):
    #vary rotational speed in the range of 50 to 150% of nominal speed and calculate the eta_turb, plot it over U_over_c_is - legacy code
    
    n_range = np.linspace(0.5*n_design,1.5*n_design,100)
    eta_turb_range = []
    U_over_c_is_range = []
    for n in n_range:
        res=meanline_design(D_mid,n,alpha_stator,eta_guess,p_inlet,T_inlet)
        eta_turb_range.append(res["eta_turb"])
        U_over_c_is_range.append(res["U_over_c_is"])
    #find maximum eta_turb and corresponding U_over_c_is and rpm
    max_eta_turb = max(eta_turb_range)
    max_eta_turb_index = eta_turb_range.index(max_eta_turb)
    max_eta_turb_rpm = n_range[max_eta_turb_index]
    print(f"Maximum efficiency: {max_eta_turb} at U/c_is: {U_over_c_is_range[max_eta_turb_index]} and rpm: {max_eta_turb_rpm}")
    fig, ax = plt.subplots()
    ax.plot(U_over_c_is_range,eta_turb_range, label='Turbine efficiency', color='black')
    ax.set(xlabel='U/c_is [-]', ylabel='Turbine efficiency [-]')
    #ax.legend()
    ax.grid(True, linestyle='--')
    plt.savefig('sensitivity_analysis_rpm.png', dpi = 600)
    plt.show()
    plt.close()

def sensitivity_partial_admission_vs_height(
        D_mid, n_design, alpha_stator, eta_guess,
        p_inlet, T_inlet, beta_rotor, m_dot, sigma, beta_rotor_2, chord,
        e_min=0.30, e_max=1.00, n_points=16, save_prefix="sens_e_height"):
    """
    Sweep partial-admission fraction e and show how blade height (nozzle & rotor),
    efficiency, and PA losses vary.

    Parameters
    ----------
    e_min, e_max : float
        Range of partial admission fraction to sweep.
    n_points : int
        Number of points in the sweep.
    save_prefix : str
        Prefix for saved figures.
    """
    e_vals   = np.linspace(e_min, e_max, n_points)
    ht_n     = []
    ht_r     = []
    eta_list = []
    Pmech    = []
    Paero    = []
    Ppa      = []
    Ppump    = []
    Pfric    = []
    Ks_list  = []

    for e in e_vals:
        res = meanline_design(D_mid, n_design, alpha_stator,
                              eta_guess, p_inlet, T_inlet, beta_rotor,
                              m_dot, float(e), sigma, beta_rotor_2, chord)
        ht_n.append(res["ht_nozzle"])
        ht_r.append(res["ht_rotor"])
        eta_list.append(res["eta_turb"])
        Pmech.append(res["P_mech"])
        Paero.append(res["P_turb_aero"])
        Ppa.append(res["P_partial_admission"])
        Pfric.append(res["P_fric"])
        # P_pump is included if you added it to res in step 1; otherwise fill with NaN
        Ppump.append(res.get("P_pump", np.nan))
        Ks_list.append(res.get("Ks", np.nan))

    # --- Table to terminal ---
    try:
        from tabulate import tabulate
        rows = []
        for i, e in enumerate(e_vals):
            rows.append([
                f"{e:.3f}",
                ht_n[i], ht_r[i],
                eta_list[i],
                Paero[i]/1e3, Ppa[i]/1e3, Ppump[i]/1e3 if np.isfinite(Ppump[i]) else np.nan,
                Pfric[i]/1e3, Pmech[i]/1e3,
                Ks_list[i]
            ])
        print("\nPartial admission sensitivity (height & performance):")
        print(tabulate(
            rows,
            headers=[
                "e", "h_noz [m]", "h_rot [m]", "eta [-]",
                "P_aero [kW]", "P_PA [kW]", "P_pump [kW]",
                "P_fric [kW]", "P_mech [kW]", "K_s [-]"
            ],
            floatfmt=(".3f",".5f",".5f",".4f",".2f",".2f",".2f",".2f",".2f",".3f"),
            tablefmt="github"
        ))
    except Exception:
        # Fallback simple print if tabulate unavailable for some reason
        print("e, h_noz[m], h_rot[m], eta[-], P_aero[kW], P_PA[kW], P_pump[kW], P_fric[kW], P_mech[kW], Ks[-]")
        for i, e in enumerate(e_vals):
            print(e, ht_n[i], ht_r[i], eta_list[i], Paero[i]/1e3, Ppa[i]/1e3,
                  (Ppump[i]/1e3 if np.isfinite(Ppump[i]) else np.nan), Pfric[i]/1e3, Pmech[i]/1e3, Ks_list[i])

    # --- Plots ---
    import matplotlib.pyplot as plt
    # (A) Blade height vs e
    fig, ax1 = plt.subplots()
    ax1.plot(e_vals, ht_n, label="Nozzle height", lw=2)
    ax1.plot(e_vals, ht_r, label="Rotor height", lw=2)
    ax1.set_xlabel("Partial admission fraction e [-]")
    ax1.set_ylabel("Height [m]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_height_vs_e.png", dpi=300)

    # (B) Efficiency vs e (with heights on a twin axis if useful)
    fig, ax = plt.subplots()
    ax.plot(e_vals, eta_list, lw=2)
    ax.set_xlabel("Partial admission fraction e [-]")
    ax.set_ylabel("Turbine efficiency [-]")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_eta_vs_e.png", dpi=300)

    # (C) Power components vs e
    fig, ax = plt.subplots()
    ax.plot(e_vals, np.array(Paero)/1e3, label="P_aero", lw=2)
    ax.plot(e_vals, np.array(Ppa)/1e3,   label="P_PA (total)", lw=2)
    if np.isfinite(Ppump).any():
        ax.plot(e_vals, np.array(Ppump)/1e3, label="P_pump", lw=2, ls="--")
    ax.plot(e_vals, np.array(Pfric)/1e3, label="P_fric", lw=2, ls="-.")
    ax.plot(e_vals, np.array(Pmech)/1e3, label="P_mech", lw=2)
    ax.set_xlabel("Partial admission fraction e [-]")
    ax.set_ylabel("Power [kW]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_power_vs_e.png", dpi=300)

    # (D) Ks vs e (optional diagnostic)
    if np.isfinite(Ks_list).any():
        fig, ax = plt.subplots()
        ax.plot(e_vals, Ks_list, lw=2)
        ax.set_xlabel("Partial admission fraction e [-]")
        ax.set_ylabel("Sector coefficient K_s [-]")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(f"{save_prefix}_Ks_vs_e.png", dpi=300)

    plt.show()
    plt.close("all")


def calculate_efficiency_for_n(args):
    n, D_mid, alpha_stator, eta_guess, p_inlet, T_inlet, m_dot = args
    res = meanline_design(D_mid, n, alpha_stator, eta_guess, p_inlet, T_inlet, beta_rotor, m_dot)
    return res["eta_turb"], res["P_turb_aero"]

def sensitivity_analysis_rpm_pool(n_design, D_mid, alpha_stator, eta_guess, p_inlet, T_inlet):
    p_inlet_range = np.linspace(100000, 700000, 100)  # Pa
    n_range = np.linspace(3000, 20000, 100)           # rpm

    eta_turb_range = []
    P_turb_range = []

    dT_SH = 10  # K; superheating temperature

    # Initialize REFPROP (done outside the loop to avoid reinitialization in each process)
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
    RP.SETPATHdll(os.environ['RPPREFIX'])
    RP.SETFLUIDSdll(fluid)
    baseSI = RP.GETENUMdll(0, "MASSBASESI").iEnum
    iMass = 0
    iFlag = 0

    n_cpus = max(1, os.cpu_count() - 4)

    def check_err(r):
        if r.ierr > 0:
            raise ValueError(r.herr)

    for p_inlet in p_inlet_range:
        # Convert p_inlet from Pa to kPa
        p_inlet_kPa = p_inlet / 1000  # Convert Pa to kPa

        # Calculate V̇ in l/min using the given relationship
        V_dot_lpm = (p_inlet_kPa - 14.48) / 29.136  # l/min

        # Check for negative or zero volumetric flow rate
        if V_dot_lpm <= 0:
            continue  # Skip this iteration if V̇ is not physically meaningful

        # Convert V̇ from l/min to m³/s
        V_dot_m3s = V_dot_lpm * (1e-3) / 60  # m³/s

        # Calculate the inlet temperature based on the saturation temperature plus superheat
        r = RP.REFPROPdll(fluid, "PQ", "T", baseSI, iMass, iFlag, p_inlet, 1, z)
        check_err(r)
        T_inlet = r.Output[0] + dT_SH  # K; inlet temperature

        # Calculate the density at the new inlet conditions
        r = RP.REFPROPdll(fluid, "PT", "D", baseSI, iMass, iFlag, p_inlet, 335, z)
        check_err(r)
        rho_inlet = r.Output[0]  # kg/m³

        # Update the mass flow rate based on the new density
        m_dot = V_dot_m3s * rho_inlet  # kg/s

        # Prepare arguments for parallel processing
        args = [(n, D_mid, alpha_stator, eta_guess, p_inlet, T_inlet, m_dot) for n in n_range]

        # Parallel computation
        with Pool(processes=n_cpus) as pool:
            results = pool.map(calculate_efficiency_for_n, args)

        eta_turb_for_p, P_turb_for_p = zip(*results)
        eta_turb_range.append(eta_turb_for_p)
        P_turb_range.append(P_turb_for_p)

    """
    # Plotting the results
    fig, ax = plt.subplots()
    for i, p_inlet in enumerate(p_inlet_range):
        ax.plot(n_range, eta_turb_range[i], label=f'p_inlet = {p_inlet:.0f} Pa')

    ax.set(xlabel='Rotational speed [rpm]', ylabel='Turbine efficiency [-]')
    ax.grid(True, linestyle='--')
    plt.savefig('sensitivity_analysis_rpm.png', dpi=600)
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    for i, p_inlet in enumerate(p_inlet_range):
        ax.plot(n_range, P_turb_range[i], label=f'p_inlet = {p_inlet:.0f} Pa')
    
    ax.set(xlabel='Rotational speed [rpm]', ylabel='Turbine power [W]')
    ax.grid(True, linestyle='--')
    plt.savefig('sensitivity_analysis_rpm_power.png', dpi=600)
    plt.show()
    plt.close()

    # Find maxima for each P_turb_range and plot it over n
    max_P_turb = [max(P_turb) for P_turb in P_turb_range]
    max_P_turb_index = [P_turb.index(max_P) for P_turb, max_P in zip(P_turb_range, max_P_turb)]
    max_P_turb_rpm = [n_range[index] for index in max_P_turb_index]
    
    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Plot the maximum rotational speed corresponding to the maximum turbine power
    ax.plot(p_inlet_range / p_outlet, max_P_turb_rpm, 'bo', label='Maximum Power Points')

    # Perform polynomial fitting
    p = np.polyfit(p_inlet_range / p_outlet, max_P_turb_rpm, 4)

    # Plot the polynomial fit
    ax.plot(p_inlet_range / p_outlet, np.polyval(p, p_inlet_range / p_outlet), 'r--', label='Polynomial Fit')

    # Set labels, grid, and legend
    ax.set(xlabel=r'Pressure ratio $i$ [-]', ylabel='Rotational speed $n$ [rpm]')
    ax.grid(True, linestyle='--')
    ax.legend()

    # Save and display the figure
    plt.savefig('sensitivity_analysis_rpm_power_max.png', dpi=600)
    plt.show()
    plt.close()

    # Write the polyfit coefficients to the terminal
    print(f"Polyfit coefficients: {p}")
    """
    #create a heatmap of the turbine power over the rotational speed (x) and the pressure ratio (y), use matplotlib
    fig, ax = plt.subplots()
    n_range, p_inlet_range = np.meshgrid(n_range, p_inlet_range)
    P_turb_range = np.array(P_turb_range)
    c = ax.pcolormesh(n_range, p_inlet_range/p_outlet, P_turb_range, cmap='viridis')
    #plot iso-power lines
    iso_power = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
    for power in iso_power:
        ax.contour(n_range, p_inlet_range/p_outlet, P_turb_range, levels=[power], colors='black')
    fig.colorbar(c, ax=ax, label=r'Turbine power $P_{mech}$ [W]')
    ax.set(xlabel=r'Rotational speed $n$ [rpm]', ylabel=r'Pressure ratio $\Pi$ [1]')
    plt.savefig('sensitivity_analysis_rpm_power_heatmap.png', dpi=600)
    plt.show()
    plt.close()

    #create a heatmap of the turbine efficiency over the rotational speed (x) and the pressure ratio (y), use matplotlib, show iso-efficiency lines 0.8,0.775,0.75,0.725,0.7,0.675,0.65,0.625,0.6
    fig, ax = plt.subplots()
    c = ax.pcolormesh(n_range, p_inlet_range/p_outlet, eta_turb_range, cmap='viridis')
    fig.colorbar(c, ax=ax, label=r'Turbine efficiency $\eta_{{is}_{t-s}}$ [1]')
    ax.set(xlabel=r'Rotational speed $n$ [rpm]', ylabel=r'Pressure ratio $\Pi$ [1]')
    #plot iso-efficiency lines
    iso_efficiency = [0.75,0.725,0.7,0.675,0.65,0.625,0.6,0.575,0.55]
    for eff in iso_efficiency:
        ax.contour(n_range, p_inlet_range/p_outlet, eta_turb_range, levels=[eff], colors='black')
    plt.savefig('sensitivity_analysis_rpm_efficiency_heatmap.png', dpi=600)
    plt.show()
    plt.close()

    #create a heatmap of the turbine torque over the rotational speed (x) and the pressure ratio (y), use matplotlib, show iso-torque lines 2,3,4,5,6,7,8,9,10,11,12 Nm
    fig, ax = plt.subplots()
    c = ax.pcolormesh(n_range, p_inlet_range/p_outlet, P_turb_range/(2*np.pi*n_range/60), cmap='viridis')
    fig.colorbar(c, ax=ax, label=r'Turbine torque $T_{mech}$ [Nm]')
    ax.set(xlabel=r'Rotational speed $n$ [rpm]', ylabel=r'Pressure ratio $\Pi$ [1]')
    #plot iso-torque lines
    iso_torque = [2,3,4,5,6,7,8,9,10,11,12]
    for torque in iso_torque:
        ax.contour(n_range, p_inlet_range/p_outlet, P_turb_range/(2*np.pi*n_range/60), levels=[torque], colors='black')
    plt.savefig('sensitivity_analysis_rpm_torque_heatmap.png', dpi=600)
    plt.show()
    plt.close()

def sensitivity_analysis_alpha_stator(n_design,D_mid,alpha_stator,eta_guess):
    #vary alpha_stator in the range of 10° to 20° and calculate the eta_turb, plot it over alpha_stator
    alpha_stator_range = np.linspace(10,20,100)
    eta_turb_range = []
    for alpha_stator in alpha_stator_range:
        res=meanline_design(D_mid,n_design,alpha_stator,eta_guess)
        eta_turb_range.append(res["eta_turb"])
    fig, ax = plt.subplots()
    ax.plot(alpha_stator_range,eta_turb_range, label='Turbine efficiency', color='black')
    ax.set(xlabel='Stator inlet angle [°]', ylabel='Turbine efficiency [-]')
    #ax.legend()
    ax.grid(True, linestyle='--')
    plt.savefig('sensitivity_analysis_alpha_stator.png', dpi = 600)
    plt.show()
    plt.close()

def sensitivity_analysis_D_mid(n_design,D_mid,alpha_stator,eta_guess):
    #vary alpha_stator in the range of 0.1m to 0.18m and calculate the eta_turb, plot it over D_mid
    D_mid_range = np.linspace(0.1,0.18,100)
    eta_turb_range = []
    for D_mid in D_mid_range:
        res=meanline_design(D_mid,n_design,alpha_stator,eta_guess)
        eta_turb_range.append(res["eta_turb"])
    fig, ax = plt.subplots()
    ax.plot(D_mid_range,eta_turb_range, label='Turbine efficiency', color='black')
    ax.set(xlabel='Mean diameter [m]', ylabel='Turbine efficiency [-]')
    #ax.legend()
    ax.grid(True, linestyle='--')
    plt.savefig('sensitivity_analysis_D_mid.png', dpi = 600)
    plt.show()
    plt.close()

def evaluate(individual):
    D_mid, n_design, alpha_stator, beta_rotor, beta_rotor_2, chord_opt, eps = individual
    result = meanline_design(D_mid, n_design, alpha_stator, eta_guess,p_inlet,T_inlet,beta_rotor, m_dot, eps, sigma, beta_rotor_2, chord_opt)
    return (result["eta_turb"],)

def run_GA(population, toolbox, ngen, stats):
    result, log = eaSimpleProgress(
    population, toolbox,
    cxpb=0.5,
    mutpb=0.2,
    ngen=ngen,
    stats=stats,
    halloffame=None,
    verbose=True
    )
    return result, log

def plot_convergence(log):
    gen = log.select("gen")
    max_fitness = log.select("max")
    avg_fitness = log.select("avg")

    fig, ax = plt.subplots()

    # Plot both lines on the same axes
    line1 = ax.plot(gen, max_fitness, "b-", label="Maximum Fitness")
    line2 = ax.plot(gen, avg_fitness, "r-", label="Average Fitness")
    
    # Set common labels
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")

    # Add legend
    ax.legend(loc="best")

    plt.show()

from deap import base, creator, tools, algorithms
from tqdm import tqdm

def eaSimpleProgress(population, toolbox, cxpb, mutpb, ngen, stats=None, 
                     halloffame=None, verbose=__debug__):
    """
    A variant of the deap.algorithms.eaSimple function that shows
    a tqdm progress bar for the generations in the terminal.
    
    :param population: A list of individuals.
    :param toolbox: A Toolbox that contains the evolution operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generations.
    :param stats: A Statistics object that is updated in place, optional.
    :param halloffame: A HallOfFame object that will contain the best
                       individuals, optional.
    :param verbose: Whether or not to print the statistics.
    :returns: The final population and a Logbook with the statistics of the
              evolution.
    """

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Initialize tqdm progress bar
    pbar = tqdm(total=ngen, desc="Generations", ncols=80)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Update tqdm progress bar
        pbar.update(1)

    pbar.close()

    return population, logbook

if __name__=='__main__':
    #run 1DTDT
    tic = timeit.default_timer()

    if not RUN_GA:
        best_res=meanline_design(D_mid,n_design,alpha_stator,eta_guess,p_inlet,T_inlet,beta_rotor, m_dot, e, sigma, beta_rotor_2,chord_opt)
        from tabulate import tabulate

        # helper for safe value extraction + optional scaling
        def _val(key, scale=1.0):
            v = best_res.get(key, None)
            try:
                return float(v) * scale
            except (TypeError, ValueError):
                return np.nan

        rows = [
            # --- original lines (unchanged) ---
            ["eta_turb",            best_res["eta_turb"],                 "-"],
            ["U_over_c_is",         best_res["U_over_c_is"],              "-"],
            ["P_turb_aero",         best_res["P_turb_aero"]/1e3,          "kW"],
            ["P_partial_admission", best_res["P_partial_admission"]/1e3,  "kW"],
            ["P_fric",              best_res["P_fric"]/1e3,               "kW"],
            ["P_mech",              best_res["P_mech"]/1e3,               "kW"],
            ["rho_nozzle_out",      best_res["rho_nozzle_out"],           "kg/m³"],
            ["c_nozzle_out",        best_res["c_nozzle_out"],             "m/s"],
            ["Ma_nozzle_out",       best_res["Ma_nozzle_out"],            "-"],
            ["Ma_nozzle_out_is",    best_res["Ma_is_out"], "-"],
            ["c1a",                 best_res["c1a"],                      "m/s"],
            ["c1u",                 best_res["c1u"],                      "m/s"],
            ["c2a",                 best_res["c2a"],                      "m/s"],
            ["c2u",                 best_res["c2u"],                      "m/s"],
            ["U",                   best_res["U"],                        "m/s"],
            ["p_throat",            convert(best_res["p_throat"],"Pa","kPa"), "kPa"],
            ["T_throat",            convert(best_res["T_throat"],"K","C"),    "°C"],
            ["p_outlet",            convert(best_res["p_outlet"],"Pa","kPa"), "kPa"],
            ["T_outlet",            convert(best_res["T_outlet"],"K","C"),    "°C"],
            ["ht_nozzle",           best_res["ht_nozzle"],                "m"],
            ["A_nozzle_throat",     best_res["A_throat"]*1e6,             "mm²"],
            ["A_nozzle_out",        best_res["A_outlet"]*1e6,             "mm²"],
            ["A_ratio_nozzle",      best_res["A_ratio"],                  "-"],
            ["b_nozzle_throat",     best_res["b_nozzle_throat"]*1e3,      "mm"],
            ["b_nozzle_out",        best_res["b_nozzle_out"]*1e3,         "mm"],
            ["A_Ratio_id",          best_res["A_ratio_id"],               "-"],


            # --- new capacity-aware additions (present if toggle populated res keys) ---
            ["eps_eff (cap)",               _val("eps_eff"),                        "-"],
            ["no_nozzles_cap",              _val("no_nozzles_cap"),                 "-"],
            ["ht_nozzle_cap",               _val("ht_nozzle_cap"),                  "m"],
            ["ht_rotor_cap",                _val("ht_rotor_cap"),                   "m"],
            ["A_throat_geo_total",          _val("A_throat_geo_total", 1e6),        "mm²"],
            ["A_outlet_geo_total",          _val("A_outlet_geo_total", 1e6),        "mm²"],
            ["b_nozzles_throat_cap",        _val("b_nozzles_throat_cap", 1e3),      "mm"],
            ["b_nozzles_out_cap",           _val("b_nozzles_out_cap",    1e3),      "mm"],
            ["b_nozzle_throat_cap",         _val("b_nozzle_throat_cap",  1e3),      "mm"],
            ["b_nozzle_out_cap",            _val("b_nozzle_out_cap",     1e3),      "mm"],

            #rotor
            ["no_blades",                  best_res["no_blades"],                "-"],
            ["circumference_r",            best_res["circumference_r"],          "m"],
            ["D_tip",                      best_res["Dtip"],                    "m"],
            ["D_hub",                      best_res["Dhub"],                    "m"],
            ["w1",                         best_res["w1"],                       "m/s"],    
        ]

        print("Results of meanline turbine design:")
        print("-----------------------------------")
        print(tabulate(rows, headers=["Quantity", "Value", "Unit"],
                    floatfmt=".4f", tablefmt="github"))
        """
        # --- Sensitivity: partial admission vs blade height ---
        sensitivity_partial_admission_vs_height(
            D_mid=D_mid,
            n_design=n_design,
            alpha_stator=alpha_stator,
            eta_guess=eta_guess,
            p_inlet=p_inlet,
            T_inlet=T_inlet,
            beta_rotor=beta_rotor,
            m_dot=m_dot,
            sigma=sigma,
            beta_rotor_2=beta_rotor_2,
            chord=chord_opt,
            e_min=0.30, e_max=1.00, n_points=16,
            save_prefix="sens_e_height"
        )
        """

        if PLOTTING:
             #uncomment as you wish to draw the plots for best res
            draw_nozzle_expansion(best_res["PR_is"],best_res["Ma_is"],best_res["p_is"],best_res["throat_index"],best_res["PR_act"],best_res["Ma_act"])
            draw_expansion_line(best_res["s"],best_res["h"],best_res["s_nozzle_out"],best_res["h_nozzle_out"],best_res["s2"],best_res["h2"],best_res["h2t"],best_res["s_out_eta"],best_res["h_out_eta"],best_res["p_inlet"],best_res["p_outlet"],best_res["p_outlet_total"],best_res["fluid"],best_res["z"])
            draw_velocity_triangles(best_res["c1u"],best_res["c1a"],best_res["w1u"],best_res["w1a"],best_res["U"],best_res["c2u"],best_res["c2a"],best_res["w2u"],best_res["w2a"],best_res["alpha_stator"],best_res["beta2"],best_res["alpha_rotor"])

    else:
        #Genetic Algorithm 
        import multiprocessing
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)

        # Define the optimization problem to maximize eta_turb
        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximization problem
        creator.create("Individual", list, fitness=creator.FitnessMax) # Individual class

        # Setup the parameter ranges
        parameter_bounds = [
            (0.34, 0.6),    # D_mid
            (2950, 3000),  # n_design
            (11.5, 18),         # alpha_stator (degrees)
            (18, 30),       # beta_rotor (degrees)
            (18, 30),       # beta_rotor_2 (degrees)
            (0.01, 0.05),    # chord (m)
            (0.11, 0.33)   # eps (-)
        ]

        param_labels = ['D_mid', 'n_design', 'alpha_stator', 'beta_rotor', 'beta_rotor_2', 'chord', 'eps']

        # Helper to create a random individual
        def create_individual():
            return [random.uniform(low, high) for low, high in parameter_bounds]

        # Register the genetic algorithm functions
        toolbox = base.Toolbox()

        toolbox.register("map", pool.map) # Parallel map
        toolbox.register("individual", tools.initIterate, creator.Individual, create_individual) # Individual
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Population
        toolbox.register("evaluate", evaluate) # Evaluation function
        toolbox.register("mate", tools.cxBlend, alpha=0.6) # Blend crossover
        toolbox.register("mutate", tools.mutPolynomialBounded, low=[lb for lb, ub in parameter_bounds], up=[ub for lb, ub in parameter_bounds], eta=0.5, indpb=0.1) # Polynomial mutation
        toolbox.register("select", tools.selTournament, tournsize=3) # Tournament selection

        # Define a repair (clamp) function to enforce parameter bounds
        def repair(individual, bounds):
            for i, (low, high) in enumerate(bounds):
                if individual[i] < low:
                    individual[i] = low
                elif individual[i] > high:
                    individual[i] = high
            return individual

        # Decorator to apply the repair function after an operator is executed
        def checkBounds(func):
            def wrapper(*args, **kwargs):
                offspring = func(*args, **kwargs)
                # In case the operator returns a tuple of individuals
                for child in offspring:
                    repair(child, parameter_bounds)
                return offspring
            return wrapper

        # Decorate the mating and mutation operators with the bounds-checking decorator
        toolbox.decorate("mate", checkBounds)
        toolbox.decorate("mutate", checkBounds)
        
        # Generate the initial population
        population = toolbox.population(n=40)

        # Run the genetic algorithm
        ngen = 40  # Number of generations
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        print("Population size:", len(population))
        print("Example individual:", population[0])
        print("Crossover probability (cxpb):", 0.6)
        print("Mutation probability (mutpb):", 0.25)
        print("Number of generations (ngen):", ngen)

        result, log = run_GA(population, toolbox, ngen, stats)

        # Find the best result
        best_ind = tools.selBest(result, k=1)[0]
        print('Best Individual: ', best_ind)
        print('Best Fitness: ', best_ind.fitness.values[0])
        
        # 1. Rozbalte vítězné parametry z listu 'best_ind'
        (D_mid_best, n_design_best, alpha_stator_best, beta_rotor_best, beta_rotor_2_best, chord_opt_best, eps_opt_best) = best_ind

        # 2. Znovu spusťte simulaci s těmito nejlepšími parametry
        best_res = meanline_design(
            D_mid_best, 
            n_design_best, 
            alpha_stator_best, 
            eta_guess,  # fixní parametr
            p_inlet,    # fixní parametr
            T_inlet,    # fixní parametr
            beta_rotor_best, 
            m_dot,      # fixní parametr
            eps_opt_best,
            sigma,      # fixní parametr
            beta_rotor_2_best,
            chord_opt_best
        )

        # 3. Vytiskněte detailní výsledky pro vítěznou kombinaci
        print("\n--- Výsledky pro vítěznou kombinaci ---")
        print(best_res)
        print(f"Mechanický výkon (P_mech): {best_res['P_mech'] / 1000:.2f} kW")
        print(f"Mach na výstupu z trysek: {best_res['Ma_act'][-1]:.3f}")
        #best individual
        print(f"Nejlepší individuální parametry: {best_ind}")
        pool.close()
        pool.join()

        def _np_to_native(o):
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            # Last-resort stringification (rarely needed)
            return str(o)

        #save best res to file
        with open("best_res.json", "w") as f:
            json.dump(best_res, f, default=_np_to_native, indent=2)

        plot_convergence(log) # Plot the convergence of the GA
        
        toc = timeit.default_timer()
        print(f"Elapsed time: {toc-tic} s")

    #sensitivity_analysis_rpm_pool(n_design,D_mid,alpha_stator,eta_guess,p_inlet,T_inlet)
    #sensitivity_analysis_alpha_stator(n_design,D_mid,alpha_stator,eta_guess)
    #sensitivity_analysis_D_mid(n_design,D_mid,alpha_stator,eta_guess)

        if PLOTTING:
            #load best res from file
            with open("best_res.json", "r") as f:
                best_res = json.load(f)

            #uncomment as you wish to draw the plots for best res
            draw_nozzle_expansion(best_res["PR_is"],best_res["Ma_is"],best_res["p_is"],best_res["throat_index"],best_res["PR_act"],best_res["Ma_act"])
            draw_expansion_line(best_res["s"],best_res["h"],best_res["s_nozzle_out"],best_res["h_nozzle_out"],best_res["s2"],best_res["h2"],best_res["h2t"],best_res["s_out_eta"],best_res["h_out_eta"],best_res["p_inlet"],best_res["p_outlet"],best_res["p_outlet_total"],best_res["fluid"],best_res["z"])
            draw_velocity_triangles(best_res["c1u"],best_res["c1a"],best_res["w1u"],best_res["w1a"],best_res["U"],best_res["c2u"],best_res["c2a"],best_res["w2u"],best_res["w2a"],best_res["alpha_stator"],best_res["beta2"],best_res["alpha_rotor"])

     # --- Optional Gamma plots (toggle with ENABLE_GAMMA_PLOTS) ---
    if ENABLE_GAMMA_PLOTS:
        # T–s Γ map
        plot_gamma_Ts(fluid_name=fluid,
                      T_min=300.0, T_max=450.0,
                      s_min=900.0, s_max=2000.0,
                      steps=1200, outfile="Gamma_T_s_diagram.png")

        # p–v Γ map
        plot_gamma_Pv(fluid_name=fluid,
                      P_min=1e4, P_max=4e6,
                      v_min=1.5e-3, v_max=5e-1,
                      steps=1200, outfile="Gamma_P_v_diagram.png")