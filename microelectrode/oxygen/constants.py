import numpy as np

def O2_solubility(sal,tempC):
    # O2 solutility in umol/kg from Garcia and Gordon 1992
    
    A0 = 5.80818
    A1 = 3.20684
    A2 = 4.11890
    A3 = 4.93845
    A4 = 1.01567
    A5 = 1.41575
    B0 = -7.01211e-3
    B1 = -7.25958e-3
    B2 = -7.93334e-3
    B3 = -5.54491e-3
    C0 = -1.32412e-7

    Ts = np.log((298.15 - tempC) / (273.15 + tempC))

    lnC0 = (A0 + A1 * Ts + A2 * Ts**2 + A3 * Ts**3 + A4 * Ts**4 + A5 * Ts**5 +
            sal * (B0 + B1 * Ts + B2 * Ts**2 + B3 * Ts**3) + C0 * sal**2)

    return np.exp(lnC0)

def O2_diffusivity(tempC, sal):
    # O2 diffusivity in cm2/s from Li and Gregory, 1974
    return

def seawater_density(sal, tempC):
    # seawater density from millero and Poisson 1981 in kg/L
    rho_0 = (
        999.842594 + 
        6.793952e-2 * tempC -
        9.095290e-3 * tempC**2 +
        1.001685e-4 * tempC**3 -
        1.120083e-6 * tempC**4 +
        6.536332e-9 * tempC**5
    )

    A = (
        8.24493e-1 - 
        4.0899e-3 * tempC + 
        7.6438e-5 * tempC**2 - 
        8.2467e-7 * tempC**3 + 
        5.3875e-9 * tempC**4
    )

    B = (
        -5.72466e-3 + 
        1.0227e-4 * tempC - 
        1.6546e-6 * tempC**2
    )

    C = 4.8314e-4

    rho_sal = rho_0 + A * sal + B * sal**1.5 + C * sal**2

    return rho_sal / 1000  # convert to kg/L

def seawater_dynamic_viscocity(sal, tempC):
    # sewater dynamic viscocity from Sharqawy et al., 2010

    return
