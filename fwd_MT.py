import numpy as np
import cmath
import math

def f_MT(freq,ind_desimal,n_lapis):
    rho_m = ind_desimal[0:n_lapis]
    thick = ind_desimal[n_lapis:(2*n_lapis)]
    k   = freq.shape
    Rho_ap = np.zeros(k)  
    phs_ap = np.zeros(k)
    sum=0
    for frequency in freq:
        w =  2*np.pi*frequency;   #kecepatan sudut
        mu = 4*np.pi*1E-7;        #permeabilitas magnetic       
        z0 = list(range(n_lapis));
        z0[n_lapis-1] = cmath.sqrt(cmath.sqrt(-1)*w*mu*rho_m[n_lapis-1]);
        for j in range(n_lapis-2,-1,-1):
            k = cmath.sqrt(cmath.sqrt(-1)*w * mu/rho_m[j]);  #bilangan gelombang
            Z0j = k * rho_m[j];
            rj = (Z0j - z0[j + 1])/(Z0j + z0[j + 1]);
            ej = cmath.exp(-2*thick[j]*k);
            re = rj*ej; 
            zj = Z0j * ((1 - re)/(1 + re));       
            z0[j]= zj ;    
        Z = z0[0];
        Zreal = abs(Z);
        #freq_nya = math.log10(frequency)
        
        Rho_ap[sum] = (Zreal* Zreal)/(mu * w);
        phs_ap[sum] = math.atan2(Z.imag, Z.real);
        sum = sum + 1
    return(Rho_ap,phs_ap)        
        
