# !/usr/bin/env python3
# ----------------------------------
# Compute Moment of Inertia and CmC
# ----------------------------------
# Author: Stefano Bertone
# Created: 30-Jul-2020

import numpy as np


def G201(e):

    return (1-e**2)**(-3./2)

def G210(e):

    return (7/2)*e - (123/16.)*(e**3) + (489/128.)*(e**5)

def MoI(obliq, n, e, C20, C22, dom_sini, dom_cosi):

    print("G201",G201(e))
    print("G210",G210(e))

    # derived from Eq.1 of Baland 2017 (typo in Genova 2019 supp.mat.!!)
    out = obliq*(2.*n*G210(e)*C22-n*G201(e)*C20)/(dom_sini + obliq*dom_cosi)

    return out*-1

def MoI_error(obliq, n, e, C20, C22, dom_sini, dom_cosi, err_obliq, err_C20, err_C22):

    # dc/c = sqrt((dc/deps * s_eps)^2+(dc/dC20 * s_C20)^2+(dc/C22 * s_C22)^2)
    num = (2. * n * G210(e) * C22 - n * G201(e) * C20)
    den = (dom_sini + obliq * dom_cosi)
    k =  num / den
    term1 = k - (obliq*num*dom_cosi)/den**2
    term2 = -1.*obliq*n * G201(e)/den
    term3 = obliq*2. * n * G210(e)/den

    print("error prop:",(k*err_obliq)**2,((obliq*num*dom_cosi)/den**2*err_obliq)**2,(term1*err_obliq)**2,(term2*err_C20)**2,(term3*err_C22)**2)

    # return MoI(obliq, n, e, C20, C22, dom_sini, dom_cosi)*np.sqrt((term1*err_obliq)**2+(term2*err_C20)**2+(term3*err_C22)**2)
    return np.sqrt((term1*err_obliq)**2+(term2*err_C20)**2+(term3*err_C22)**2)

def MoI_mantle(moi,e,L,C22):

    print(L)
    print(moi)
    print((1.-11*e**2+959/48.*e**4))
    print(C22)
    return 1.5*(1.-11*e**2+959/48.*e**4)*4*C22/L/moi

def MoI_mantle_err(MoI_mantle,C22,moi,L,err_C22,err_moi,err_L):

    term1 = (MoI_mantle/C22*err_C22)**2
    term2 = (-1.*MoI_mantle/moi*err_moi)**2
    term3 = (-1.*MoI_mantle/L*err_L)**2

    print("moimantleerr",term1,term2,term3)
    # return MoI_mantle*np.sqrt(term1+term2+term3)
    return np.sqrt(term1+term2+term3)

def normal_fact_kaula(n,m):

    if m == 0:
        return np.sqrt((2*n+1))
    else:
        return np.sqrt(2*(2*n+1)*np.math.factorial(n-m)/np.math.factorial(n+m))

if __name__ == '__main__':

    obliq = np.array([1.97139,2.031]) *np.pi/(60.*180.) # rad
    err_obliq = np.array([0.027,0.029]) *np.pi/(60.*180.) # rad

    print("eps",obliq,err_obliq)

    # librations from IAU and Bertone
    L = np.array([38.5,39.03])*np.pi/(3600.*180.)
    err_L = np.array([1.6,0.95])*np.pi/(3600.*180.)

    print("L",L,err_L)

    # values from Baland et al 2017, appendix A
    n = 4.092345556*np.pi/180.*365. # rad/y
    e = 0.2056318

    dom_sini = -2.864081e-6 # /y
    dom_cosi = -19.088758e-6 # /y

    A = 6.6374e-5
    B = 3.2166e-5

    C20 = B/2. - A #-5.03216e-5 #
    C22 = B/4. # 0.80389e-5 #

    err_C20 = 4.e-8
    err_C22 = 5.e-9

    print(C20, C22)
    print("obliq (AG, SB):",obliq)

    moi = MoI(obliq,n,e,C20,C22,dom_sini,dom_cosi)
    err_moi = MoI_error(obliq, n, e, C20, C22, dom_sini, dom_cosi, err_obliq, err_C20, err_C22)
    print("MoI (Genova, Bertone)=", moi)
    print("MoI_error (Genova, Bertone)=", err_moi)

    Cmc = MoI_mantle(moi, e, L, C22)
    print("MoI mantle=",Cmc)
    print("MoI_mantle_error (Genova, Bertone)=", MoI_mantle_err(Cmc,C22,moi,L,err_C22,err_moi,err_L))

    moi = 0.344
    print("Check: For MoI=",moi, "eps=",-1*(moi*dom_sini/(moi*dom_cosi+2*n*G210(e)*C22-n*G201(e)*C20))*60*180./np.pi)

    # Konopliv
    obliq = np.array([1.99]) *np.pi/(60.*180.) # rad
    err_obliq = np.array([0.12]) *np.pi/(60.*180.) # rad

    print("eps",obliq,err_obliq)

    C20 = -1*normal_fact_kaula(2,0)*2.25025369765e-5 # B/2. - A #-5.03216e-5 #
    C22 = normal_fact_kaula(2,2)*1.24553974706e-5 # B/4. # 0.80389e-5 #

    print(C20,C22)

    err_C20 = 5.812e-9 #4.e-8
    err_C22 = 8.094e-9 # 5.e-9

    moi = MoI(obliq,n,e,C20,C22,dom_sini,dom_cosi)
    err_moi = MoI_error(obliq, n, e, C20, C22, dom_sini, dom_cosi, err_obliq, err_C20, err_C22)
    print("MoI (Konopliv)=", moi)
    print("MoI_error (Konopliv)=", err_moi)

    L = np.array([38.5])*np.pi/(3600.*180.)
    err_L = np.array([1.6])*np.pi/(3600.*180.)

    Cmc = MoI_mantle(moi, e, L, C22)
    print("MoI mantle (Konopliv)=",Cmc)
    print("MoI_mantle_error (Konopliv)=", MoI_mantle_err(Cmc,C22,moi,L,err_C22,err_moi,err_L))
