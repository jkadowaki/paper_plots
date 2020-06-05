#!/usr/bin/env python
import numpy as np

# Constants
RAD_TO_ARCSEC = 206265  # arcseconds in a radian
C = 299792.458 # km/s

################################################################################

def get_kpc(angular_size, velocity, H0=70):
    
    """
    GET_KPC: Computes an object's physical size given its angular
    size, it's recessional veloctiy, and an assumed
    Hubble's constant.
    
    ARGS:
    (float) angular_size: Object's Angular Size in Arcseconds (")
    (float) velocity: Recessional Velocity in (km/s) Attributed to the
    Universe's Expansion Rate
    (float) H0: Hubble's Constant in (km/s)/Mpc
    
    RETURNS:
    (float) physical_size: Object's Physical Size in kpc
    """
    
    return 1000 * velocity * angular_size / H0 / RAD_TO_ARCSEC

################################################################################

def get_arcsec(physical_size, velocity, H0=70):
    
    """
    GET_ARCSEC: Computes an object's physical size given its angular
    size, it's recessional veloctiy, and an assumed
    Hubble's constant.
    
    ARGS:
    (float) physical_size: Object's Physical Size in kpc
    (float) velocity: Recessional Velocity in (km/s) Attributed to the
    Universe's Expansion Rate
    (float) H0: Hubble's Constant in (km/s)/Mpc
    
    RETURNS:
    (float) angular_size: Object's Angular Size in Arcseconds (")
    """
    
    return  physical_size * H0 * RAD_TO_ARCSEC / 1000 / velocity

################################################################################

def main(velocity=7000, physical_size=None, angular_size=None, H0=70, Mg=-19):

    print("\nVelocity:", velocity, "km/s")
    print("Redshift:", round(velocity/C, 5))
    
    mu = round(5 * np.log10(velocity/H0) + 25, 5)
    print("Distance Modulus:", mu)
    print("Mg =", Mg, "mag \t-->\tm_g =", round(Mg+mu,5), "mag")

    if physical_size:
        arcsec = get_arcsec(physical_size, velocity, H0=70)
        arcmin = arcsec/60
        print(physical_size, "kpc \t-->\t", round(arcsec,3), "arcsec\t",
                                            round(arcmin,3), "arcmin\n")

    elif angular_size:
        kpc = get_kpc(angular_size, velocity, H0=70)
        print(angular_size, "arcsec \t-->\t", round(kpc,3), "kpc\n")

    else:
        print("Call function as:\n",
              "'python convert_kpc_arcsec.py velocity physical_size angular_size'")

################################################################################

if __name__ == '__main__':
    
    import sys
    
    try:
        main(velocity=float(sys.argv[1] if sys.argv[2]!='None' else 7000),
             physical_size=float(sys.argv[2]) if sys.argv[2]!='None' else None,
             angular_size=float(sys.argv[3]) if sys.argv[3]!='None' else None,
             H0=67.37, Mg=-19)
    except:
        print("\nCall function as:\n",
              "'python convert_kpc_arcsec.py velocity physical_size angular_size'", "\n")

