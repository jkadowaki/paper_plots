#!/usr/bin/env python

import numpy as np

# Constants
RAD_TO_ARCSEC = 206265  # arcseconds in a radian

################################################################################

def get_physical_size(velocity, angular_size, H0=70):
    """
    Computes an object's physical size given its angular size, its recessional
    veloctiy, and an assumed Hubble's constant.
        
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

def get_absolute_magnitude(magnitude, velocity, extinction=0, H0=70):
    """
    Computes an object's absolute magnitude given its apparent magnitude,
    its recessional veloctiy, and an assumed Hubble's constant.
    
    ARGS:
        magnitude (float): Object's Apparent Magnitude
        velocity (float): Recessional Velocity in (km/s) Attributed to the
                          Universe's Expansion Rate
        extinction (float): The amount of extinction due to line-of-sight dust
        H0 (float): Hubble's Constant (km/s / Mpc)
    
    RETURNS:
        (float): Object's Absolute Magnitude
    """
    
    return magnitude - 5 * np.log10(velocity / H0) - 25 - extinction


################################################################################

def get_separation(ra1, dec1, ra2, dec2, velocity, H0=70):
    """
    Computes the projected physical separation between two objects given their
    coordinates and their velocity.
    
    ARGS:
        ra1  (float or string): 1st Object's Right Ascension
        dec1 (float or string): 1st Object's Declination
        ra2  (float or string): 2nd Object's Right Ascension
        dec2 (float or string): 2nd Object's Declination
        velcity (float): Recessional Velocity (km/s)
        H0 (float): Hubble's Constant (km/s / Mpc)
        
    RETURNS:
        (float) The projected physical separation between the two objects in Mpc.
    """
    ############################################################################
    
    def change_ra(ra):
        try:
            return float(ra)
        except:
            hour, min, sec = ra.split(':')
            ra = (int(hour) + int(min)/60. + float(sec)/3600.) /24.* 360.
            return ra


    def change_dec(dec):
        try:
            return float(dec)
        except:
            deg, min, sec = dec.split(':')
            dec = int(deg) + int(min)/60. + float(sec)/3600.
            return dec

    ############################################################################

    ra1 = change_ra(ra1)
    ra2 = change_ra(ra2)
    dec1 = change_dec(dec1)
    dec2 = change_dec(dec2)

    # Computes RA & Dec Offsets (in arcsec)
    ra_offset  = 3600 * (ra2-ra1) * np.cos((dec1+dec2)/2 * np.pi/180.)
    dec_offset = 3600 * (dec2-dec1)

    # Computes Angular Separation (in arcsec)
    angular_separation = np.sqrt( ra_offset**2 + dec_offset**2 )

    # Computes Physical Separation (in Mpc)
    return round(angular_separation * velocity / RAD_TO_ARCSEC / H0, 3)
