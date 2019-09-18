#!/usr/bin/env python

from collections.abc import Iterable
import numpy as np

# Constants
RAD_TO_ARCSEC = 206265  # arcseconds in a radian


################################################################################

def get_absolute_magnitude(app_magnitude, velocity, extinction=0, H0=70):
    """
    Computes an object's absolute magnitude given its apparent magnitude,
    its recessional veloctiy, and an assumed Hubble's constant.
    
    ARGS:
        app_magnitude (float): Object's Apparent Magnitude
        velocity (float): Recessional Velocity in (km/s) Attributed to the
                          Universe's Expansion Rate
        extinction (float): The amount of extinction due to line-of-sight dust
        H0 (float): Hubble's Constant (km/s / Mpc)
    
    RETURNS:
        abs_magnitude (float): Object's Absolute Magnitude
    """
    
    abs_magnitude = app_magnitude - 5 * np.log10(velocity / H0) - 25 - extinction

    return abs_magnitude


################################################################################

def single_coord_conversion(coord, right_ascension):
    """
    Converts 1 RA or Declination coordinate to degrees.
        
    ARGS:
        coord (float/str): Right ascension or declination in fractional
                           degrees or in HH:MM:SS (RA) / DD:MM:SS (dec).
        right_ascension (bool): Flag to indicate right ascension (True)
                                or declination (False).

    RETURNS:
        degree (float)
    """
    try:
        # Return if coord is already in units of fractional degrees
        return float(coord)
    except:
        if right_ascension:
            # Converts HH:MM:SS --> degree
            hour, min, sec = coord.split(':')
            ra = (int(hour) + int(min)/60. + float(sec)/3600.) /24.* 360.
            return ra
        else:
            # Converts DD:AM:AS --> degree
            deg, min, sec = coord.split(':')
            dec = int(deg) + int(min)/60. + float(sec)/3600.
            return dec

#------------------------------------------------------------------------------#

def coord2degree(coord, right_ascension=True):
    """
    Converts a single or an iterable of coords to degrees.
    ARGS:
        coord (float/str -or- array of floats/strs)
    RETURNS:
        degree (float -or- array of floats)
    """
    if not isinstance(coord, Iterable):
        return single_coord_conversion(coord, right_ascension)
    
    return np.array([single_coord_conversion(coo, right_ascension) for coo in coord])

################################################################################

def get_angular_size(ra1, dec1, ra2, dec2):
    """
    Computes the projected angular size/separation between two points.
    
    ARGS:
        ra1  (float or string): 1st Object's Right Ascension
        dec1 (float or string): 1st Object's Declination
        ra2  (float or string): 2nd Object's Right Ascension
        dec2 (float or string): 2nd Object's Declination
        
    RETURNS:
        angular_size (float): Angular size/separation (arcsec) between 2 points.
    """
    
    ra1  = coord2degree(ra1,  right_ascension=True)
    ra2  = coord2degree(ra2,  right_ascension=True)
    dec1 = coord2degree(dec1, right_ascension=False)
    dec2 = coord2degree(dec2, right_ascension=False)

    # Computes RA & Dec Offsets (in arcsec)
    ra_offset  = 3600 * (ra2-ra1) * np.cos((dec1+dec2)/2 * np.pi/180.)
    dec_offset = 3600 * (dec2-dec1)

    # Computes Angular Separation (in arcsec)
    angular_size = np.sqrt( ra_offset**2 + dec_offset**2 )

    return angular_size


################################################################################

def get_physical_size(angular_size, velocity, H0=70):
    """
    Computes the projected physical size/separation for a given angular size or
    separation in the local universe (which obeys Hubble's Law: v = cz = H0*d).
    
    ARGS:
        angular_size (float): Angular size/separation (arcsec) between 2 points.
        velcity (float): Recessional Velocity (km/s) due to expansion
        H0 (float): Hubble's Constant [(km/s) / Mpc]
        
    RETURNS:
        physical_size (float): Projected physical size/separation (Mpc)
                               between two points.
    """

    # Computes Physical Separation (in Mpc)
    physical_size = round(angular_size * velocity / RAD_TO_ARCSEC / H0, 3)
    
    return physical_size

################################################################################
