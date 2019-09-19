#!/usr/bin/env python

from collections.abc import Iterable
import numpy as np

# Constants
RAD2SEC  = 206265       # Arcseconds in a Radian
DEG2RAD  = np.pi / 180  # Radians in a Degree
MIN2SEC  = 60           # (Arc)Seconds in a(n) (Arc)Minute
DEG2SEC  = 3600         # Arc-seconds in a Degree
HOUR2DEG = 360/24       # Degrees in an Hour

NUM_DECIMAL_PLACES = 3  # User-defined rounding preference


################################################################################
"""
CONVERSIONS.py
    List of useful methods for common unit conversions in astronomy for the
    local universe, where Hubble's Law (v = cz = H0*d) applies.
    
    Methods:
    (1) GET_ABSOLUTE_MAGNITUDE: Converts apparent to absolute magnitudes.
    (2) COORD2DEG: Converts coordinates in HH:MM:SS/DD:MM:SS format to degrees.
    (3) GET_ANGULAR_SIZE: Computes angular distance between 2 coordinates.
    (4) GET_PHYSICAL_SIZE: Computes physical distance given angular size.
"""
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
        H0 (float): Hubble's Constant [(km/s) / Mpc]
    
    RETURNS:
        abs_magnitude (float): Object's Absolute Magnitude
    """
    abs_magnitude = app_magnitude - 5 * np.log10(velocity / H0) - 25 - extinction

    return np.round(abs_magnitude, NUM_DECIMAL_PLACES)


################################################################################

def convert_single_coord(coord, right_ascension):
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
            ra = (int(hour) + int(min)/MIN2SEC + float(sec)/DEG2SEC) * HOUR2DEG
            return ra
        else:
            # Converts DD:AM:AS --> degree
            deg, min, sec = coord.split(':')
            dec = int(deg) + int(min)/MIN2SEC + float(sec)/DEG2SEC
            return dec

#------------------------------------------------------------------------------#

def coord2degree(coord, right_ascension=True):
    """
    Converts a single or an iterable of coords to degrees.
    
    ARGS:
        coord (float/str -or- array of floats/strs)
        
    RETURNS:
        (float -or- array of floats): Converted coordinate(s) in degrees
    """
    # Checks if coord is a single value or an iterable
    if not isinstance(coord, Iterable):
        return convert_single_coord(coord, right_ascension)
    
    return np.array([convert_single_coord(c, right_ascension) for c in coord])

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
    ra_offset  = DEG2SEC * (ra2-ra1) * np.cos((dec1+dec2)/2 * DEG2RAD)
    dec_offset = DEG2SEC * (dec2-dec1)

    # Computes Angular Separation (in arcsec)
    angular_size = np.sqrt( ra_offset**2 + dec_offset**2 )

    return np.round(angular_size, NUM_DECIMAL_PLACES)


################################################################################

def get_physical_size(angular_size, velocity, H0=70):
    """
    Computes the projected physical size/separation for a given angular size or
    separation in the local universe.
    
    ARGS:
        angular_size (float): Angular size/separation (arcsec) between 2 points.
        velcity (float): Recessional Velocity (km/s) due to expansion
        H0 (float): Hubble's Constant [(km/s) / Mpc]
        
    RETURNS:
        physical_size (float): Projected physical size/separation (Mpc)
                               between two points.
    """
    # Computes Physical Separation (in Mpc)
    physical_size = angular_size * velocity / RAD2SEC / H0
    
    return np.round(physical_size, NUM_DECIMAL_PLACES)


################################################################################
