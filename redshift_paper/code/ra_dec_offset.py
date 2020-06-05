#! /usr/bin/env python

#########################################################################

from __future__ import print_function
import numpy as np

#########################################################################

def change_ra(ra):
    if ':' in ra:
        hour, min, sec = ra.split(':')
        ra = (int(hour) + int(min)/60. + float(sec)/3600.) /24.* 360.
        return ra
    else:
        return float(ra)


def change_dec(dec):
    if ':' in dec:
        deg, min, sec = dec.split(':')
        dec = int(deg) + int(min)/60. + float(sec)/3600.
        return dec
    else:
        return float(dec)


#########################################################################


ra1  = input('Enter RA in either hour:min:sec format OR deg.xxx format:   ')
dec1 = input('Enter dec in either deg:min:sec format OR deg.xxx format:   ')

ra2  = input('Enter RA in either hour:min:sec format OR deg.xxx format:   ')
dec2 = input('Enter dec in either deg:min:sec format OR deg.xxx format:   ')

ra1 = change_ra(ra1)
ra2 = change_ra(ra2)
dec1 = change_dec(dec1)
dec2 = change_dec(dec2)


ra_offset  = (ra2-ra1) * np.cos((dec1+dec2)/2 * np.pi/180.)
dec_offset = dec2-dec1


print("\nRA Offset: \t",  (ra_offset* 3600.) ,  "\t in arcsec")
print("dec Offset: \t", dec_offset* 3600., "\t in arcsec\n")


separation = np.sqrt( (ra_offset*60)**2 + (dec_offset*60)**2 )
print("Separation: \t", separation, "\t in arcmin\n\n")

