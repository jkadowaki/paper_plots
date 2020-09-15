#!/usr/bin/env python

################################################################################

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy.random import rand, seed

font = {'size': 18}
plt.rc('text', usetex=True)
plt.rc('font', **font)


# Seed the Random Number Generator for Reproducible Results
seed(0)


################################################################################

def sample_angles(num=100):
    """
    Args:
        num (float): Number of Samples

    Returns:
        theta (float): Transformation angle [0, pi)
        phi (float): Transformation angle [0, 2pi)
    """

    # Random angle with sin(x) distribution between [0, pi) for rotation 
    theta = np.arccos(1 - rand(num))
    
    # Uniformly random angle between [0,2pi) for rotation about the z-axis.
    phi   = 2 * np.pi * rand(num) 

    return theta, phi


################################################################################

def find_angle(theta, phi):
    """
    Args:
        theta (float): Transformation angle about the observer's line-of-sight
        phi (float): Transformation angle about an orthogonal axis

    Returns:
        w (float): Angle identifying the direction of the semi-major or semi-
                   minor axis of the projected ellipse.

    BACKGROUND:
    We transform a circle (i.e., a thin disk) centered at the origin in the
    xy-plane with 2 rotations: angle theta about the x-axis and angle phi about
    the z-axis to simulate a randomly oriented disk. We project this
    transformation onto the yz-plane, projecting what an observer along the
    x-axis would see.
    To find the projected axis ratio, we first need to find a point (x',y',z')
    on the ellipse, (i.e., originally point (x,y,z) = (r cos(w), r sin(w), 0) on
    the circle), which maximizes/minimizes the distance from the origin. The
    minimum/maximum distance from the origin will occur an angle w' = w = pi/2.
    The ratio between the shortest and longest distance on the ellipse from the
    origin will yield the projected axis ratio.

    METHOD:
    This method computes angle w, identifying the direction to the semi-major
    or semi-minor axis of the ellipse. 
    """

    # Derivation:
    # [ x  y  z ] = [ r*cos(w)   r*sin(w)   0 ]
    # [ x' y' z'] = Proj_{yz} R_z R_x [ x y z ]
    #             = [ 0   r*sin(phi)*cos(w)+r*cos(phi)*cos(theta)*sin(w)   r*sin(theta)*sin(w) ]
    # D = sqrt(y'^2 + z'^2)  <-- Distance from origin
    # d(D)/dw = 0
    term1 = np.tan(theta) * np.sin(theta) / np.sin(phi) / np.cos(phi)
    term2 = np.cos(theta) / np.tan(phi)
    term3 = np.tan(phi) / np.cos(theta)

    return np.arctan(-2 / (term1 + term2 - term3)) / 2


################################################################################

def get_y(theta, phi, w, r=1):
    """
    Computes the y-coordinate of a point on the ellipse projected on the yz-plane.
    
    Args:
        theta (float): Transformation angle about the observer's line-of-sight
        phi (float): Transformation angle about an orthogonal axis
        w (float): Angle identifying the direction of the semi-major or semi-
                   minor axis of the projected ellipse.
        r (float): Radius of circle (i.e., thin disk)

    Returns:
        y (float): y-coordinate of a point specified by angle w on the ellipse
    """
    return np.sin(phi) * r * np.cos(w) + np.cos(phi) * np.cos(theta) * r * np.sin(w)


def get_z(theta, w, r=1):
    """"
    Computes the z-coordinate of a point on the ellipse projected on the yz-plane.
    
    Args:
        theta (float): Transformation angle about the observer's line-of-sight
        phi (float): Transformation angle about an orthogonal axis
        w (float): Angle identifying the direction of the semi-major or semi-
                   minor axis of the projected ellipse.
        r (float): Radius of circle (i.e., thin disk)
    
    Returns:
        z (float): z-coordinate of a point specified by angle w on the ellipse
    """
    return np.sin(theta) * r * np.sin(w)

################################################################################

def distance_from_origin(theta, phi, w, r=1):
    """"
    Computes the distance of a point on the ellipse projected on the yz-plane
    fron the origin.
    
    Args:
        theta (float): Transformation angle about the observer's line-of-sight
        phi (float): Transformation angle about an orthogonal axis
        w (float): Angle identifying the direction of the semi-major or semi-
                   minor axis of the projected ellipse.
        r (float): Radius of circle (i.e., thin disk)

    Returns:
        dist (float): Distance of point on the ellipse on the semi-major or
                      semi-minor axis from the origin 
    """
    
    return np.sqrt( get_y(theta, phi, w, r)**2 + get_z(theta, w, r)**2 )

################################################################################

def get_axisratio(theta, phi, r=1):
    """
    Computes the axis ratio by identifying the closest and furthest points on
    an ellipse and getting the ratio of their distances.
    
    Args:
        theta (float): Transformation angle about the observer's line-of-sight
        phi (float): Transformation angle about an orthogonal axis
        r (float): Radius of circle (i.e., thin disk)

    Returns:
        axis_ratio (float): The projected axis ratio of the ellipse
    """
    w         = find_angle(theta, phi)
    semimajor = distance_from_origin(theta, phi, w, r)
    semiminor = distance_from_origin(theta, phi, w + np.pi/2, r)

    axis_ratio = semiminor / semimajor

    return np.array([elem if elem < 1 else 1/elem for elem in axis_ratio])

################################################################################

def monte_carlo(num=100, plot_fname='monte_carlo_thindisk.pdf'):
    """
    Performs monte carlo simulation by randomly sampling (theta, phi) on a sphere
    to derive the distribution of axis ratios of randomly oriented disks.

    Args:
        num (int): Number of realizations in the monte carlo simulation
        plot_fname (str): Filename of histogram results from the simulation
    """
    theta, phi = sample_angles(num)

    axis_ratios = get_axisratio(theta, phi, r=1)
    bins = np.arange(0,1.1,0.1)

    
    plt.figure(figsize=(8,5))
    plt.hist(axis_ratios, bins, histtype='step', #density=True,
             linewidth=3, alpha=0.2, color='red',
             label=r'$\mathrm{Randomly \, Oriented \, Thin \, Disks}$')
    plt.xlabel(r'$\mathrm{Axis \, Ratio}$', fontsize=22)
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.close()

    """
    # Uncomment code block to view sampling distribution of (theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, marker='.', s=1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    plt.savefig('monte_carlo_sampling.pdf')

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.scatter(x,y, marker='.', s=1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.savefig('monte_carlo_xy.pdf')

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.scatter(y, z, marker='.', s=1)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$z$')
    plt.savefig('monte_carlo_yz.pdf')
    """

################################################################################

if __name__ == "__main__":

    monte_carlo(10000)

################################################################################
