import numpy as np 
import scipy.special as special
import scipy.signal
import matplotlib.pyplot as plt
import cv2

def propagator(plane_size = 512, grid = 1, propagate_distance = 500, wavelength = 1):
    '''
        plane_size: the resolution of the propagator
        grid: the physical size of one cell on the plane.
        prop distance: physical distance.
        wavelength: physical wavelength. 
        The unit of the physical quantity is some fake unit. Eg: I can take 10^-6 meter as 1.
    '''
    # this is near to far field transform
    def Near2far(x1,y1,z1, wavelength):
        r = np.sqrt((x1)**2+(y1)**2+(z1)**2)
        k =2*np.pi/wavelength

        H = special.hankel1(1, k*r)

        g = -1j*k/4*H*z1/r
        return g
    def W(x, y, z, wavelength):
        r = np.sqrt(x*x+y*y+z*z)
        #w = z/r**2*(1/(np.pi*2*r)+1/(relative_wavelength*1j))*np.exp(1j*2*np.pi*r/relative_wavelength)
        w = z/(r**2)*(1/(wavelength*1j))*np.exp(1j*2*np.pi*r/wavelength)
        return w
    
    #plane_size: the numerical size of plane, this is got by (physical object size)/(grid)

    x = np.arange(-(plane_size-1), plane_size,1) * grid
    y = x.copy()
    coord_x, coord_y = np.meshgrid(x,y, sparse = False)
    G = W(coord_x, coord_y, propagate_distance, wavelength)
    # theta = np.arctan(plane_size * grid/propagate_distance)
    # Sigma = 2 * np.pi * (1 - np.cos(theta))
    # G_norm = (np.abs(G)**2).sum() * 4 * np.pi / Sigma 
    # print(f"Free space energy conservation normalization G_norm: {G_norm:.2f}")
    # G = G / G_norm
    G = np.reshape(G, (1,1) + G.shape)

    return G

def lens_profile(plane_size, grid, focal_length, wavelength):
    
    x = (np.arange(plane_size) - plane_size // 2) * grid
    y = x.copy()
    coord_x, coord_y = np.meshgrid(x,y, sparse = False)
    
    radius = plane_size * grid * np.sqrt(2) / 2
    
    thickness = (radius**2 - coord_x**2 - coord_y**2) / (2 * focal_length)
    
    phase = thickness * (2 * np.pi / wavelength)
    
    phase = np.reshape(phase, (1, 1, plane_size, plane_size))
    return phase

def init_aperture(plane_size):
    A = np.zeros((plane_size, plane_size))
    c = (plane_size//2, plane_size//2)
    radius = plane_size//2
    A = cv2.circle(A, c, radius, color = 1, thickness = -1)
    A = np.reshape(A, (1, 1, plane_size, plane_size))
    
    return A