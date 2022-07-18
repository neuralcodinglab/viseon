import numpy as np
import simulator.cortex_models as cortex_models

import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def init_magnification(params, r):
    ECCENTRICITY_SCALING = cortex_models.get_cortical_magnification(params['cortex_model'])
    magnification = ECCENTRICITY_SCALING(r)
    return magnification

def get_truncated_normal(size, mean, sd, low=0., upp=1.):
    return truncnorm.rvs((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd, size=size)

#def init_threshold_curves(n_phosphenes, threshold_range, slope_range):
#    threshold = threshold_range[0]+(np.random.power(0.4, size=n_phosphenes)*(threshold_range[1]-threshold_range[0]))  # TODO: threshold dist based on Schmidt et al., 1996 
#    slope = np.random.uniform(low=slope_range[0], high=slope_range[1], size=n_phosphenes)
#
#    return threshold, slope 

def init_threshold_curves(n_phosphenes, threshold_mean, threshold_sd, slope_range):
    threshold = get_truncated_normal(n_phosphenes, threshold_mean, threshold_sd)
    slope = np.random.uniform(low=slope_range[0],high=slope_range[1], size=n_phosphenes)
    return threshold, slope
def init_from_cortex(cortex_params, coords=None):
    """Transform electrode locations into a phosphene map, given certain parameters for visuotopic models

    :param cortex_params: parameters for the visuotopic model used
    :param coords: tuple (x_coords, y_coords) describing the locations of electrodes on a cortex map, in mm
    :return: location of phosphenes in the visual field, given in polar coordinates
    """

    if coords is None:
        x_coords, y_coords = cortex_models.init_covering_electrode_grid(32, 32, max_x=40, mapping_model=cortex_params['model'], plot=False)
    else:
        x_coords, y_coords = coords

    r, theta = cortex_models.get_phosphene_map_from_electrodes(cortex_params, x_coords, y_coords)

    return r, theta

def init_probabilistically(params, n_phosphenes):
    """generate a number of phosphene locations probabilistically

    :param n_phosphenes: number of phosphenes
    :return: polar coordinates of n_phosphenes phosphenes
    """
    #random generator
    rng = np.random.default_rng()
    max_r = params['run']['view_angle']/2
    # Spatial phosphene characteristics
    # ECCENTRICITY_SCALING = cortex_models.get_cortical_magnification(params['cortex_model'])#lambda r: 17.3/(0.75+r) #mm activated cortex per degree , Horten & Hoyt, cortical magnification
    # ECC_PROB = lambda r: (ECCENTRICITY_SCALING(r)-ECCENTRICITY_SCALING(0.)) / (ECCENTRICITY_SCALING(max_r)-ECCENTRICITY_SCALING(0.)) #probability version of cortical magnification
    # SAMPLE_RADIAL_DISTR  = lambda n: ECC_PROB(rng.random(n))*max_r
    ECCENTRICITY_SCALING = cortex_models.get_cortical_magnification(params['cortex_model'])
    valid_ecc = np.linspace(1e-3,max_r,1000)
    weights = ECCENTRICITY_SCALING(valid_ecc)

    probs = weights/np.sum(weights)

    r = np.random.choice(valid_ecc,size=n_phosphenes,replace=True,p=probs)
    theta = np.pi *2 * np.random.rand(n_phosphenes)

    return r, theta

def init_from_cortex_full_view(params, electrode_coords = None):
    """ initialise phosphene locations in the full field of view, given parameters and a list of electrode coordinates
    :param params: dictionary with the several parameters in subdictionaries
    :param electrode_coords: tuple (x_coords, y_coords) with electrode locations on a visuotopic map. If None, default coordinates will be used
    :return: phosphene locations in polar coordinates
    """

    left_r, left_phi = init_from_cortex(params['cortex_model'], electrode_coords) #DISCUSS: phi or theta, and should we always assume mirrored cortices?
    right_r, right_phi = init_from_cortex(params['cortex_model'], electrode_coords)
    r = np.concatenate([left_r,right_r])
    phi = np.concatenate([left_phi,-1*right_phi+np.pi])

    #to simulate only right visual field
    # r = left_r
    # phi = left_phi

    return r, phi


    


