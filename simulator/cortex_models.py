import numpy as np
import matplotlib.pyplot as plt

def to_complex(x,y): 
    return x + 1j*y

def init_full_view(plot=False):
    # points in the right visual field map to the left visual cortex:
    ecc_range = np.arange(0,90,1)
    ang_range = np.linspace(-np.pi/2,np.pi/2,10)

    r, theta = np.meshgrid(ecc_range,ang_range)
    r = r.flatten()
    theta = theta.flatten()

    if plot:
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        plt.scatter(x,y)
        plt.axis('square')

    z = r*np.exp(1j*theta)
    return z

def init_mapping(mapping_model='monopole'):
    if mapping_model == 'monopole':
        mapping = lambda z, a, b, k, alpha: k*np.log(z+a) - k*np.log(a)
    elif mapping_model == 'dipole':
        mapping = lambda z, a, b, k, alpha: k*np.log((z+a)/(z+b)) - k*np.log(a/b)
    elif mapping_model == 'wedge-dipole':
        wedge_map = lambda r, theta, alpha: r*np.exp(1j*alpha*theta)
        dipole_mapping = lambda z, a, b, k: k*np.log((z+a)/(z+b)) - k*np.log(a/b)
        mapping = lambda z, a, b, k, alpha: dipole_mapping(wedge_map(np.sqrt(z.real**2+z.imag**2),np.angle(z),alpha),a,b,k)
    
    return mapping

def get_cortical_magnification(params):
    mapping_model = params['model']
    a = params['a']
    b = params['b']
    k = params['k']
    if mapping_model=='monopole':
        return lambda r: k/(r+a)
    elif mapping_model=='dipole' or mapping_model=='wedge-dipole':
        return lambda r: k*(1/(r+a)-1/(r+b))

def generate_cortical_map(mapping, z, params, plot=False):
    """_summary_

    :param mapping: function to map from visual field locations to a cortical mapping of those locations
    :param z: visual field locations in the complex plane
    :param a: shear parameter
    :param b: _description_ #TODO
    :param k: scaling factor
    :param alpha: _description_
    :param plot: whether to plot the resulting mapping defaults to False
    :return: cortical coordinates
    """
    w = mapping(z,params['a'],params['b'],params['k'],params['alpha'])

    x_cortex = np.array([coor.real for coor in w])
    y_cortex = np.array([coor.imag for coor in w])

    if plot:
        plt.scatter(x_cortex,y_cortex)
        plt.xlabel('cortical distance (mm)')
        plt.ylabel('cortical distance (mm)')
        plt.show()

    return x_cortex, y_cortex

def generate_phosphene_map(mapping, w, params):
    z = mapping(w,params['a'],params['b'],params['k'],params['alpha'])
    return z

def filter_invalid_electrodes(z,x_coords, y_coords):
    r_total = np.sqrt(z.real**2+z.imag**2)
    theta_total = np.angle(z)

    z = z[(r_total>=0)&(r_total<=90)&(theta_total>-np.pi/2)&(theta_total<np.pi/2)]
    x_coords = x_coords[(r_total>=0)&(r_total<=90)&(theta_total>-np.pi/2)&(theta_total<np.pi/2)]#(w.real>=0)&
    y_coords = y_coords[(r_total>=0)&(r_total<=90)&(theta_total>-np.pi/2)&(theta_total<np.pi/2)]#(w.real>=0)&

    print(f"initialized {len(r_total)} phosphenes, removed {len(r_total)-len(x_coords)}, {len(x_coords)} left")

    return z,x_coords,y_coords

def init_reverse_mapping(mapping_model='monopole'):
    if mapping_model == 'monopole':
        mapping = lambda w, a, b, k, alpha: a*np.exp(w/k) - a
    elif mapping_model == 'dipole':
        mapping = lambda w, a, b, k, alpha: a*b*(np.exp(w/k)-1)/(b-a*np.exp(w/k))
    elif mapping_model == 'wedge-dipole':
        wedge_map_inverse = lambda xi, alpha: np.sqrt(xi.real**2+xi.imag**2)*np.exp(1j*np.angle(xi)/alpha)
        dipole_mapping_inverse = lambda w, a, b, k: a*b*(np.exp(w/k)-1)/(b-a*np.exp(w/k))
        mapping = lambda w, a, b, k, alpha: wedge_map_inverse(dipole_mapping_inverse(w,a,b,k),alpha) 

    return mapping
    
def init_covering_electrode_grid(params, n_electrodes_x, n_electrodes_y, max_x=None, plot=False):

    z = init_full_view()
    mapping = init_mapping(params['model'])
    x_cortex, y_cortex = generate_cortical_map(mapping, z, params)

    if max_x:
        x = np.linspace(x_cortex.min(), max_x, n_electrodes_x)
    else:
        x = np.linspace(x_cortex.min(), x_cortex.max(), n_electrodes_x)

    y = np.linspace(y_cortex.min(), y_cortex.max(), n_electrodes_y)

    x_coords, y_coords = np.meshgrid(x,y)
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    if plot:
        plt.scatter(x_cortex,y_cortex,c='r')
        plt.scatter(x_coords,y_coords)
        plt.xlabel('cortical distance (mm)')
        plt.ylabel('cortical distance (mm)')
        plt.show()

    return x_coords, y_coords

def add_noise(x_coords, y_coords, noise_scale=0.5):
    rng = np.random.default_rng()

    noise = rng.normal(scale=noise_scale,size=(2,x_coords.shape[0])) #assuming a random shift in both x and y direction
    x_coords += noise[0]
    y_coords += noise[1]

    return x_coords, y_coords

def add_dropout(x_coords, y_coords, dropout_rate=0.5):
    rng = np.random.default_rng()

    active = rng.choice(np.arange(len(x_coords)),int(len(x_coords)*(1-dropout_rate)))
    x_coords = x_coords[active]
    y_coords = y_coords[active]

    return x_coords,y_coords

def get_phosphene_map_from_electrodes(params, x_coords, y_coords):
    mapping = init_reverse_mapping(params['model'])

    x_coords,y_coords = add_noise(x_coords,y_coords, params['noise_scale'])
    x_coords,y_coords = add_dropout(x_coords,y_coords, params['dropout_rate'])

    w = to_complex(x_coords,y_coords)

    z = generate_phosphene_map(mapping,w,params)

    z, x_coords, y_coords = filter_invalid_electrodes(z, x_coords, y_coords)

    r = np.sqrt(z.real**2+z.imag**2)
    theta = np.angle(z)

    return r, theta






