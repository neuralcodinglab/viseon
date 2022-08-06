import numpy as np
import yaml
import time
import cv2
from matplotlib import pyplot as plt
from simulator.image_processing import sobel_processor, canny_processor, sample_receptive_fields, sample_centers

def get_deg2pix_coeff(run_params):
    deg2pix = run_params['resolution'][0]/run_params['view_angle']
    # deg2pix = 1/pixels_per_degree

    print(f"displaying {run_params['view_angle']} degrees of vision in a resolution of {run_params['resolution']}")
    print(f"one degree is {deg2pix} pixels")

    return deg2pix

def calculate_dpi(params):
    w_pixels = params['display']['screen_resolution'][0]
    h_pixels = params['display']['screen_resolution'][1]
    diagonal = params['display']['screen_diagonal']
    w_inches = (diagonal ** 2 / (1 + h_pixels ** 2 / w_pixels ** 2)) ** 0.5
    dpi = round(w_pixels / w_inches)
    return dpi

def display_real_size(params, img):
    mm_per_degree = params['display']['dist_to_screen']*np.tan((2*np.pi)/360)
    view_angle = params['run']['view_angle']
    resolution = params['run']['resolution']
    aspect_ratio = resolution[0]/resolution[1]

    mm = 0.1/2.54
    fig_width = mm_per_degree*view_angle*mm
    fig_height = mm_per_degree*(view_angle/aspect_ratio)*mm
    dpi = calculate_dpi(params)
    print(f"sizes: {fig_width}, {fig_height} | dpi: {dpi}")

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    im = ax.imshow(img, cmap='gray', vmin=0, vmax=255,origin='lower')
    plt.colorbar(im)
    plt.show()

#force matplotlib to show the 'real' size
def display_image_in_actual_size(img, dpi=100):

    height, width = img.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize,dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    im = ax.imshow(img, cmap='gray', vmin=0, vmax=255,origin='lower')
    # plt.colorbar(im)
    plt.show()

def load_coords_from_yaml(path, n_coords=None):
    with open(path, 'r') as f:
        coords = yaml.load(f, Loader=yaml.SafeLoader)
        x_coords = np.array(coords['x'])
        y_coords = np.array(coords['y'])

    if n_coords:
        sample = np.random.choice(len(x_coords), n_coords)
        x_coords = x_coords[sample]
        y_coords = y_coords[sample]

    return x_coords, y_coords

def load_params(path):
    with open(path, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    return params

def webcam_demo(simulator, params, resolution=(256,256)):
    # video_capture 
    IN_VIDEO = 0 # use 0 for webcam, or string with video path"
    FRAMERATE = params['run']['fps']
    RESOLUTION = resolution

    # # device
    # gpu_nr = params['run']['gpu']
    # device = f'cuda:{int(gpu_nr)}' if gpu_nr else 'cpu'

    prev = 0
    cap = cv2.VideoCapture(IN_VIDEO)
    ret, frame = cap.read()

    while(ret):

        # Capture the video frame by frame
        ret, frame = cap.read()

        time_elapsed = time.time() - prev
        # ret, image = cap.read()
        if time_elapsed > 1./FRAMERATE:
            prev = time.time()

            # Square input
            # if frame.shape[0] != frame.shape[1]:
                # shortest_side = min(frame.shape)
                # frame = frame[frame.shape[0]//2-shortest_side//2:frame.shape[0]//2+shortest_side//2,frame.shape[1]//2-shortest_side//2:frame.shape[1]//2+shortest_side//2]

            # Create Canny edge detection mask
            frame = cv2.resize(frame, RESOLUTION)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (3,3), 0)

            if params['sampling']['filter'] == 'sobel':
                processed_img = sobel_processor(frame)
            elif params['sampling']['filter'] == 'canny':
                processed_img = canny_processor(frame,params['sampling']['T_high']//2,params['sampling']['T_high'])
            elif params['sampling']['filter'] == 'none':
                processed_img = frame
            else:
                raise ValueError(f"{params['sampling']['filter']} is not a valid filter keyword")
                
            # Generate phosphenes 
            if params['sampling']['sampling_method'] == 'receptive_fields': 
                stim_pattern = sample_receptive_fields(processed_img, simulator.sampling_mask)
            elif params['sampling']['sampling_method'] == 'center':
                stim_pattern = sample_centers(processed_img, simulator.pMap)
            else:
                raise ValueError(f"{params['sampling']['sampling_method']} is not a valid sampling method")

            phs = simulator(stim_pattern.view(1,-1))
            phs = np.squeeze(phs.cpu().numpy())*255

            # Concatenate results
            cat = np.concatenate([frame, processed_img, phs], axis=1).astype('uint8')
        
            # Display the resulting frame
            cv2.imshow('Simulator', cat)
            

        # the 'q' button is set as the quit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def process_video(simulator, params, video_path, save_path, n_frames = None, timing=False, save_history=False, device='cpu'):
    simulator.reset()
    FRAMERATE = params['run']['fps']
    #save activations for plotting. If saving the history for all phosphenes, only simulate for a few frames (uses a lot of memory and fries my laptop)
    # save_history = False
    if save_history:
        history = {key: [] for key in simulator.get_state()}
        history['stimulation'] = []
        print(history)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Unable to read file :(')
        
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #codec
    sim_resolution = simulator.resolution

    out = cv2.VideoWriter(save_path, fourcc, FRAMERATE, (sim_resolution[0]*2,sim_resolution[1]),False)
    frame_nr = 0
    if timing:
        start_time = time.time()
    while frame_nr<n_frames:
        ret, frame = cap.read()
        frame_nr+=1
        if ret:
            # to one channel, grayscale 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # get square if frame is not square
            if frame.shape[0] != frame.shape[1]:
                shortest_side = min(frame.shape)
                frame = frame[frame.shape[0]//2-shortest_side//2:frame.shape[0]//2+shortest_side//2,frame.shape[1]//2-shortest_side//2:frame.shape[1]//2+shortest_side//2]
                
            # preprocess: to resize, grayscale and blur
            frame = cv2.resize(frame, sim_resolution)
            frame = cv2.GaussianBlur(frame, (3,3), 0)

            if params['sampling']['filter'] == 'sobel':
                processed_img = sobel_processor(frame)
            elif params['sampling']['filter'] == 'canny':
                processed_img = canny_processor(frame,params['sampling']['T_high']//2,params['sampling']['T_high'])
            elif params['sampling']['filter'] == 'none':
                processed_img = frame
            else:
                raise ValueError(f"{params['sampling']['filter']} is not a valid filter keyword")
                
            # Generate phosphenes 
            if params['sampling']['sampling_method'] == 'receptive_fields': 
                stim_pattern = sample_receptive_fields(processed_img, simulator.sampling_mask)
            elif params['sampling']['sampling_method'] == 'center':
                stim_pattern = sample_centers(processed_img, simulator.pMap)
            else:
                raise ValueError(f"{params['sampling']['sampling_method']} is not a valid sampling method")

            stim_pattern = stim_pattern.view(1,-1).to(device)

            # Generate phosphenes 
            phs = simulator(stim_pattern).clamp(0,1)
            phs = phs.squeeze().cpu().numpy()*255

            if save_history:
                state = simulator.get_state()
                for key in state:
                    print(key)
                    print(state[key])
                    history[key].append(state[key])
                    print(history[key])
                history['stimulation'].append(stim_pattern.cpu().numpy())

            # Concatenate results
            # cat = np.concatenate([frame, processed_img, phs], axis=1).astype('uint8')
            cat = np.concatenate([processed_img, phs], axis=1).astype('uint8')
            
            out.write(cat)

        else:
            break
        # the 'q' button is set as the quit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if timing:
        print("--- %s seconds ---" % (time.time() - start_time))
    cap.release()
    out.release() 

    cv2.destroyAllWindows()
