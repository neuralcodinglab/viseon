# End-to-end optimization of prosthetic vision
## Branch for the end-to-end experiments with biologically plausible phosphene simulation

### Download model parameters
https://surfdrive.surf.nl/files/index.php/s/vMG4UuJHo0njlWH 

## Publication
Our preprint can be cited as:

van der Grinten, M., de Ruyter van Steveninck, J., Lozano, A., Pijnacker, L., Rückauer, B., Roelfsema, P., van Gerven, M., van Wezel, R., Güçlü, U., &amp; Güçlütürk, Y. (2022). Biologically plausible phosphene simulation for the differentiable optimization of visual cortical prostheses. https://doi.org/10.1101/2022.12.23.521749 


### Previous publications
- (de Ruyter van Steveninck et al., 2022) [End-to-end optimization of prosthetic vision](https://doi.org/10.1167/jov.22.2.20)
- (Küçükoğlu et al., 2022) [Optimization of Neuroprosthetic Vision via End-to-end Deep Reinforcement Learning](http://dx.doi.org/10.1142/S0129065722500526)


## Usage

1. Download the datasets: 
    - [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
    - [Moving MNIST video dataset](https://www.cs.toronto.edu/~nitish/unsupervised_video/)
    - Or use a different dataset (modify *local_datasets.py* accordingly).
2. Install the dynaphos simulator package for biologically plausible simulation of cortical prosthetic vision (see [the dynaphos repository](https://github.com/neuralcodinglab/dynaphos))
    - either install via pip:
    
          pip install dynaphos

    - or install the bleeding edge version:

          git clone https://github.com/neuralcodinglab/dynaphos dynaphos-core
          cd dynaphos-core
          python setup.py install

3.  Download the simulator configuration (yaml file with the onfiguration parameters, and pickle file with the phosphene maps) here : https://surfdrive.surf.nl/files/index.php/s/vMG4UuJHo0njlWH
4.  (optional) Adjust or add training pipelines in *init_training.py* according to needs.
5.  Adjust or add yaml file with training configuration (in the ./_config directory).
    - Make sure to correctly set the *save_path*, *data_directory*, the simulator  *base_config* and *phosphene_map* with the right paths. 
    - Set *load_preprocessed* to *False* if no access to preprocessed dataset.
6. Initialize training.
    - For example, run:
    
          python training.py -c .\_config\naturalistic_unconstrained.yaml

    - Monitor training using tensorboard:
    
          tensorboard --logdir [your output path]\tensorboard
          
