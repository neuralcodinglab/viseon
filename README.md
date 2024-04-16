# End-to-end optimization of prosthetic vision
## Branch for the end-to-end experiments with biologically plausible phosphene simulation

### Trained  model parameters can be found here:
https://surfdrive.surf.nl/files/index.php/s/vMG4UuJHo0njlWH 

## Publication
Our paper can be cited as:

van der Grinten, M., de Ruyter van Steveninck, J., Lozano, A., Pijnacker, L., Rückauer, B., Roelfsema, P., van Gerven, M., van Wezel, R., Güçlü, U., Güçlütürk, Y. (2024) Towards biologically plausible phosphene simulation for the differentiable optimization of visual cortical prostheses eLife 13:e85812. https://doi.org/10.7554/eLife.85812



### Previous publications
- (de Ruyter van Steveninck et al., 2022) [End-to-end optimization of prosthetic vision](https://doi.org/10.1167/jov.22.2.20) (see e2e_paper branch).
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

3.  (optional) Adjust or add training pipelines in *init_training.py* according to needs.
4.  (optional) Adjust or add yaml file with training configuration (in the ./_config directory).
    - Make sure to correctly set the *save_path*, *data_directory*, the simulator  *base_config* and *phosphene_map* with the right paths. 
    - Set *load_preprocessed* to *False* if no access to preprocessed dataset.
5. Initialize training.
    - For example, run:
    
          python training.py -c .\_config\exp2\naturalistic_unconstrained.yaml

    - Monitor training using tensorboard:
    
          tensorboard --logdir [your output path]\tensorboard
          
