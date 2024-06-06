# End-to-end optimization of prosthetic vision

## About

### This project
Simulation-based optimization of prosthetic vision through end-to-end neural nets (see [paper](https://doi.org/10.1167/jov.22.2.20)). 

Citation: 
de Ruyter van Steveninck, J., Güçlü, U., van Wezel, R., &amp; van Gerven, M. (2022). End-to-end optimization of Prosthetic Vision. Journal of Vision, 22(2), 20. https://doi.org/10.1167/jov.22.2.20 

### Usage
To reproduce the experiments:
1. clone this repository and go to the branch `e2e_paper`:
``` shell
$ git clone https://github.com/neuralcodinglab/viseon.git
$ cd viseon
$ git checkout e2e_paper
```
2. Install dependencies
``` shell
$ pip install -r requirements.txt
```
3. Run the training script. 

With the default demo configuration:
``` shell
$ python training.py 
```
or with a specific training configuration:
```
$ python training.py -csv training_configuration/<name of config>.csv
```

###  Follow-up work
For follow-up experiments with an improved biologically plausible simulation of cortical prosthetic vision (described in [this paper](https://doi.org/10.7554/eLife.85812)), see [this branch](https://github.com/neuralcodinglab/viseon/tree/dynaphos-paper) in the current repository, and [this repository](https://github.com/neuralcodinglab/dynaphos) for the source code of the new simulator.

We also refer to [this paper](https://doi.org/10.1142/S0129065722500526) by Küçükoğlu et al. on the optimization of neuroprosthetic vision via end-to-end deep reinforcement learning.


## Model Parameters
download model parameters from: https://surfdrive.surf.nl/files/index.php/s/A6COkf7HTy7Waog

 