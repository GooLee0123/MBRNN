# MBRNN Pytorch Implementation

# Installation
This package requires Python >= 3.7.

## Prerequisites 
- PyTorch: Refer to [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch with proper version for your local setting.
- Numpy: pip install numpy (Refer [here](https://github.com/numpy/numpy) for any issues installing Numpy).

# Run Model

## Data Preparation
```
TBP
```

## Training
Although our deploy version code includes the pre-trained network, one can train a new model from scratch using below command.
```
python main.py --train True
```

## Test
Since the default setting for the train option is *False*, one may use the below command for the test of the model. The outputs are classification probabilities and photometric redshifts for individual samples. The outputs will be dumped in the folder 'Outputs' with *npy* format.
```
python main.py
```

## Option Change
We deploy the model with the best-performing configuration in our paper, but one can adjust the model structure and other setting by modifying the options of the *config_file/config.cfg* file.
