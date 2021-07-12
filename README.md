# MBRNN Pytorch Implementation
This repository contains a [PyTorch](https://pytorch.org/) implementation of the MBRNN introduced in "[Lee et al. 2021]()".

# Installation
This package requires Python >= 3.7.

## Library Dependencies 
- PyTorch: refer to [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch with proper version for your local setting.
- Numpy: use the below command with pip to install Numpy (Refer [here](https://github.com/numpy/numpy) for any issues installing Numpy).
```
pip install numpy
```

# How to Run The Model

## Data Preparation
As described in the paper, the photometric data for both training and inference comes from the PS1 DR1 catalog. We extracted the data from [the Vizier 
catalog II/349](https://vizier.u-strasbg.fr/viz-bin/VizieR?-source=II/349) (see [ReadMe](https://cdsarc.unistra.fr/viz-bin/ReadMe/II/349?format=html&tex=true)). The input features adopted in the paper include colors, their uncertainties, and E(B-V) which comes from the Planck Galactic dust measurement (Planck Collaboration 2013). The following is the code snippet which produces the input data except for the E(B-V) from the given CSV file which are downloaded from the Vizier catalog. Importantly, our training samples do not include objects that do not satisfy the mask condition (i.e., row['Qual'] & use_mask == 4). The tool like [dustmaps](https://dustmaps.readthedocs.io/en/latest/modules.html#module-dustmaps.planck) can be useful to estimate E(B-V) values for the given coordinate.
```
data = numpy.genfromtxt(infn, delimiter=",", skip_header=1, \
dtype=[('RAJ2000','S16'), ('DEJ2000','S16'), ('objID', 'i8'), \
('Qual', 'i'), ('gmag', 'f8'), ('e_gmag', 'f8'), ('gKmag', 'f8'), \
('e_gKmag', 'f8'), ('rmag', 'f8'), ('e_rmag','f8'), \
('rKmag', 'f8'), ('e_rKmag', 'f8'), ('imag', 'f8'), \
('e_imag', 'f8'), ('iKmag', 'f8'), ('e_iKmag', 'f8'), \
('zmag', 'f8'), ('e_zmag', 'f8'), ('zKmag', 'f8'), \
('e_zKmag', 'f8'), ('ymag', 'f8'), ('e_ymag', 'f8'), \
('yKmag', 'f8'), ('e_yKmag', 'f8')])

use_mask = 0b100

for row_ind in range(0, num_rows):
    row = data[row_ind]
    objid = row['objID']
    RA = row['RAJ2000']
    DEC = row['DEJ2000']
    mag_list = []
    color_list = []
    use_gmag = row['gmag']
    use_e_gmag = row['e_gmag']
    use_gKmag = row['gKmag']
    use_e_gKmag = row['e_gKmag']
    mag_list.append(use_gmag)
    mag_list.append(use_e_gmag)
    mag_list.append(use_gKmag)
    mag_list.append(use_e_gKmag)
    use_rmag = row['rmag']
    use_e_rmag = row['e_rmag']
    use_rKmag = row['rKmag']
    use_e_rKmag = row['e_rKmag']
    mag_list.append(use_rmag)
    mag_list.append(use_e_rmag)
    mag_list.append(use_rKmag)
    mag_list.append(use_e_rKmag)
    use_gr = use_gmag - use_rmag
    use_grK = use_gKmag - use_rKmag
    use_e_gr = math.sqrt(use_e_gmag**2 + use_e_rmag**2)
    use_e_grK = math.sqrt(use_e_gKmag**2 + use_e_rKmag**2)
    color_list.append(use_gr)
    color_list.append(use_e_gr)
    color_list.append(use_grK)
    color_list.append(use_e_grK)
    use_imag = row['imag']
    use_e_imag = row['e_imag']
    use_iKmag = row['iKmag']
    use_e_iKmag = row['e_iKmag']
    mag_list.append(use_imag)
    mag_list.append(use_e_imag)
    mag_list.append(use_iKmag)
    mag_list.append(use_e_iKmag)
    use_ri = use_rmag - use_imag
    use_riK = use_rKmag - use_iKmag
    use_e_ri = math.sqrt(use_e_rmag**2 + use_e_imag**2)
    use_e_riK = math.sqrt(use_e_rKmag**2 + use_e_iKmag**2)
    color_list.append(use_ri)
    color_list.append(use_e_ri)
    color_list.append(use_riK)
    color_list.append(use_e_riK)
    use_zmag = row['zmag']
    use_e_zmag = row['e_zmag']
    use_zKmag = row['zKmag']
    use_e_zKmag = row['e_zKmag']
    mag_list.append(use_zmag)
    mag_list.append(use_e_zmag)
    mag_list.append(use_zKmag)
    mag_list.append(use_e_zKmag)
    use_iz = use_imag - use_zmag
    use_izK = use_iKmag - use_zKmag
    use_e_iz = math.sqrt(use_e_imag**2 + use_e_zmag**2)
    use_e_izK = math.sqrt(use_e_iKmag**2 + use_e_zKmag**2)
    color_list.append(use_iz)
    color_list.append(use_e_iz)
    color_list.append(use_izK)
    color_list.append(use_e_izK)
    use_ymag = row['ymag']
    use_e_ymag = row['e_ymag']
    use_yKmag = row['yKmag']
    use_e_yKmag = row['e_yKmag']
    mag_list.append(use_ymag)
    mag_list.append(use_e_ymag)
    mag_list.append(use_yKmag)
    mag_list.append(use_e_yKmag)
    use_zy = use_zmag - use_ymag
    use_zyK = use_zKmag - use_yKmag
    use_e_zy = math.sqrt(use_e_zmag**2 + use_e_ymag**2)
    use_e_zyK = math.sqrt(use_e_zKmag**2 + use_e_yKmag**2)
    color_list.append(use_zy)
    color_list.append(use_e_zy)
    color_list.append(use_zyK)
    color_list.append(use_e_zyK)
    out_str = "%d %s %s " % (objid, RA.decode("utf-8"), DEC.decode("utf-8"))
    for oneitem in color_list:
        if math.isnan(oneitem):
            out_str = out_str + "NA "
        else:
            out_str = out_str + "%.8f " % (oneitem)
    outfd.write(out_str.strip() + "\n")
```
The code provided here loads the data in the format of NumPy file saved in Pickle as coded in PS1.py. If the X variable in the following 
code snippet has values of 0.40379906 0.07061161 0.25110054 0.15079354 0.11000061 0.06245294 0.18779945 0.12533208 -0.02160072 0.20311637 -0.10390091 0.10137894 0.40800095 0.26574471 0.76920128 0.25393150 0.01558769 for data[4:21] which corresponds to the input features including the E(B-V) as the last value, you can produce the input data in the right format. Here, the example data has the redshift 1.11199999 with its uncertainty 0.001 which is not 
used in the training step.
```
import numpy as np

use_minX = np.array([-7.9118004., 0., -9.394201, 0., -3.9944992, 0., -4.2058992, 0., -2.851099, 0., -6.1702003, 0., -4.963501, 0., -6.359, 0., -5.72029], dtype=np.float32) # used in the paper
use_maxX = np.array([5.9019985, 0.5281896, 5.8084, 0.46895373, 2.9131012, 0.52544963, 3.900301, 0.45075417, 3.905901, 0.5185917, 4.9472, 0.4172655, 6.077201, 0.5891852, 7.9728994, 0.46186885, 3.2700593], dtype=np.float32) # used in the paper

# Preparation of X and Y
X = np.array([0.40379906, 0.07061161, 0.25110054, 0.15079354, 0.11000061, 0.06245294, 0.18779945, 0.12533208, -0.02160072, 0.20311637, -0.10390091, 0.10137894, 0.40800095, 0.26574471, 0.76920128, 0.25393150, 0.01558769], dtype=np.float32)
X[-1] = np.log(X[-1]) # E(B-V)
zspec = np.array([1.11199999], dtype=np.float32)
zerr = np.array([0.001], dtype=np.float32) # not really used
labels = np.zeros(len(zspec))
Y = np.vstack((labels, zspec, zerr)).astype(np.float32).T

normedX = (X-use_minX)/(use_maxX-use_minX)*2.-1.
normed = np.hstack((Y, normedX.T.astype(np.float32)))

np.save("example.npy", normed)
```

The code snippet is for the data min-max normalization in a feature-wise manner. Note that minima and maxima values of input features referred to as 'use_minX' and 'use_maxX' were estimated from the loaded PS1 data. When training the model with different data, the values should be separately estimated.

For the errorless implementation of the code, train, validation, and test samples should be stored in the files named 'train.npy', 'val.npy', 'test.npy' under the directory 'PS1_data', respectively. One may do so by modifying "example.npy" in the last line of the code.

## Model Training
Although our deploy version code includes the pre-trained network, one can train a new model from scratch using the below command.
```
python main.py
```

## Model Inference
One may use the below command for the inference of the trained model.

```
python main.py --infer
```

The process will dump an array shaped [*nsamp*, *nbin*+1] into the folder '*Outputs*' with Numpy format, where *nsamp* and *nbin* are the number of samples and bins, respectively. The first *nbin* columns of the array are model output probabilities, and the last column is the photometric redshift.

## Option Change
We deploy the model with the best-performing configuration described in our paper, but one can adjust the model structure and other settings by modifying the options of the *config_file/config.cfg* file.
