# MBRNN Pytorch Implementation
This repository contains a [PyTorch](https://pytorch.org/) implementation of the MBRNN introduced in "[Lee et al. 2021]()". Note that only the single MBRNN model is embodied in the code, excluding the execution of ensemble learning.

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
The code provided here loads the data in the format of NumPy file saved in Pickle as coded in PS1.py. The produced CSV file can be 
changed to the Numpy format with the right normalization step. See convert_csv_to_npy.py and example_inference_data.csv as examples.

The code convert_csv_to_npy.py is for the data min-max normalization. Note that minima and maxima values of input features referred to as 'use_minX' and 'use_maxX' were estimated from the loaded PS1 data. When training the model with different data, the values should be separately estimated.

For the errorless implementation of the code, train, validation, and test samples should be stored in the files named 'train.npy', 'val.npy', 'test.npy' under the directory 'PS1_data', respectively. For inference, the file 'infer.npy' stored in the 'PS1_data' is used as input.

## Model Training
Although our deploy version code includes the pre-trained network, one can train a new model from scratch using the below command.
```
python main.py
```

## Model Testing
One may use the below command for the test of the trained model.

```
python main.py --test
```

The process will dump an array shaped [*nsamp*, *nbin*+1] into the folder '*Outputs*' with Numpy format, where *nsamp* and *nbin* are the number of samples and bins, respectively. The first *nbin* columns of the array are model output probabilities, and the last column is the photometric redshift.

## Model Inference
People can conduct inference for the given data with the provided trained model.

```
python main.py --infer
```

The process will dump an array shaped [*nsamp*, *nbin*+3] into the folder '*Outputs*' with Numpy format. The first *nbin* columns of the array are model output probabilities, and each of the following columns is the average photometric redshift, the mode redshift, and the standard deviation of the model output probability distribution.

## Option Change
We deploy the model with the best-performing configuration described in our paper, but one can adjust the model structure and other settings by modifying the options of the *config_file/config.cfg* file.
