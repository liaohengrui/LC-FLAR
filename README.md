# [Learned Lossless Image Compression with Fast and Lightweight Autoregressive Model]


## Description
The code supports learning-based lossless image compression and serves as the official implementation of LC-FLAR,
providing encoding and decoding through ```encode.py``` and ```decode.py```. The environment can be easily set up using ```environment.yml``` with Conda.
### Code Information
```
|──  img : directory containing test images
|── utils : files for utility functions
|── custom_layers.py : implementation of masked convolution layers
|── decode.py : decodes compressed files into images
|── encode.py : encodes images into compressed format
|── environment.yml : specification of the virtual environment
|── flar_model.py and flar_model_eval.py : architecture definition of LC-FLAR
|── gaussmixturemodel.py : CDF-based probabilistic modeling
└── ckp_best.tar  model used in this study 
```
Mask convolution is implemented in ```custom_layers.py```. The checkerboard strategy is implemented in lines 71–120 of ```encode.py```. 
The for-loop in ```encode.py``` realizes the parallel autoregressive process described in Fig. 3(d-opt) of the paper. The Gaussian probability modeling is implemented in ```gaussmixturemodel.py```.
## Usage Instructions

1. Place the image in `./img/`.


2. In `encode.py`, update the image path on line 234, then run the following command to encode:
```
python encode.py
```
The encoded result will be saved in `./img/Bitstream.bin`

3. Run the following command for decoding
```
python decode.py 
```

This decodes the compressed files in `./img/rec.png`.


## Requirements
- Ubuntu 22.04
- Pytorch 2.1.0
- GPU: NVIDIA RTX 4080 Super
- Python 3.9

You can type the following command to easily build the environment.
Download 'environment.yml' and type the following command.
```
conda env create -f environment.yml
```
