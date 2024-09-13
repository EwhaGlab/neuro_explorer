# neuro_explorer_train
**neuro_explorer_train** package contains the essential python scripts for training the FRNet, A*Net, and VizNet model as detailed in in our IROS2024 paper [Neuro-Explorer: Efficient and Scalable Exploration Planning via Learned Frontier Regions](http://graphics.ewha.ac.kr/neuro_explorer/).

## Base dependencies

This package was developed in a Conda environment on Ubuntu 20.04, using the specified TensorFlow version and its dependencies. It is recommended to install these dependencies beforehand.

[Miniconda](https://docs.anaconda.com/miniconda/)
[CUDA: 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
[CUDNN: v8.9.2](https://developer.nvidia.com/rdp/cudnn-archive)
[Tensorflow 12.2](https://www.tensorflow.org/install/pip)

## Clone and install the training package

```
git clone https://github.com/han-kyung-min/neuro_explorer_train.git
conda create -n neuro_explorer_train python=3.9
conda activate neuro_explorer_train
pip install -r requirements.txt

```

## Download the dataset

The training dataset is available in [this folder](https://drive.google.com/file/d/1Z5u4hQpcBV-KLRwYiKwXX42Zox5LgwAD/view?usp=sharing)
Unzip this file and place it under your data directory. Then, make sure to have ```data_dir``` in config files points to your data directory.

## To train models
```<base_dir>/neuro_explorer_train/examples``` contains essential scripts to train the models.
For example, to retrain A*Net model, follow these steps.

1. Open the ```<base_dir>/neuro_explorer_train/configs/astarnet_params.yaml``` file.
2. Modify the necessary fields, such as ```"data_dir"```, ```"outpath"```, and ```"outmodelfile"```, based on your requirements. 
There are additional parameters available for tunning. Users are encouraged to adjust these settings, as long as they have a clear understanding of their effects.

3. Then, change directory to ```<base_dir>/neuro_explorer_train/example/astar_pot_predictor```
4. Execute the commands below.

```
conda activate neuro_explorer_train
python train_astarnet.py ../../configs/astarnet_params.yaml
```

Similarly, you can train FRNet by executing 

```
cd <base_dir>/neuro_explorer_train/example/frontier_detector
python train_fr_detector.py ../../configs/frnet_params.yaml
```
and train VizNet by
```
cd <<base_dir>/neuro_explorer_train/example/coverage_predictor
python train_viznet.py ../../configs/viznet_params.yaml
```

## Citation

Please cite our paper if you think this package is helpful for your research work

```
K.M. Han and Y.J. Kim, "Neuro-Explorer: Efficient and Scalable Exploration Planning via Learned Frontier Regions," 
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024
```

#### Feel free to send us an email if you are having a trouble with compiling this package.



