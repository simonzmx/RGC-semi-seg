# RGC-Semi-Seg

This is the repository for our paper [Reducing Manual Labeling Requirements and Improved Retinal Ganglion Cell Identification in 3D AO-OCT Volumes Using Semi-supervised Learning](https://opg.optica.org/boe/fulltext.cfm?uri=boe-15-8-4540&id=553141).

Please cite the papers listed at the end of this document if you use any component of this repository.

## Installation
Install with the conda environment.yml file:

```
conda env create -f environment.yml
conda activate RGC-Semi-Seg
```

## Dataset
Download the WeakGCSeg [[ref](https://opg.optica.org/optica/fulltext.cfm?uri=optica-8-5-642&id=450700)] dataset from https://people.duke.edu/~sf59/Soltanian_Optica_2021.htm.

Put the data into the folder "\data". Then run the following command to organize the data files.

```
python tools/reorganize_data.py
```

The folder should now have the following structure:

```
.
├── ...
├── data
│   ├── IU
│   ├── FDA
│   ├── all                # Organized Folder
│   ├── ├── images
│   ├── ├── labels
│   ├── └── 2nd grader
│   ├── gc_layers_all.json
│   └── ...
└── ...
```

## Train

To train the model, simply run:

```
python main.py --data_cfg ./cfg/data/dataset_iu.yaml --model_cfg ./cfg/model/hyp.semi_seg_cps_iu.yaml --batch_size 8 --resume_training
```

The configuration files used here are for training with the RGC-CPS method using the IU dataset with the 1/7 labeling scenario. Other configurations for different training methods demonstrated in the paper can be found in "/cfg/model". Configurations for different data/label settings can be found in "/cfg/data".

## Evaluate

The training pipeline will automatically perform an evaluation at the end (if "--no_test" not set). However, if one would like to evaluate a trained model only, simply run:
```
python main.py --data_cfg ./cfg/data/dataset_iu.yaml --model_cfg ./cfg/model/hyp.semi_seg_cps_iu.yaml --no_train
```

To evaluate the inter-grader variability, please use the script "tools/measure_2nd_grader.py". Only a few arguments (e.g. dataset name, targeted retinal locations) need to be identified.

## Citation
Please cite the following papers if you use any component of this repository.

```
Mengxi Zhou, Yue Zhang, Amin Karimi Monsefi, Stacey S. Choi, Nathan Doble, Srinivasan Parthasarathy, and Rajiv Ramnath, "Reducing manual labeling requirements and improved retinal ganglion cell identification in 3D AO-OCT volumes using semi-supervised learning," Biomed. Opt. Express 15, 4540-4556 (2024)

Somayyeh Soltanian-Zadeh, Kazuhiro Kurokawa, Zhuolin Liu, Furu Zhang, Osamah Saeedi, Daniel X. Hammer, Donald T. Miller, and Sina Farsiu, "Weakly supervised individual ganglion cell segmentation from adaptive optics OCT images for glaucomatous damage assessment," Optica 8, 642-651 (2021)

```
