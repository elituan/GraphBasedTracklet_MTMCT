# Multi-Vehicle Multi-Camera Tracking with Graph-Based Tracklet Features
[![python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![pyg](https://img.shields.io/badge/PyG-3C2179.svg?style=for-the-badge&logo=PyG&logoColor=white)](https://pyg.org/)


# Overview
This repository is an official implementation of the "Multi-Vehicle Multi-Camera Tracking with Graph-Based Tracklet Features" paper being published in the IEEE Transactions on Multimedia journal.

## Requirements
Python 3.8 or later with all ```requirements.txt``` dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```

## Data Preparation
If you want to reproduce our results on AI City Challengef,
please download the datasets from: (https://www.aicitychallenge.org/)
and put it under the folder datasets.
Make sure the data structure is like:

> **[GraphBasedTracklet_MTMCT Google Drive](https://drive.google.com/drive/folders/)**
>   * datasets
>     * [AIC21_Track3_MTMC_Tracking](https://www.aicitychallenge.org/2021-data-and-evaluation/)
>       * unzip AIC21_Track3_MTMC_Tracking.zip
>     * detect_provided (Including detection and corresponding Re-ID features)
>   * detector
>     * yolov5
>       * [yolov5x.pt](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5x.pt) (Pre-trained yolov5x model on COCO)
>   * reid
>     * reid_model (Pre-trained reid model on Track 2)
>       * resnet101_ibn_a_2.pth
>       * resnet101_ibn_a_3.pth
>       * resnext101_ibn_a_2.pth

## Reproduce frome detect_provided 
If you just want reproduce our results, you can directly download ```detect_provided```:
```
cd GraphBasedTracklet_MTMCT
mkdir datasets
cd datasets
```
Then put ```detect_provided``` folder under this folder and modify yml ```config/aic_graphbase.yml```:
```
CHALLENGE_DATA_DIR: '/home/xxx/GraphBasedTracklet_MTMCT/datasets/AIC21_Track3_MTMC_Tracking/'
DET_SOURCE_DIR: '/home/xxx/GraphBasedTracklet_MTMCT/datasets/detection/images/test/S06/'
DATA_DIR: '/home/xxx/GraphBasedTracklet_MTMCT/datasets/detect_provided'
REID_SIZE_TEST: [384, 384]
ROI_DIR: '/home/xxx/GraphBasedTracklet_MTMCT/datasets/AIC21_Track3_MTMC_Tracking/test/S06/'
CID_BIAS_DIR: '/home/xxx/GraphBasedTracklet_MTMCT/datasets/AIC21_Track3_MTMC_Tracking/cam_timestamp/'
USE_RERANK: True
USE_FF: True
SCORE_THR: 0.1
MCMT_OUTPUT_TXT: 'track3.txt'
```
Then run:
```
bash ./run_graphbase_reproduce.sh
```

The final results will locate at path ```./reid/reid-matching/tools/track3.txt```



## Citation
Nguyen, T. T., Nguyen, H. H., Sartipi, M., & Fisichella, M. (2023). Multi-Vehicle Multi-Camera Tracking with Graph-Based Tracklet Features. IEEE Transactions on Multimedia. [Preprint](https://hoanghnguyen.com/assets/pdf/nguyen2023multi.pdf)

```
@article{nguyen2023multi,
  title={Multi-Vehicle Multi-Camera Tracking With Graph Based Tracklet Features},
  author={Nguyen, Tuan T. and Nguyen, Hoang H. and Sartipi, Mina and Fisichella, Marco},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  volume={},
  number={},
  publisher={IEEE},
  pages={1-13},
  doi={10.1109/TMM.2023.3274369}
}
```
