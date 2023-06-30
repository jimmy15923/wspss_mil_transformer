# An MIL-Derived Transformer for Weakly SupervisedPoint Cloud Segmentation

This repository is the implementation of **An MIL-Derived Transformer for Weakly SupervisedPoint Cloud Segmentation**.

## Requirements
- Install MinkowskiEngine by using [Conda](https://github.com/NVIDIA/MinkowskiEngine#anaconda)

## Data Preparation

- Download the dataset from the official ScanNet and S3DIS websites.

## Training

- Start training:
```sh tool/train.sh EXP_NAME /PATH/TO/CONFIG NUMBER_OF_THREADS```

- Resume: 
```sh tool/resume.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS```

NUMBER_OF_THREADS is the threads to use per process (gpu), so optimally, it should be **Total_threads / gpu_number_used**


## Citation
If you find our work useful in your research, please consider citing our paper:
```
@inproceedings{yang2022mil,
  title={An mil-derived transformer for weakly supervised point cloud segmentation},
  author={Yang, Cheng-Kun and Wu, Ji-Jia and Chen, Kai-Syun and Chuang, Yung-Yu and Lin, Yen-Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11830--11839},
  year={2022}
}
```
