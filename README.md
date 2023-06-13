# MFA-Conformer

This repository contains the training code accompanying the paper "MFA-Conformer: Multi-scale Feature Aggregation Conformer for Automatic Speaker Verification", which is submitted to Interspeech 2022.

<p align="center"><img width="95%" src="docs/mfa_conformer.png" /></p>

The architecture of the MFA-Conformer is inspired by recent state-of-the-art models in speech recognition and speaker verification. Firstly, we introduce a convolution subsampling layer to decrease the computational cost of the model. Secondly, we adopt Conformer blocks which combine Transformers and convolution neural networks (CNNs) to capture global and local features effectively. Finally, the output feature maps from all Conformer blocks are concatenated to aggregate multi-scale representations before final pooling. The best system obtains 0.64%, 1.29% and 1.63% EER on VoxCeleb1-O, SITW.Dev, and SITW.Eval set, respectively. 

## Setting up python environment
```bash
pip install -r requirements.txt
```

## Data Preparation

* [VoxCeleb 1&2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
* [SITW](http://www.speech.sri.com/projects/sitw/)

```bash
# format Voxceleb test trial list
rm -rf data; mkdir data
wget -P data/ https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
python3 scripts/format_trials.py \
            --voxceleb1_root $voxceleb1_dir \
            --src_trials_path data/veri_test.txt \
            --dst_trials_path data/vox1_test.txt

# make csv for voxceleb1&2 dev audio (train_dir)
python3 scripts/build_datalist.py \
        --extension wav \
        --dataset_dir data/$train_dir \
        --data_list_path data/train.csv
```

## Model Training

```bash
bash start.sh
```

## Inference
To run inference with the pretrained model, run:
```bash
python3 inference.py --checkpoint_path=path_to/pretrained_model.ckpt --audio_path_1=path_to/audio_1.wav --audio_path_2=path_to/audio_2.wav 
```
You can download pretrained model from [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/truongdu001_e_ntu_edu_sg/EfeIgwS89qpGpp8oZFyDuHcBQh2w0NwH2cABV6uKvMwLdA?e=kNNe2E).

## Citation
Most of the code in this repo is from: https://github.com/zyzisyz/mfa_conformer

The original paper:
```
@article{zhang2022mfa,
  title={MFA-Conformer: Multi-scale Feature Aggregation Conformer for Automatic Speaker Verification},
  author={Zhang, Yang and Lv, Zhiqiang and Wu, Haibin and Zhang, Shanshan and Hu, Pengfei and Wu, Zhiyong and Lee, Hung-yi and Meng, Helen},
  journal={arXiv preprint arXiv:2203.15249},
  year={2022}
}
```
