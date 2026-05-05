
Zenodo DOI: https://doi.org/10.5281/zenodo.20025365
# NLOS Direction Detection

Raw code and representative processed dataset for the study:

**Early Fusion of Laser and Acoustic Features for Human Orientation Detection in Non-Line-of-Sight Environments**


# Early Fusion of Laser and Acoustic Features for Human Orientation Detection

This repository contains the MATLAB code and processed dataset associated with the manuscript:

**Early Fusion of Laser and Acoustic Features for Human Orientation Detection in Non-Line-of-Sight Environments**

The repository was created to support the reproducibility and transparency of the study.

## Study Overview

This study investigates human orientation detection in non-line-of-sight (NLOS) environments using early fusion of laser and acoustic chirp signal features. The experimental data were collected in a controlled NLOS laboratory setup. Laser and acoustic signals were processed separately, features were extracted from both modalities, and the resulting laser-acoustic feature vectors were combined using an early fusion strategy.

The classification task includes four human orientation classes:

- Behind
- Front
- Left
- Right

The final fused dataset was used to train, validate, and test machine learning/artificial intelligence models and the proposed LAO-Net model.

## Repository Structure
Zenodo DOI: https://doi.org/10.5281/zenodo.20025365
```text
dataset/
  processed/
    laser_acoustic_fused_dataset.csv

  splits/
    train80_test20.csv

code/
  lazer_windows_create36_preproces.m
  acoustic_import_8_avgchannel.m
  laser_acoustic_dataset_create.m
  laser_acoustic_data_fusion.m
  acoustic_channel_graph.m
  evaluation_pca.m
  evaluation_corelation_matrix.m

results/
  figures/
  tables/
