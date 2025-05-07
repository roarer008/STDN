# STDN

A pytorch implementation for the paper: **[Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting](https://arxiv.org/abs/2502.12213)**

## Run the model in JiNan or PeMS:

first:
```
python prepareData.py
```
second:
```
python train.py
```

The default setting is in conf/JiNan_1dim_12.conf
 
## Download the data from:

Google Drive: https://drive.google.com/drive/folders/1oo-eO41kbQS8aDyFWER66DdPT2k3k8_m?usp=sharing

and **[SSTBAN](https://github.com/guoshnBJTU/SSTBAN)**

# Environment
```
python 3.9.19
torch 2.3.0
numpy 1.26.3
```
or
```
python 3.10.14
torch 2.4.1
numpy 1.26.4
```
# JiNan dataset

For the Jinan dataset, we selected 406 intersection nodes in Jinan, China. At the same time, for safety reasons, we provided the relative longitude and latitude of the nodes in the 'JiNan of lalo.csv'.

# Citation

If you find our work is helpful, please cite as:

```
@inproceedings{cao2025spatiotemporal,
  title={Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting},
  author={Cao, Lingxiao and Wang, Bin and Jiang, Guiyuan and Yu, Yanwei and Dong, Junyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={11},
  pages={11463--11471},
  year={2025}
}
```
