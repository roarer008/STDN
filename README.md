# STDN

A pytorch implementation for the paper: **[Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting](https://arxiv.org/abs/2502.12213)**

Run the model in JiNan or PeMS:
first:
```
python prepareData.py
```
second:
```
python train.py
```
The default setting is in conf/PEMSD4_1dim_12.conf

JiNan dataset will be available soon.
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

You can download the data from:

Google Drive: https://drive.google.com/drive/folders/1oo-eO41kbQS8aDyFWER66DdPT2k3k8_m?usp=sharing
