# Dynamic Structure Learning through Graph Neural Network for Forecasting Soil Moisture in Precision Agriculture

This is a Pytorch implementation of Spatio-temporal graph neural networks to estimate soil moisture

## Requirements
- matplotlib=3.3.4=pypi_0
- networkx=2.4=py_1
- numpy=1.18.5=py38h1da2735_0
- scikit-learn=0.23.1=py38h603561c_0
- scipy=1.5.0=py38hbab996c_0
- sklearn=0.0=pypi_0
- statsmodels=0.11.1=py38haf1e3a3_0
- torchvision=0.7.0=py38_cpu

Dependency can be installed using the following command:

```
pip install -r requirements.txt
```

## Model Training & Testing

Hyperparamters can be sent through command line or can be changed in the train files itself. Exact values to be used is given in the appendix of the [paper](https://arxiv.org/abs/2012.03506).

- DGLR (Shared)
```
#USA
python train.py --dataset USA --nofstations 68 --model DGLR_SHARE

#Spain
python train.py --dataset Spain --nofstations 20 --model DGLR_SHARE

#Alabama
python train.py --dataset Alabama --nofstations 8 --model DGLR_SHARE

#Mississippi
python train.py --dataset Mississippi --nofstations 5 --model DGLR_SHARE
```
- DGLR (w/o SL)
```
#USA
python train.py --dataset USA --nofstations 68 --model DGLR

#Spain
python train.py --dataset Spain --nofstations 20 --model DGLR

#Alabama
python train.py --dataset Alabama --nofstations 8 --model DGLR

#Mississippi
python train.py --dataset Mississippi --nofstations 5 --model DGLR
```
- DGLR (w/o Sm)
```
#USA
python train_graphrefine.py --dataset USA --nofstations 68

#Spain
python train_graphrefine.py --dataset Spain --nofstations 20

#Alabama
python train_graphrefine.py --dataset Alabama --nofstations 8

#Mississippi
python train_graphrefine.py --dataset Mississippi --nofstations 5
```
- DGLR (our model)
```
#USA
python train_graphrefine_smoothness.py --dataset USA --nofstations 68

#Spain
python train_graphrefine_smoothness.py --dataset Spain --nofstations 20

#Alabama
python train_graphrefine_smoothness.py --dataset Alabama --nofstations 8

#Mississippi
python train_graphrefine_smoothness.py --dataset Mississippi --nofstations 5
```

## Training Details
We have run all the experiments on a CPU with 2.6GHz 6-core Intel Core i7. We conduct  extensive hyperparameter tuning for all the baseline algorithms and report the best results obtained. Detailed hyperparameter set up is given in the appendix of the [paper](https://arxiv.org/abs/2012.03506).
