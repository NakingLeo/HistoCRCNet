# HistoCRCNet

This project implements a neural network for image classification and regression risk score prediction.
## Installation

To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Training
To train the image classification model, run:
```
python src/train/train_classification.py
```
To train the risk score regression model, run:
```
python src/train/train_regression.py
```
## Prediction
To classify images, run:
```
python src/utils/predict_classification.py
```
To predict risk scores, run:
```
python src/utils/predict_regression.py
```
