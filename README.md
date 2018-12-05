# Device Characterisation Methods

This repository includes the code accompanying the paper *Methods for device characterisation in media services* submitted to the ACM International Conference on Interactive Experiences for Television and Online Video (TVX'19 - https://tvx.acm.org/2019/).

The repository contains all the code used to create the tables shown in the paper, as well as the dataset used for both training and testing.

The code is written in Python 3.6 and requires pandas, scikit-learn, tensorflow, keras.

The repository also includes the pyCharm project file, making it easy to train the models and test them.

## Vector Space Model

In order to train and test the vector space model, just move to the corresponding folder and run python:

```bash
cd vsmProfiler
python3 run.py
```

## Logistic Regression 

In order to train the model, move to the corresponding folder and call makeModel.py


```bash
cd logRegProfiler
python3 makeModel.py
```

To test the model once training is finished

```bash
python3 predict.py
```

## Neural Network

Similarly to what has been done for logisti regression, this is how to train the neural net:


```bash
cd neuralProfiler
python3 makeModel.py
```

To test the model once training is finished

```bash
python3 predict.py
```

### Creating the confusion matrix

Just run `jupyter notebook` and open the notebook `plotFigures/heatmaps_device_char.ipynb`. Running the notebook in your browser will show the confusion matrix and save a copy on disk

### Dataset

The folder UA contains the datasets used. Browscap.csv is the dataset available at http://browscap.org and it has been used during training. The other files represent the test dataset and were assembled by the authors. Each file represent one device class. The file userAgent5.csv, available in each model folder, is just a concatenation of the 4 files.



