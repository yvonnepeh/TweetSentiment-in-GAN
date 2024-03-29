# TweetSentiment-in-GAN

## Description

The goal of the capstone is to evaluate the success of the usage of Sentiment Analysis through VADER as a model feature in predicting the prices of 3 big stocks in the US market, namely Apple, Google and Microsoft, using GAN model. Two types of models: one with and one without sentiment analysis feature are built for each stock and evaluated using RMSE and MAPE. 


## Getting Started
### Datasets
* Stock Prices: Yahoo Finance (AAPL.csv, GOOG.csv, MSFT.csv) 
* Twitter tweets: https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020/data (Tweet.csv, Company_Tweet.csv) | 
  This has to be downloaded from the website as datasets are too large to upload onto Github
  
### Dependencies
All packages and its versions are listed in requirements.txt.
```
$ pip install -r requirements.txt
```

### Executing program

Main Code: 
* Capstone_v1.ipynb
  
Python Modules:
* functions.py
* sentiment_analysis.py
* model.py

Setup/Notes:
* Place data folder with input datasets and python modules in the same directory as Capstone_v1.ipynb
* Create folder for each model before training model in data/models_gan 
* To run each model, restart kernel, load Library, run Build Model Section before training specific model
  
## Authors

Yvonne Peh \
Jian Hui
