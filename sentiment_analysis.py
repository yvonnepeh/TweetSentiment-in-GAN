'''
This script contains python functions utilised for sentiment analysis in main code
v1.0
'''
import pandas as pd
import numpy as np

def process_sa_data(tweet_old, coy_tweet):
    '''
    This function merges the dataset consisting the tweets with the dataset consisting the stock ticker
    using the primary key of tweet_id
    param tweet_old: takes in tweet dataset
    param coy_tweet: takes in stock ticker dataset
    returns:
        tweets: merged tweet and stock ticker dataset for sentiment analysis
    '''
    print('There are %s tweet in total.' %(len(tweet_old)))
    print('Data Read: ')
    display(tweet_old.head(2))
    display(coy_tweet.head(2))
    tweet_old['date'] = pd.to_datetime(tweet_old['post_date'], unit='s').dt.date
    tweet_old.date=pd.to_datetime( tweet_old.date,errors='coerce')
    tweets=tweet_old.merge(coy_tweet,how='left',on='tweet_id')
    tweets['ticker_symbol'] = np.where(tweets['ticker_symbol'] =='GOOGL', 'GOOG', tweets['ticker_symbol'])
    print('After merging: ')
    display(tweets.head(2))
    return tweets

def get_sentiment(sia, tweets,ticker,start='2015-01-01',end='2019-12-31'):
    '''
    This function takes in Vader with augmented vocabulary catered to financial market context
    and apply the sentiment analysis on the stock tweets

    param sia: Vader SentimentIntensityAnalyzer with augmented vocabulary
    param tweets: dataset containing the tweets
    param ticker: stock ticker
    param start: starting period for dataset; default is 1 Jan 2015
    param end: ending period for dataset; default is 31 Dec 2019
    returns:
        df: dataset consisting tweets, sentiment analysis compound scores and dates
    '''
    df=tweets.loc[((tweets.ticker_symbol==ticker)&(tweets.date>=start)&(tweets.date<=end))]
    # apply the SentimentIntensityAnalyzer
    df.loc[:,('score')]=df.loc[:,'body'].apply(lambda x: sia.polarity_scores(x)['compound'])
    # create label
    df.loc[:,('label')]=pd.cut(np.array(df.loc[:,'score']),bins=[-1, -0.66, 0.32, 1],right=True ,labels=["bad", "neutral", "good"])

    df=df.loc[:,["date","score","label","tweet_id","body"]]
    df.to_pickle(f'sentiment_{ticker}.pkl')
    return df

def preprocess(df, score_col):
    '''
    This function preprocess dataset consisting all sentiment analysis compound scores
    and get the average compound scores for each day
    param df: dataset consisting of sentiment analysis compound scores
    param score_col: the column name for daliy compound score to be created
    returns:
        df_daily: dataset with mean/average daily compound scores
    '''
    df['date'] = pd.to_datetime(df['date'])
    # group by date and get average sentiment score per day
    daily = df.groupby(df['date'].dt.date)['score'].mean()
    df_daily = pd.DataFrame({'Date': daily.index, score_col: daily.values})
    df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.date
    df_daily.Date=pd.to_datetime( df_daily.Date,errors='coerce')
    return df_daily

def get_merge_sa(stock_name, stock, stock_col, df):
    '''
    This function takes in the dataset with historical stock prices and technical indicators
    and merge it with the dataset consisting of daliy average compound scores to create
    the input feature set for modelling purpose
    param stock_name: stock name
    param stock: dataset consisting of dataset with sentiment analysis compound scores
    param stock_col: the column name for daily compound score to be created
    param df: dataset consisting historical stock prices and technical indicators
    returns:
        final: dataset that consists of all the features required for modelling purpose
    '''
    temp = preprocess(stock, stock_col)
    final = df.merge(temp, on='Date', how='left')
    datetime_series = pd.to_datetime(final['Date'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    final = final.set_index(datetime_index)
    final = final.sort_values(by='Date')
    final = final.drop(columns=['Date','20SD'])
    display(final.head(1))
    # display(final[final.isnull().T.any()])
    # Remove any missing NA (some may not have any sentiment text on that day)
    final = final.dropna()
    final.to_pickle(f'{stock_name}_sa.pkl')
    return final
