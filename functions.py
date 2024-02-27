'''
This script contains some of the python functions utilised for analysis and data prep in main code
v1.0
'''
import pandas as pd
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import seaborn as sns

def chg(df, name):
    df.columns=df.columns.map(lambda x : x+name if x !='Date' else x)

def read_process_stock():
    '''
    This function will read in the stock market prices downloaded from yahoo finance
    in the period from 2015 to 2019.
    returns:
        df: all three US stocks
        df_msft: microsoft dataset
        df_aapl: apple dataset
        df_goog: google dataset
    '''
    msft = pd.read_csv('~/Downloads/proj/data/Stock/MSFT.csv')
    aapl = pd.read_csv('~/Downloads/proj/data/Stock/AAPL.csv')
    goog = pd.read_csv('~/Downloads/proj/data/Stock/GOOG.csv')
    chg(goog, '_goog')
    chg(msft, '_msft')
    chg(aapl, '_aapl')
    data = aapl.merge(msft).merge(goog)
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data.Date=pd.to_datetime( data.Date,errors='coerce')
    df = data[data.Date < '2020-01-01']
    df_msft = msft[msft.Date < '2020-01-01']
    df_aapl = aapl[aapl.Date < '2020-01-01']
    df_goog = goog[goog.Date < '2020-01-01']
    return df, df_msft, df_aapl, df_goog

def plot_ts(df, stock, currency):
    '''
    This function displays the plot of the closing price of the stock dataset.
    param df: dataset where the closing price of the stock can be found
    param stock: the stock ticker
    param currency: the currency the closing price is in
    '''
    fig, ax = plt.subplots(figsize=(12,2))
    ax.plot(df['Date'], df[f'Close_{stock}'], color='#008B8B')
    ax.set(xlabel="Date", ylabel=f"{currency}", title=f"{stock} Price")
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.show()

def get_ta(df):
    '''
    This function calculates the different technical indicators -
    MA7, MA20, MACD, Bollinger lower and upper bands
    and add to the input dataset
    param df: dataset consisting closing price of the stock
    returns:
        df: input dataset with columns of calculated technical indicators
    '''
    df['MA7'] = df.iloc[:,4].rolling(window=7).mean() #take close
    df['MA20'] = df.iloc[:,4].rolling(window=20).mean() #take close
    # MACD: subtracting the 26-period exponential moving average (EMA) from the 12-period EMA
    df['MACD'] = df.iloc[:,4].ewm(span=26).mean() - df.iloc[:,1].ewm(span=12,adjust=False).mean()
    # Bollinger Bands
    df['20SD'] = df.iloc[:, 4].rolling(20).std()
    df['upper_band'] = df['MA20'] + (df['20SD'] * 2)
    df['lower_band'] = df['MA20'] - (df['20SD'] * 2)
    return df

def prep_stock_with_technical_analysis(df, stock_name):
    '''
    This function displays the plots of closing price of the stock with technical indicators
    param df: dataset consiting closing price of the stock
    param stock_name: stock name 
    '''
    data = get_ta(df)
    df_ta = data.iloc[20:,:].reset_index(drop=True)
    df_ta['Date'] = pd.to_datetime(df_ta['Date']).dt.date
    df_ta.Date=pd.to_datetime(df_ta.Date,errors='coerce')
    display(df_ta.head(2))

    #plot graph
    fig,ax = plt.subplots(figsize=(13, 4), dpi = 250)
    x_ = range(3, df_ta.shape[0])
    x_ = list(df_ta.index)

    ax.plot(df_ta['Date'], df_ta[f'Close_{stock_name}'], label='Closing Price', color='#6A5ACD')
    ax.plot(df_ta['Date'], df_ta['MA7'], label='Moving Average (7 days)', color='g', linestyle='--')
    ax.plot(df_ta['Date'], df_ta['MA20'], label='Moving Average (20 days)', color='r', linestyle='-.')
    ax.plot(df_ta['Date'], df_ta['upper_band'], label='Boillinger upper', color='y', linestyle=':')
    ax.plot(df_ta['Date'], df_ta['lower_band'], label='Boillinger lower', color='y', linestyle=':')
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title('Technical indicators')
    plt.ylabel('Closing Price (USD)')
    plt.xlabel("Year")
    plt.legend()

    plt.show()
    return df_ta
