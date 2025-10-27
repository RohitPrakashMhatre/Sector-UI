import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import joblib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def chunk_batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def safe_fetch(symbol, period, attempts=3):
    for attempt in attempts:
        try:
            stock = yf.Ticker(symbol)
            history = stock.history(period=period)

            if not history.empty:
                return history
        except Exception as e:
            if attempt == attempts - 1:
                print(f"Failed for {symbol} after {attempts} retries: {e}")
            else:
                time.sleep(1)
    return pd.DataFrame()

def fetch_batch(batch, weeks, dates, period='1y'):
    temp_result = {}
    for symbol in batch:
        try:
            hist = safe_fetch(symbol, period)

            temp_result[symbol] = {}
            for week, date in zip(weeks, dates):
                if date in hist.index:
                    close = np.ceil(hist.at[date, 'Close'] * 100) / 100
                    temp_result[symbol][week] = close
        except Exception as e:
            print(f"Fetching error for ticker {symbol}: {e}")
    return temp_result

def multi_threads(all_result, batches, weeks, dates, max_workers=10):
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_batch, batch, weeks, dates) for batch in batches]
        for future in futures:
            all_result.append(future)
    end = time.time()
    return all_result, (start - end)

def process_fetched_data(df, fetched_data):
    df = df.copy()
    for result in fetched_data:
        for sym, week_data in result.items():
            for week, price in week_data.items():
                df.loc[sym, week] = price 
    return df

def clean_data(df):
    df = df.copy()
    # fix SYMBOL formatting and set index
    df['SYMBOL'] = df['SYMBOL'].str.replace(':', '.')
    df.set_index('SYMBOL', inplace=True)
    
    # initialize price columns
    df['base'] = np.nan
    df['Current'] = np.nan
    for i in range(1, 10):
        df[f"Week {i}"] = np.nan
        
    return df 


def pre_processing(df, dates, weeks):
    df = df.copy()
    # only drop rows where base is NaN
    df.dropna(subset=['base'], inplace=True)

    # Calculate Live % for stock data
    df['Live'] = np.ceil(((df['Current'] - df['base']) / df['base']) * 100)
    for i in range(1, 10):
        df[f'Live {i}'] = np.ceil(((df[f'Week {i}'] - df['base']) / df['base']) * 100)

    df = df[['Sector', 'NSE CODE', 'Live'] + [f'Live {i}' for i in range(1, 10)]].copy()

    # NIFTY DataFrame
    nifty_df = pd.DataFrame(index=['^NSEI'])
    nifty_df['base'] = np.nan
    nifty_df['Current'] = np.nan
    for i in range(1, 10):
        nifty_df[f"Week {i}"] = np.nan

    # Fetch NIFTY data
    Symbol = '^NSEI'
    stock = yf.Ticker(Symbol)
    hist = stock.history(period='1y')
    
    # assign base and current
    nifty_df.loc[Symbol, 'base'] = np.ceil(hist.loc[pd.Timestamp(dates[0])]['Close'] * 100) / 100
    nifty_df.loc[Symbol, 'Current'] = np.ceil(hist.iloc[-1]['Close'] * 100) / 100
    
    for date, week in zip(dates, weeks):
        if pd.Timestamp(date) in hist.index:
            nifty_df.loc[Symbol, week] = np.ceil(hist.loc[pd.Timestamp(date)]['Close'] * 100) / 100

    # Calculate NIFTY % change
    nifty_df["Live"] = round(((nifty_df['Current'] - nifty_df['base']) / nifty_df['base']) * 100, 2)
    for i in range(1, 10):
        nifty_df[f"Live {i}"] = round(((nifty_df[f'Week {i}'] - nifty_df['base']) / nifty_df['base']) * 100, 2)

    # Add NIFTY effect to all stocks
    nifty_values = nifty_df.loc[Symbol, ['Live'] + [f'Live {i}' for i in range(1, 10)]]

    for col, val in nifty_values.items():
        df[col] = df[col] + val

    return df

def find_top_n_live(df, n):
    df = df.copy()
    group_sec = df.groupby('Sector')[['Live','Live 1','Live 2','Live 3','Live 4','Live 5','Live 6','Live 7','Live 8','Live 9']].mean()
    group_sec = group_sec.sort_values(by='Live',ascending=False)
    group_sec.reset_index(inplace=True)
    group_sec.index = group_sec.index + 1
    top_live = group_sec.head(n)

    return top_live

def top_n_histogram(top_live):
    weekly_df = top_live.set_index('Sector').iloc[:,:]
    weekly_df.sort_values(by = 'Live',ascending=False)

    plt.figure(figsize=(12,6))
    sns.heatmap(weekly_df,cmap="RdYlGn",annot=True,cbar=False,linewidth=0.5)
    plt.title("Heatmap of Sector performance over time")
    plt.xlabel('Live Weeks')
    plt.ylabel('Sectors')
    plt.show()


def sector_volatility(weekly_df):
    plt.figure(figsize=(12,6))
    sns.boxplot(weekly_df.T)
    plt.xticks(rotation=90)
    plt.title('Sector Volatility analysis')
    plt.show()

def rank_sector(top_live):
    columns = ['Live', 'Live 1', 'Live 2', 'Live 3', 'Live 4', 'Live 5',
       'Live 6', 'Live 7', 'Live 8', 'Live 9']
    grid_df = top_live.copy()
    for col in columns:
        grid_df[col] = grid_df[col].rank(method='dense',ascending=False).astype(int)

    styled_df = grid_df.style
    for col in columns:
        styled_df = styled_df.background_gradient(cmap='RdYlGn_r',subset=[col])

    return styled_df

def improved_sectors(data):
    def find_improved_sector(data):
        condition = (grid_df['Live 1'] - grid_df['Live'] >= 2) | (grid_df['Live 2'] - grid_df['Live'] >= 2)
        return data.loc[condition,['Sector','Live','Live 1','Live 2']]

    improved = find_improved_sector(grid_df)
    improved_sec = improved['Sector']
    return improved_sec

def find_improved_sectors_stocks(unprocessed_df, avg_df, improved_sec):

    results = {sec: [] for sec in improved_sec}
    dummy_df = unprocessed_df.copy().reset_index()
    avg_sec = avg_df.groupby('Sector')['Live'].mean()

    # print("Improved Sectors Stocks:")
    # print("---------------------------------------------------------")

    for imp in improved_sec:
        if imp in avg_sec.index:
            avg_value = avg_sec[imp]
            # filter stocks in improved sector above average live
            stock_list = dummy_df[
                (dummy_df['Sector'] == imp) & 
                (dummy_df['Live'] >= avg_value)
            ]['NSE CODE'].to_list()

            results[imp] = stock_list

            # if stock_list: 
            #     print(f"{imp}: {stock_list}")

    return results












