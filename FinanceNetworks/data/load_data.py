from pathlib import Path

import yfinance as yf
import os
import pandas as pd

data_dir = Path(__file__).parent / "data_files"

def download_data(ticker: str, start_date: str = "2015-01-01", end_date: str = "2026-01-01"):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        actions=False
    )
    df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    return df

def save_df(df, ticker_name: str):
    df.to_csv(data_dir / f"{ticker_name}_data.csv")

def find_sp_top_n_stocks(n: int = 10):
    with open(data_dir / "sp500_list.txt", "r") as f:
        lines = f.read().splitlines()[2:]
    
    tickers = []
    for l in lines[:n]:
        ticker = l.split("\t")[2]
        tickers.append(ticker)
    return tickers

def check_exists_data(ticker_name: str):
    return (data_dir / f"{ticker_name}_data.csv").exists()

def download_data(num_tickers: int = 10):
    tickers = find_sp_top_n_stocks(num_tickers)
    for ticker in tickers:
        if check_exists_data(ticker):
            print(f"Data for {ticker} already exists. Skipping download.")
            continue
        df = download_data(ticker)
        save_df(df, ticker)

def get_data(num_tickers: int = 10):
    tickers = find_sp_top_n_stocks(num_tickers)
    data = {}
    for ticker in tickers:
        df = pd.read_csv(data_dir / f"{ticker}_data.csv", index_col=0)
        data[ticker] = df
    return data

if __name__ == "__main__":
    download_data(100)