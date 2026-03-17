from pathlib import Path

import numpy as np
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
    for i in range(n):
        l = lines[i]
        ticker = l.split("\t")[2]
        tickers.append(ticker)
    return tickers

def check_exists_data(ticker_name: str):
    return (data_dir / f"{ticker_name}_data.csv").exists()

def download_data_all(num_tickers: int = 10):
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

def get_market_correlation():
    if check_exists_data("SPY"):
        print(f"Data for SPY already exists. Skipping download.")
        return
    df = download_data("SPY")
    save_df(df, "SPY")


def get_spy_returns() -> pd.Series:
    """Load SPY data and return a daily log-returns Series indexed by Date."""
    path = data_dir / "SPY_data.csv"
    if not path.exists():
        get_market_correlation()
    df = pd.read_csv(path, index_col=0)
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    returns = 100 * np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    returns.name = "Market_Returns"
    return returns.dropna()
    

if __name__ == "__main__":
    download_data_all(500)