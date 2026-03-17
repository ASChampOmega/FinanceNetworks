from .load_data import get_data, download_data, get_spy_returns
from .preprocess import preprocess, preprocess_for_har

def get_data_for_har(num_tickers: int = 10):
    data = get_data(num_tickers=num_tickers)
    market_returns = get_spy_returns()
    preprocessed_data = preprocess_for_har(data, market_returns=market_returns)
    return preprocessed_data