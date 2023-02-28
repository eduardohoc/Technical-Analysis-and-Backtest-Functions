+import pandas as pd
import numpy as np

from scipy.stats import kstest



def clear_signals(signals, window_size):
    """
    Clear out signals in a Series of just long or short signals.
    
    Remove the number of signals down to 1 within the window size time period.
    
    Parameters
    ----------
    signals : Pandas Series
        The long, short, or do nothing signals
    window_size : int
        The number of days to have a single signal       
    
    Returns
    -------
    signals : Pandas Series
        Signals with the signals removed from the window size
    """
    # Start with buffer of window size
    # This handles the edge case of calculating past_signal in the beginning
    clean_signals = [0]*window_size
    
    for signal_i, current_signal in enumerate(signals):
        # Check if there was a signal in the past window_size of days
        has_past_signal = bool(sum(clean_signals[signal_i:signal_i+window_size]))
        # Use the current signal if there's no past signal, else 0/False
        clean_signals.append(not has_past_signal and current_signal)
        
    # Remove buffer
    clean_signals = clean_signals[window_size:]

    # Return the signals as a Series of Ints
    return pd.Series(np.array(clean_signals).astype(np.int), signals.index)


def filter_signals(signal, lookahead_days):
    """
    Filter out signals in a DataFrame.
    
    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_days : int
        The number of days to look ahead
    
    Returns
    -------
    filtered_signal : DataFrame
        The filtered long, short, and do nothing signals for each ticker and date
    """
    #TODO: Implement function
    
    long_signal = pd.DataFrame(0, columns = signal.columns, index = signal.index)
    short_signal = pd.DataFrame(0, columns = signal.columns, index = signal.index)
    filter_l = pd.DataFrame(0, columns = signal.columns, index = signal.index)
    filter_s = pd.DataFrame(0, columns = signal.columns, index = signal.index)
    long_signal[signal>0] = 1
    short_signal[signal<0] = 1
      
    #Iterate over columns of the signal DataFrame
      
    for i, column in enumerate(signal):
        filter_l.loc[:,column] = clear_signals(long_signal.loc[:,column], lookahead_days) 
        filter_s.loc[:,column] = -1*clear_signals(short_signal.loc[:,column], lookahead_days)    
    
    
    return filter_l + filter_s


def get_lookahead_prices(close, lookahead_days):
    """
    Get the lookahead prices for `lookahead_days` number of days.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_days : int
        The number of days to look ahead
    
    Returns
    -------<  
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    """
    #TODO: Implement function
    lookahead_prices = close.shift(-lookahead_days)
    
    return lookahead_prices


def get_return_lookahead(close, lookahead_prices):
    """
    Calculate the log returns from the lookahead days to the signal day.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    
    Returns
    -------
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    """
    #TODO: Implement function
    lookahead_returns = np.log(lookahead_prices)-np.log(close)
    
    
    return lookahead_returns


def get_signal_return(signal, lookahead_returns):
    """
    Compute the signal returns.
    
    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    
    Returns
    -------
    signal_return : DataFrame
        Signal returns for each ticker and date
    """
    #TODO: Implement function
    
    signal_return = signal*lookahead_returns
    
    
    return signal_return


def calculate_kstest(long_short_signal_returns):
    """
    Calculate the KS-Test against the signal returns with a long or short signal.
    
    Parameters
    ----------
    long_short_signal_returns : DataFrame
        The signal returns which have a signal.
        This DataFrame contains two columns, "ticker" and "signal_return"
    
    Returns
    -------
    ks_values : Pandas Series
        KS static for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    """
    #TODO: Implement function
    
    grouped_l_s = long_short_signal_returns.groupby('ticker')
    norm_arg = (np.mean(long_short_signal_returns), np.std(long_short_signal_returns,ddof = 0))

    ks = np.array([])
    p =  np.array([])
    i = np.array([])
    
    
    for name, group in grouped_l_s:
        ks = np.append(ks,kstest(group['signal_return'],'norm',norm_arg)[0])
        p =np.append(p,kstest(group['signal_return'],'norm',norm_arg)[1])
        i = np.append(i,name)
   
    ks_values = pd.Series(ks,index=i)
    p_values = pd.Series(p,index=i)
    
    return (ks_values, p_values)


def find_outliers(ks_values, p_values, ks_threshold, pvalue_threshold=0.05):
    """
    Find outlying symbols using KS values and P-values
    
    Parameters
    ----------
    ks_values : Pandas Series
        KS static for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    ks_threshold : float
        The threshold for the KS statistic
    pvalue_threshold : float
        The threshold for the p-value
    
    Returns
    -------
    outliers : set of str
        Symbols that are outliers
    """
    #TODO: Implement function
    
    s = pd.Series(1,index=ks_values.index)
    outliers = s[(ks_values > ks_threshold) | (p_values < pvalue_threshold)]
    
    
    
    return set(outliers.index.values)
