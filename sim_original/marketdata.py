import pandas as pd
import numpy as np

class marketdata: 
    '''
    calculateVol(freq: str, sample: str):
        Returns the volatility sampled at freq for the rolling sample of sample

        Example vol = calculateVol(freq = '5t', sample = '24h')

    calculatePriceSeries(start: pd.DateTime, end: pd.Datetime, freq: str)
        Returns a price series from start to end of frequency freq.
        Will default to start and end of series if no start/end provided
        Will return an unsampled series if freq is not provided

    
        Example px = calculatePriceSeries(freq = '5t')

    calculateGasSeries(start: pd.Datetime, end: pd.Datetime)
        Returns 
    '''
    def __init__(self, pool):
        self.pool = pool
        
    def _checkFreq(self, freq):
        if freq[-1] == 'm':
            print("m stands for month - use t for minute!")
            
    def _resampleOrSamplePrice(self, freq):
        px = self.pool.swaps.set_index("block_ts").copy()
        px['px'] = (px['sqrtPriceX96'].astype(float) ** 2 ) / (2 ** 192)
        if freq != '':
            px = px.resample(freq).last().ffill()
        
        return px
        
    def calculateVol(self, freq = '5t', sample = '24h'):
        self._checkFreq(freq)
            
        min_periods = int(pd.Timedelta(sample) / pd.Timedelta(freq))
        annualization = pd.Timedelta("365d") / pd.Timedelta(freq)
        
        vol = self._resampleOrSamplePrice(freq)

        vol['px'] = np.log(vol['px']).diff()
        
        vol = (
                vol
               .dropna()['px']
               .rolling(sample, min_periods = min_periods)
               .std()
               .dropna()
                )
        
        vol = vol * np.sqrt(annualization) * 100
        return vol
    
    def calculatePriceSeries(self, start = '', end = '', freq = ''):
        if freq != '':
            self._checkFreq(freq)
        px = self._resampleOrSamplePrice(freq)
    
        if start == '':
            start = px.index.min()
        if end == '':
            end = px.index.max()

        px = px[(px.index >= start) & (px.index <= end)]['px']

        return px
    
    def calculateGasSeries(self, start = '', end = ''):
        gas = self.pool.block_info.set_index("block_ts")['median_gas']
        gas.index = pd.to_datetime(gas.index)
        
        if start == '':
            start = gas.index.min()
        if end == '':
            end = gas.index.max()
            
        gas = gas[(gas.index >= start) & (gas.index <= end)]
        
        return gas
