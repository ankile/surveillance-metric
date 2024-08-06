import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("/Users/austin/code/liquidity-distribution-history/")
from swap_utils import *
from pool_state import v3Pool
from swap import swap

from tqdm import tqdm

from collections import defaultdict

class sim:
    def __init__(self, v3Pool):
        self.pool = v3Pool
        
        self.trades = None
        self.portfolio = None
        self.deltas = None
        
        self.start = None
        self.end = None
        self.performance = None

        self.gas = False
        
    def defBTDate(self, tsS, tsE):
        assert tsS < tsE, "Incorrectly specified start and end date"
        
        self.start = tsS
        self.end = tsE
        
    def loadTrades(self, trades):
        self.trades = trades
    
    def loadPort(self, port):
        self.portfolio = port
        
    def applyTrades(self, gas = False):
        if gas:
            self.gas = True

        deltas = []
        for trade in self.trades:
            ts, tokenIn, swapIn = trade

            as_of = self.pool.getBlockAtTS(ts)

            token0IsTokenIn = True
            if tokenIn == self.pool.getToken0():
                tokenOut = self.pool.getToken1()
            else:
                tokenOut = self.pool.getToken0()
                token0IsTokenIn = False

            swapParams = {'tokenIn': tokenIn,
                            'input': swapIn,
                            'as_of': as_of,
                            'gasFee': gas}

            out, heur = self.pool.swapIn(swapParams)

            if token0IsTokenIn:
                delta = [ts, tokenIn, -swapIn, tokenOut, out, heur[-1]]
            else:
                delta = [ts, tokenOut, out, tokenIn, -swapIn, heur[-1]]

            deltas.append(delta)
            
        self.deltas = deltas
    
    def createPerformance(self):
        data = []
        ser = pd.date_range(self.start, self.end, freq = "1D")
        df = pd.DataFrame(self.deltas, columns = ['ts', 'token0', 'token0Delta', 'token1', 'token1Delta', 'gas'])
        prev = None
        for t in ser:
            prev = t - pd.Timedelta("1d")

            bn = self.pool.getBlockAtTS(t)
            token1ToToken0 = 1 / (self.pool.getPriceAt(bn) ** 2) * 1e12

            hodl = self.portfolio[0] / 1e6 * 1 + self.portfolio[1] / 1e18 * token1ToToken0
            deltas = df[df['ts'] < t]
            daily_deltas = df[(df['ts'] < t) & (df['ts'] >= prev)]

            d0, d1, gas_total, trades_per_day = (deltas['token0Delta'].sum(), 
                                                 deltas['token1Delta'].sum(), 
                                                 (daily_deltas['gas'].sum() * token1ToToken0),
                                                 (daily_deltas['gas'].count()))
            portfolio = ((self.portfolio[0] + d0) / 1e6 * 1) + (self.portfolio[1] + d1) / 1e18 * token1ToToken0
            data.append([t, hodl, portfolio, gas_total, trades_per_day])

        performance = pd.DataFrame(data, columns = ['time', 'Initial Portfolio', 'Strategy', 'Gas Used', 'Trade Per Day'])
        
        self.performance = performance
        return performance
    
    def plotMetricsWOGas(self):
        df = self.performance.set_index('time').copy()
        
        df['Strategy Performance'] = df['Strategy'] - df['Initial Portfolio'] 

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 10))

        ax1.plot(df['Initial Portfolio'], label = "Initial Portfolio", color = 'black', linestyle = '--')
        ax1.plot(df['Strategy'], label = "Strategy", color = "dodgerblue")
        ax1.legend()
        ax1.set_ylabel("Portfolio value in Token 0")
        ax1.set_xlabel("Date")
        
        ax2.plot(df['Strategy Performance'], label = "Performance", color = 'black')
        ax2.axhline(0, color = 'grey', linestyle = "--")
        ax2.legend()
        ax2.set_ylabel("Performance in Token 0")
        ax2.set_xlabel("Date")
        
        return fig

    def plotMetrics(self):
        if not self.gas:
            self.plotMetricsWOGas()
        
        df = self.performance.set_index('time').copy()
        df['Gas Used'] = df['Gas Used'].cumsum()
        df['Cumulative Trades'] = df['Trade Per Day'].cumsum()

        df['Strategy Performance'] = df['Strategy'] - df['Initial Portfolio']
        df['Strategy - Gas'] = df['Strategy'] - df['Gas Used']
        df['Strategy Performance - Gas'] = df['Strategy Performance'] - df['Gas Used']

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (10, 10))

        ax1.plot(df['Initial Portfolio'], label = "Initial Portfolio", color = 'black', linestyle = '--')
        # ax1.plot(df['Strategy'], label = "Strategy", color = "firebrick")
        ax1.plot(df['Strategy - Gas'], label = "Strategy - Gas", color = "dodgerblue")
        ax1.legend()
        ax1.set_ylabel("Portfolio value in Token 0")
        ax1.set_xlabel("Date")
        
        ax2.plot(df['Strategy Performance'], label = "Performance", color = 'black')
        ax2.plot(df['Strategy Performance - Gas'], label = "Performance - Gas", color = 'mediumpurple')
        ax2.axhline(0, color = 'grey', linestyle = "--")
        ax2.legend()
        ax2.set_ylabel("Performance in Token 0")
        ax2.set_xlabel("Date")

        ax3.plot(df['Gas Used'], label = "Gas", color = 'darkcyan')
        ax3_2 = ax3.twinx()
        ax3_2.plot(df['Gas Used'] / df['Cumulative Trades'], label = 'Average Gas Per Trade', color = 'firebrick')
        ax3.legend(loc = 'upper left')
        ax3_2.legend(loc = 'upper right')
        ax3.set_ylabel("Gas in Token 0")
        ax3_2.set_ylabel("Average Gas Cost in Token 0")
        ax3.set_xlabel("Date")
        
        return fig