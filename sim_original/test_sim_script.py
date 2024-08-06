import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("/Users/austin/code/liquidity-distribution-history/")
from swap_utils import *
from pool_state import v3Pool
from swap import swap
from sim import sim

from tqdm import tqdm

from collections import defaultdict

pool_add = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
pool = v3Pool(pool_add)

initial_port = [100000 * 1e6, 33 * 1e18]
start, end = pd.to_datetime("09-01-2021"), pd.to_datetime('06-01-2022')
## create trades
token0 = pool.getToken0()
token1 = pool.getToken1()

pct = .05
times = [*pd.date_range(start=start, end=end) + pd.Timedelta("6h")]
trades = [[t, np.random.choice([token0, token1])] for t in times]
trades = [t + [initial_port[0] * pct if t[1] == token0 else initial_port[1] * pct] for t in trades]
s = sim(pool)

# define trades for trades
s.defBTDate(start, end + pd.Timedelta("15d"))
s.loadPort(initial_port)

# load trades
s.loadTrades(trades)

# apply and compute performance
s.applyTrades()
_ = s.createPerformance()

# plot performance
fig = s.plotMetrics()

fig.savefig("performance.png")