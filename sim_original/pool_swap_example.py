import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
import matplotlib.animation as animation
# Enable interactive plot
# add that if in a notebook
#%matplotlib notebook

import os
from pool_state import v3Pool
from swap_utils import *

pool_add = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
pool = v3Pool(pool_add)

# needed to pull in the variables into state to calc swaps
pool.initializeLiquidity()

# get the last pool tick
coo, kv = pool.getCOO()
as_of = max([kv[_x] for _x in kv.keys()])
as_of = math.floor(as_of + 1)


tokenIn = pool.getToken0()
tokenOut = pool.getToken1()
swapIn = 100000 * 1e6 # 100k usdc
as_of = 14814941

swapParams = {'tokenIn': tokenIn,
              'tokenOut': tokenOut,
              'input': swapIn,
              'as_of': as_of}

out = pool.swapIn(swapParams)

# '50.9930'
f'{out / 1e18:.4f}'