import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
import matplotlib.animation as animation
from tqdm import tqdm

import sys
sys.path.append("/Users/austin/code/liquidity-distribution-history/")
from pool_state import v3Pool
from swap_utils import *
from marketdata import *
from position_helpers import *
# Enable interactive plot
# add that if in a notebook
# %matplotlib notebook
from tqdm.contrib.concurrent import process_map
import os
from collections import defaultdict
import pickle as pkl

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fee_callback_calc(inputs):
    addr, swaps, payload = inputs
    
    pool = v3Pool(addr, initialize = False, verbose = False)
    pool.addMB(payload)

    tickToFees0 = {}
    tickToFees1 = {}

    for swap in swaps:
        idx = swap['block_idx']

        tokenIn = ''
        if float(swap['amount0']) > 0:
            tokenIn = pool.token0
            amtIn = float(swap['amount0'])
        elif float(swap['amount1']) > 0:
            tokenIn = pool.token1
            amtIn = float(swap['amount1'])

        swapParams = {'tokenIn': tokenIn,
                    'input': amtIn,
                    'as_of': idx}

        _, huer = pool.swapIn(swapParams)
        feesCallBack = huer[5]
        
        if tokenIn == pool.token0:
            if idx in tickToFees0.keys():
                intra_transaction = tickToFees0[idx]
                for tick in feesCallBack.keys():
                    intra_transaction[tick]+=feesCallBack[tick]

                tickToFees0[idx] = intra_transaction

            else:
                tickToFees0[idx] = feesCallBack
        else:
            if idx in tickToFees1.keys():
                intra_transaction = tickToFees1[idx]
                for tick in feesCallBack.keys():
                    intra_transaction[tick]+=feesCallBack[tick]

                tickToFees1[idx] = intra_transaction

            else:
                tickToFees1[idx] = feesCallBack

    return (tickToFees0, tickToFees1)




if __name__ == "__main__":
    addr = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    pool = v3Pool(addr)

    n_workers = 5
    '''
    tl, tu = np.percentile(px, [10, 90])

    tl = ((np.log(tl) / np.log(1.0001)) // ts) * ts
    tu = ((np.log(tu) / np.log(1.0001)) // ts) * ts
    '''

    position = {'amount1': 10 * 1e18,
                'tickLower': 193080,
                'tickUpper': 202440}

    timing = {"mint": 14000000.0,
              "burn": 15400000.0}

    m, b = createOrder(position, timing, pool)
    print(m)
    m = pd.DataFrame(m, index = [0])
    b = pd.DataFrame(b, index = [1])

    payload = pd.concat([m, b]).copy()
    pool.addMB(payload)

    mb = pool.mb.copy()    

    swaps = [swap for _, swap in [*pool.swaps.iterrows()] if (swap['block_idx'] >= timing['mint']) & (swap['block_idx'] <= timing['burn'])]

    # determine the chunk-size
    # we want to initialize the least amount of pools we have to
    # while still breaking everyting up equally
    chunk_size = len(swaps) / n_workers
    chunk_size = math.ceil(chunk_size) + 1
    t = [(addr, i, mb) for i in chunks(swaps, chunk_size)]

    r = process_map(fee_callback_calc, t, max_workers=n_workers)

    with open('fee_dump_for_position.pkl', 'wb') as f:
        pkl.dump(r, f)

    with open('fee_dump_for_position.pkl', 'rb') as f:
        r = pkl.load(f)
    
    l = m['amount'].item() / pool.bal_adj

    t0 = timing.get('mint', 'Missing')
    assert t0 != 'Missing', "Mint must be provided"

    t1 = timing.get("burn", 'Missing')
    if t1 == 'Missing':
        t1 = np.inf
    
    tl = position['tickLower']
    tu = position['tickUpper']

    cur_price = pool.getPriceAt(t1)
    print(calc_spot(cur_price, tl, tu, pool.ts, l))
    # price_fees_t0_to_t1(t0, t1, tl, tu, l, position, data)
    print(price_fees_t0_to_t1(t0, t1, tl, tu, l, r))

