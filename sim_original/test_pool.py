import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
import matplotlib.animation as animation
import os
from swap_utils import *
from pool_state import v3Pool
from swap import swap
import time

import psycopg2
from google.cloud import bigquery


if __name__ == "__main__":
    # initialize the pool
    pool_add = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    pool = v3Pool(pool_add)

    # swapIn
    tokenIn = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    tokenOut = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    swapIn = 160843290211
    as_of = (14405576 - 1)

    for q in [2500 for _ in range(0, 10)]:
        as_of+=q

        swapParams = {'tokenIn': tokenIn,
                    'tokenOut': tokenOut,
                    'input': swapIn,
                    'as_of': as_of}

        out, heur = pool.swapIn(swapParams)
        print(out, heur, (swapIn / out) * 1e12)

    # swapToPrice
    tokenIn = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    tokenOut = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    as_of = 14814941
    pcts = [-.08, -.02, -.01, -.005, -.001, .001, .005, .01, .02, .04, .08]

    swapParams = {'tokenOut': tokenIn,
                'pcts': pcts,
                'as_of': as_of}

    out, heur = pool.swapToPrice(swapParams)
    print(out)


    # test out the swap structure
    tokenIn = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    tokenOut = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    swapIn = 160843290211
    as_of = (14405576 - 1)
    swapParams = {'tokenIn': tokenIn,
                'tokenOut': tokenOut,
                'input': swapIn,
                'as_of': as_of}
    print(swap(pool).swapIn(swapParams))