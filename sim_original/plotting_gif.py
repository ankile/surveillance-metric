import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
import matplotlib.animation as animation
# Enable interactive plot
# add that if in a notebook
# %matplotlib notebook

import os

def feeToTS(feetier):
    return {
        100: 1,
        500: 10,
        3000: 60,
        10000: 200
    }.get(feetier)

def find_inc(max_inc):
    '''
    Takes the max transaction index in any mint/burn transactions
    and finds the decimals needed to ensure that the transaction index
    is a decimal.
    
    ie if max_inc is 1100, then dec will be 4
    as 1100 / 10**4 = .1100

    # TODO 
    Rework this to instead increase the blocksize to remain integers
    Also could find this using logs
    '''
    dec = 0
    
    frac, whole = math.modf(max_inc / 10 ** dec)
    while whole != 0:
        dec +=1

        # we want to find the minimum power st max_inc / 10 ** power is only decimal
        frac, whole = math.modf(max_inc / 10 ** dec)
        
    return dec

def tick_to_initialized_tick(df, col, ts):
    '''
    Ticks in Uniswap v3 have tick spacing. if tickspacing is 60, then ticks can only be
    every 60 indexes. 
    
    To make the sparse matrix smaller, we divide the ticks by the tick spacing to
    make the matrix smaller. this is to remove the rows in-between tick spaces
    that have to be 0
    '''
    # ive had one too many memory references
    df_copy = df.copy()
    df_copy['intermezzo'] = df_copy[col] / ts
    
    # ensure the positions are correctly specified (only would happen on a data error)
    err = f"Incorrect tick spacing of {ts} for lower"
    assert math.isclose(df_copy['intermezzo'].apply(lambda x: math.modf(x)[0]).sum(), 0), err
    
    return df_copy['intermezzo'].astype(int)

def create_coo(mb, ts):
    '''
    Shrink down the tick spacing then invert the liquidity on the upper tick
    as described in the v3 whitepaper.
    '''
    mb['idxLower'] = tick_to_initialized_tick(mb, 'tickLower', ts)
    mb['idxUpper'] = tick_to_initialized_tick(mb, 'tickUpper', ts)
    
    liqLower = mb[['position_index', 'idxLower', 'amount']].values
    liqUpper = mb[['position_index', 'idxUpper', 'amount']].values

    # invert the liq
    liqUpper[:, 2] = liqUpper[:, 2] * -1
    
    coo = np.concatenate([liqLower, liqUpper])
    return coo

def create_distribution(mb, ts):
    '''
    add together the block + the position in the block to get the 
    block_idx. we then create a key value pair of the mint/burn index
    to the key value pair. this is because we need the column to be
    an integer, but we have to index using decimals. we also want
    the positions to be increasing integers so the sparse matrix
    is as small as possible
    
    TODO:
    Look into multiplying the block index by 10 ** dec instead of dividing so
    everything is an integer. probably will explode the matrix tho
    '''
    dec =  find_inc(mb['transaction_index'].max())
    mb['block_frac'] = mb['transaction_index'] / 10 ** dec
    mb['block_idx'] = mb['block_number'] + mb['block_frac']
    mb['position_index'] = mb.index
    kv = {key: value for key, value in mb[['position_index', 
                                      'block_idx']].values}
    coo = create_coo(mb, ts)
    
    return coo, kv

def create_liq(ts, coordinate_payload = '', mb = '', as_of = ''):
    '''
    either pre-compute the distribution or the code will create one for you
    pre-computing is much faster
    
    the as_of date is the block + idx so that it's easier to index.
    the vectorization is to apply the block_index (an unsigned increasing int)
    to the block_index. we can then index using that coordinate matrix
    
    we then make a sparse matrix where the column are the mint burn and
    the rows are the indexes. we place a + liquidity value on the lower tick 
    and a - liquidity value on the upper tick for mints. then you then sum
    across all the columns to get the net liquidity at each tick. you can then
    cumulative sum the ticks to apply the deltas.
    
    because the ticks can expand, we have to re-create the range based off the
    sequence from the row min to row max by the tick spacing
    
    we adjust by 1e10 for the plot
    '''
    if coordinate_payload == '':
        err = "Please provide a coordinate matrix/key-value or mint/burns"
        assert mb != '', err
    
        coo, kv = create_distribution(mb, ts)
    else:
        coo, kv = coordinate_payload
        
    coordinates = np.vectorize(kv.get)(coo[:, 0])
    
    if as_of:
        coo = coo[coordinates < as_of, :]
        
    col, row, amt = coo[:, 0].astype(int), coo[:, 1].astype(int), coo[:, 2]

    # indicies must be >= 0
    # ticks can be negative, so we need to add min_tick to it
    min_tick = row.min()
    row = row - min_tick

    mat = sparse.coo_matrix((amt, (row, col)))
    netLiquidity = mat.sum(axis = 1)
    liquidity = netLiquidity.cumsum() / 1e10

    ticks = np.arange(row.min(), row.max() + 1, 1)
    ticks = (ticks + min_tick) * ts

    x, y = ticks.tolist(), liquidity.tolist()[0]
    
    return x, y

'''
Read in the data from bigquery - pull info about the pool
from the factory and pull down the swaps to get the current spot price
'''
df = pd.read_csv('labeled_mb.csv').drop("Unnamed: 0", axis = 1)

#addr = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
addr = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'
bal_adj = 1e9

factory = pd.read_csv("data/factory_v3.csv")
pool = (df[(df['address'] == addr) &
            (df['amount'] != '0')]
        .sort_values(by = ['block_number', 'transaction_index'])
        .reset_index(drop = True)
       ).copy()

pool['amount'] = pool['amount'].astype(np.float64) / bal_adj

factory_data = factory[factory['pool'] == addr]
if factory_data.empty:
    raise ValueError("Incorrect pool")
fee = factory_data['fee'].astype(int).item()

ts = feeToTS(fee)
print(ts)
pool['ts'] = ts

# swaps
swaps = pd.read_csv("data/5bps_swaps.csv")
swaps['block_timestamp'] = pd.to_datetime(swaps['block_timestamp']).dt.tz_localize(None)
swaps = swaps.sort_values(by = ['block_number', 'transaction_index'])

mb = pool[['block_number', 'transaction_index', 'tickLower', 'tickUpper', 'amount']].copy()


'''
Plotting code starts here
'''
# blocks to iterate over
blocks = np.arange(pool['block_number'].iloc[1000], pool['block_number'].max(), 5000)

# precompute the liq distribution
coo, kv = create_distribution(mb, ts)

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([],[], '-', color = "#FA3492")
ax.set_xlim(500, 6000)
ax.set_ylim(0, 10)
ax.set_xlabel("WETH/USDC exchange rate ($)")
ax.set_ylabel("Liquidity at tick")

vl = ax.axvline(0, ls = '-', linewidth = .5, color = "black", linestyle = "--")

def init(): 
    vl.set_data([],[])
    return 

def animate(i, payload):
    coo, kv, blocks, swaps, pool = payload
    as_of = blocks[i]
    
    timestamp = pool[pool['block_number'] <= as_of]['block_timestamp'].iloc[-1]
    timestamp = pd.to_datetime(timestamp)
    tick = swaps[swaps['block_number'] <= as_of].iloc[-1]['tick']
    cur_px = 1 / (1.0001 ** tick) * 1e12


    x, y = create_liq(ts, coordinate_payload = (coo, kv), as_of = as_of)
    
    # the pool is inverted so we flip it to get weth/usdc price
    px = [1 / (1.0001 ** _x) * 1e12 for _x in x]

    line.set_xdata(px)
    line.set_ydata(y)
    ax.set_title(timestamp.strftime("WETH/USDC 5 bps - %m/%d/%Y"))
    vl.set_xdata([cur_px, cur_px])
    
    return line, vl

ani = animation.FuncAnimation(fig, animate, frames=1500, fargs=((coo, kv, blocks, swaps, pool),),
                              interval=100, blit=True)
write = False
if write:
    f = "/Users/austin/code/lpActivity/animation.gif" 
    writergif = animation.PillowWriter(fps=15) 
    ani.save(f, writer=writergif)
