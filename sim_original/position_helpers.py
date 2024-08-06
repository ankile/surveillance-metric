import numpy as np
import pandas as pd
import math


def calcToken1FromParams(position_struct, pool):
    as_of = position_struct.get("block_idx")
    token0Amt = position_struct.get('amount0')

    assert token0Amt != None, "Missing token0Amt"
    tl =  position_struct.get("tickLower")
    tu =  position_struct.get("tickUpper")

    cur_price, tickAt = pool.getPriceAt(as_of, return_block = True)
    ret_cur_tick = pool.getTickAt(as_of)
    ts = pool.ts

    cur_tick = np.log(cur_price ** 2) / np.log(1.0001)
    cur_tick = (cur_tick // ts) * ts

    assert cur_tick <= tu, 'Out-of-range'

    p_b = np.sqrt(1.0001 ** tu)
    p_a = np.sqrt(1.0001 ** tl)

    if tl <= cur_tick: 
        l_x = token0Amt * (cur_price * p_b) / (p_b - cur_price)
        token1Amt = l_x * (cur_price - p_a) 
        
    else:
        # out-of-range lower
        l_x = token0Amt * (p_a * p_b) / (p_b - p_a)
        token1Amt = 0
        
    return token1Amt, l_x


def calcToken0FromParams(position_struct, pool):
    as_of = position_struct.get("block_idx")
    token1Amt = position_struct.get('amount1')

    assert token1Amt != None, "Missing token1Amt"

    tl =  position_struct.get("tickLower")
    tu =  position_struct.get("tickUpper")

    cur_price, tickAt = pool.getPriceAt(as_of, return_block = True)
    ts = pool.ts

    cur_tick = np.log(cur_price ** 2) / np.log(1.0001)
    cur_tick = (cur_tick // ts) * ts

    assert tl < cur_tick , 'Out-of-range'

    p_b = np.sqrt(1.0001 ** tu)
    p_a = np.sqrt(1.0001 ** tl)

    if tl <= cur_tick: 
        l_y = token1Amt / (cur_price - p_a)
        token0Amt = l_y * (p_b - cur_price) / (cur_price * p_b)
        
    else:
        # out-of-range lower
        l_y = token1Amt / (p_b - p_a)
        token0Amt = 0
        
    return token0Amt, l_y


def fillTradeData(position_struct, pool):
    if position_struct.get("amount0", "Missing") == "Missing":
        token0Amt, liquidity = calcToken0FromParams(position_struct, pool)
        position_struct['amount0'] = token0Amt
        position_struct['amount'] = liquidity

    elif position_struct.get("amount1", "Missing") == "Missing":
        token1Amt, liquidity = calcToken1FromParams(position_struct, pool)
        position_struct['amount1'] = token1Amt
        position_struct['amount'] = liquidity

    else:
        _, liquidity_x = calcToken0FromParams(position_struct, pool)
        _, liquidity_y = calcToken1FromParams(position_struct, pool)

        if not np.isclose(liquidity_x, liquidity_y):
            print(f"Input mismatch: X: {liquidity_x} Y: {liquidity_y}")

        liquidity = min([liquidity_x, liquidity_y])
        position_struct['amount'] = liquidity

    return position_struct


def define_time_params(position_struct, pool):
    as_of = position_struct.get("block_idx", 'Missing')

    assert as_of != "Missing", "Unspecified block index"

    idx, block = math.modf(as_of)
    idx = idx * 1e4

    timestamp = pool.getTSAtBlock(block)

    position_struct['block_number'] = int(block)
    position_struct['transaction_index'] = idx
    position_struct['block_ts'] = timestamp

    return position_struct

def fill_extra_data(position_struct, typeOfTransaction, pool):
    position_struct['tx_hash'] = '0x0'
    position_struct['log_index'] = 0
    position_struct['address'] = pool.pool
    position_struct['owner'] = '0x0'
    position_struct['tokenID'] = 'user'
    position_struct['to_address'] = '0x0'
    position_struct['from_address'] = '0x0'

    assert typeOfTransaction in [-1, 1], 'typeOfTransaction must be either -1 or 1'
    position_struct['type'] = typeOfTransaction

    return position_struct

def changeTradeDataForBurn(position_struct):
    '''
    it is not clear what the price will be after the simulation is ran
    we just know the liquidity is burnt. will calculate the spot holdings
    at the end
    '''
    position_struct['amount'] = -1 * position_struct['amount']
    position_struct['amount0'] = "0x0"
    position_struct['amount1'] = "0x0"

    return position_struct


def createOrder(position_struct, timing_struct, pool): 
    err = "At least one mint must be provided"
    assert timing_struct.get('mint', "Missing") != "Missing", err
    
    position_struct['block_idx'] = timing_struct['mint']
    position_struct = fillTradeData(position_struct, pool)
    position_struct = define_time_params(position_struct, pool)
    position_struct = fill_extra_data(position_struct, 1, pool)
            
    mint = position_struct.copy()
    
    burn_ts = timing_struct.get('burn', 'Missing')
    if burn_ts == 'Missing':
        _, burn_ts = pool.getPriceAt(np.inf, return_block = True)
        
    position_struct = changeTradeDataForBurn(position_struct)
    position_struct['block_idx'] = burn_ts
    position_struct = define_time_params(position_struct, pool)
    position_struct = fill_extra_data(position_struct, -1, pool)
    
    burn = position_struct.copy()

    return (mint, burn)


def price_fees_t0_to_t1(t0, t1, tl, tu, l, data): 
    '''
    This assumes that liquidity does not change over-time
    '''
    times = []

    posToken0Fees = 0
    posToken1Fees = 0
    for chunk in data:
        token0Fees = chunk[0]
        times = [k for k in token0Fees.keys() if (k > t0) and (k <= t1)]
        for time in times:
            ticks = [tick for tick in token0Fees[time] if (tick>=tl) and (tick<tu)]
            for tick in ticks:
                posToken0Fees+=token0Fees[time][tick] * l

        token1Fees = chunk[1]
        times = [k for k in token1Fees.keys() if (k > t0) and (k <= t1)]
        for time in times:
            ticks = [tick for tick in token1Fees[time] if (tick>=tl) and (tick<tu)]
            for tick in ticks:
                posToken1Fees+=token1Fees[time][tick] * l

    return (posToken0Fees, posToken1Fees)

def pull_fees(token_id, mb, tl, tu, data, as_of = ''):
    '''
    Calculates the position in between all of the mint/burn events
    
    Though it is the same position, it occupies different levels
    of liquidity
    '''
    times = []
    for chunk in data:
        for token_0_1 in chunk:
            t = [t for t in token_0_1.keys()]
            times.extend(t)
            
    max_time = max(times)

    position = mb[mb['tokenID'] == token_id]
    position = position.sort_values(by ='block_idx')
    if as_of != '':
        position = position[position['block_idx'] < as_of]

    start_of_time = 0
    time_ranges = []
    for idx in position['block_idx']:
        if start_of_time == 0:
            start_of_time = idx
            continue

        time_ranges.append([start_of_time, idx])
        start_of_time = idx
    time_ranges.append([idx, max_time])

    fees = []
    for t0, t1 in time_ranges:
        l = position.loc[position['block_idx'] < t1, 'amount'].sum()
        if np.isclose(0, l):
            continue

        fees.append(price_fees_t0_to_t1(t0, t1, tl, tu, l, data))

    token0Fees = sum([_t[0] for _t in fees])
    token1Fees = sum([_t[1] for _t in fees])
    
    return (token0Fees, token1Fees)


def check_ticks(position):
    tl = position['tickLower'].unique()
    tu = position['tickUpper'].unique()

    assert tl.shape[0] == 1, 'Multiple tick-lowers'
    assert tl.shape[0] == 1, 'Multiple tick-uppers'

    tl = tl[0]
    tu = tu[0]
    
    return tl, tu

def calc_spot(cur_price, tl, tu, ts, l, bal_adj = 1e9):
    curToken0, curToken1 = 0, 0
    
    tickAt = np.log(cur_price ** 2) / np.log(1.0001)
    tickAt = (tickAt // ts) * ts
    
    p_b, p_a = np.sqrt(1.0001 ** tu), np.sqrt(1.0001 ** tl)

    if tu <= tickAt:
        curToken1 = (l * bal_adj) * (p_b - p_a)
    elif tl > tickAt:

        curToken0 = (l * bal_adj) * (p_b - p_a) / (p_a * p_b)
    else:
        curToken0 = (l * bal_adj) * (p_b - cur_price) / (cur_price * p_b)
        curToken1 = (l * bal_adj) * (cur_price - p_a)
    
    return (curToken0, curToken1)

def pull_position_data(token_id, mb, pool, data, as_of=np.inf):
    ts = pool.ts
    
    position = mb[mb['tokenID'] == token_id]
    cur_price = pool.getTickAt(np.inf, return_block = True)
    l = position['amount'].sum()

    curToken0 = 0
    curToken1 = 0

    tl, tu = check_ticks(position)
    cur_price, tickAt = pool.getPriceAt(as_of, return_block = True)

    if as_of == np.inf:
        as_of_time = tickAt
    else:
        as_of_time = as_of

    if np.isclose(l, 0):
        l = 0
    else:
        curToken0, curToken1 = calc_spot(cur_price, tl, tu, ts, l, bal_adj = pool.bal_adj)
        
    (fees0, fees1) = pull_fees(token_id, mb, tl, tu, data, as_of = as_of_time)        

    ret = {'curToken0': curToken0,
             'curToken1': curToken1,
             'liquidity': l,
             'as_of': as_of_time,
             'fees0': fees0,
             'fees1': fees1}

    return ret
