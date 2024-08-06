import pandas as pd
import numpy as np
import math
from scipy import sparse
import os

from ipdb import set_trace as bp


# liquidity code
def feeToTS(feetier):
    return {100: 1, 500: 10, 3000: 60, 10000: 200}.get(feetier)


def find_inc(max_inc):
    """
    Takes the max transaction index in any mint/burn transactions
    and finds the decimals needed to ensure that the transaction index
    is a decimal.

    ie if max_inc is 1100, then dec will be 4
    as 1100 / 10**4 = .1100

    # TODO
    Rework this to instead increase the blocksize to remain integers
    Also could find this using logs
    """
    dec = 0

    frac, whole = math.modf(max_inc / 10**dec)
    while whole != 0:
        dec += 1

        # we want to find the minimum power st max_inc / 10 ** power is only decimal
        frac, whole = math.modf(max_inc / 10**dec)

    return dec


def tick_to_initialized_tick(df, col, ts):
    """
    Ticks in Uniswap v3 have tick spacing. if tickspacing is 60, then ticks can only be
    every 60 indexes.

    To make the sparse matrix smaller, we divide the ticks by the tick spacing to
    make the matrix smaller. this is to remove the rows in-between tick spaces
    that have to be 0
    """
    # ive had one too many memory references
    df_copy = df.copy()
    df_copy["intermezzo"] = df_copy[col] / ts

    # ensure the positions are correctly specified (only would happen on a data error)
    err = f"Incorrect tick spacing of {ts} for lower"
    assert math.isclose(
        df_copy["intermezzo"].apply(lambda x: math.modf(x)[0]).sum(), 0
    ), err

    return df_copy["intermezzo"].astype(int)


def create_coo(mb, ts):
    """
    Shrink down the tick spacing then invert the liquidity on the upper tick
    as described in the v3 whitepaper.
    """
    mb["idxLower"] = tick_to_initialized_tick(mb, "tickLower", ts)
    mb["idxUpper"] = tick_to_initialized_tick(mb, "tickUpper", ts)

    liqLower = mb[["position_index", "idxLower", "amount"]].values
    liqUpper = mb[["position_index", "idxUpper", "amount"]].values

    # invert the liq
    liqUpper[:, 2] = liqUpper[:, 2] * -1

    coo = np.concatenate([liqLower, liqUpper])
    return coo


def create_distribution(mb, ts):
    """
    add together the block + the position in the block to get the
    block_idx. we then create a key value pair of the mint/burn index
    to the key value pair. this is because we need the column to be
    an integer, but we have to index using decimals. we also want
    the positions to be increasing integers so the sparse matrix
    is as small as possible

    TODO:
    Look into multiplying the block index by 10 ** dec instead of dividing so
    everything is an integer. probably will explode the matrix tho
    """
    dec = find_inc(mb["transaction_index"].max())
    mb["block_frac"] = mb["transaction_index"] / 10**dec
    mb["block_idx"] = mb["block_number"] + mb["block_frac"]
    mb["position_index"] = mb.index
    kv = {key: value for key, value in mb[["position_index", "block_idx"]].values}
    coo = create_coo(mb, ts)

    return coo, kv


def create_liq(ts, coordinate_payload="", mb="", as_of=""):
    """
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
    """
    if coordinate_payload == "":
        err = "Please provide a coordinate matrix/key-value or mint/burns"
        assert mb != "", err

        coo, kv = create_distribution(mb, ts)
    else:
        coo, kv = coordinate_payload

    if as_of:
        coordinates = np.vectorize(kv.get)(coo[:, 0])
        coo = coo[coordinates < as_of, :]

    col, row, amt = coo[:, 0].astype(int), coo[:, 1].astype(int), coo[:, 2]

    # indicies must be >= 0
    # ticks can be negative, so we need to add min_tick to it
    min_tick = row.min()
    row = row - min_tick

    mat = sparse.coo_matrix((amt, (row, col)))
    netLiquidity = mat.sum(axis=1)
    liquidity = netLiquidity.cumsum()

    ticks = np.arange(row.min(), row.max() + 1, 1)
    ticks = (ticks + min_tick) * ts

    x, y = ticks.tolist(), liquidity.tolist()[0]

    return x, y


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# underlying_x = [[x,y,ts, bal_adj] for (x, y) in zip(x, y)]


## swaps code
def ticks_to_underlying_x(x):
    tick, liq, ts, bal_adj = x[0], x[1], x[2], x[3]
    next_tick = tick + ts
    ratioA = np.sqrt(1.0001**next_tick)
    ratioB = np.sqrt(1.0001**tick)

    return liq * ((ratioB - ratioA) / (ratioB * ratioA)) * bal_adj


def get_amount0_delta(ratioA, ratioB, liq, bal_adj):
    if ratioA > ratioB:
        ratioA, ratioB = ratioB, ratioA

    return liq * ((ratioB - ratioA) / (ratioB * ratioA)) * bal_adj


def get_amount1_delta(ratioA, ratioB, liq, bal_adj):
    if ratioA > ratioB:
        ratioA, ratioB = ratioB, ratioA
    return liq * (ratioB - ratioA) * bal_adj


def get_next_price_amount0(ratioA, liq, amount, add, bal_adj):
    if add:
        sqrtPrice_trade = (liq * bal_adj * ratioA) / (liq * bal_adj + amount * ratioA)
    else:
        sqrtPrice_trade = (liq * bal_adj * ratioA) / (liq * bal_adj - amount * ratioA)

    return sqrtPrice_trade


def get_next_price_amount1(ratioA, liq, amount, add, bal_adj):
    if not add:
        sqrtPrice_trade = ratioA + amount / (liq * bal_adj)
    else:
        sqrtPrice_trade = ratioA - amount / (liq * bal_adj)

    return sqrtPrice_trade


def get_next_sqrtPrice(ratioA, liq, amount, zeroForOne, bal_adj):
    if zeroForOne:
        sqrtPrice_next = get_next_price_amount0(
            ratioA, liq, amount, zeroForOne, bal_adj
        )
    else:
        sqrtPrice_next = get_next_price_amount1(
            ratioA, liq, amount, zeroForOne, bal_adj
        )

    return sqrtPrice_next


def tick_to_px(x):
    return 1.0001**x


def get_next_tick(x_range):
    for tick in x_range:
        yield tick
