import pandas as pd
import numpy as np
import math
import os
from swap_utils import *
from pool_state import v3Pool
import time

class swap:
    def __init__(self, v3Pool):
        self.v3Pool = v3Pool
        self.swapDF = pd.DataFrame()
        self.swap_df_asof = 0

        self.bal_adj = self.v3Pool.bal_adj
    
    def createSwapDF(self, as_of):
        """
        Precompute the swap dataframe for the given liquidity distribution

        This is needed to accurately know what is in/out of range.

        TODO:
        Create the ability to pass this to the swapIn/pctOut functions
        to precompute needed values
        """
        x, y = self.v3Pool.createLiq(as_of)
        x, y = np.array(x), np.array(y)
        
        swap_df = pd.DataFrame(x, columns=["ticks"])
        swap_df["liquidity"] = y
        swap_df["ts"] = self.v3Pool.ts

        swap_df["p_b"] = np.sqrt(1.0001 ** (swap_df["ticks"] + swap_df["ts"]))
        swap_df["p_a"] = np.sqrt(1.0001 ** (swap_df["ticks"]))
        swap_df["yInTick"] = (
            swap_df["liquidity"] * (swap_df["p_b"] - swap_df["p_a"]) * self.bal_adj
        )
        
        swap_df["xInTick"] = (
            swap_df["liquidity"]
            * (swap_df["p_b"] - swap_df["p_a"])
            / (swap_df["p_b"] * swap_df["p_a"])
        ) * self.v3Pool.bal_adj

        sqrt_P = self.v3Pool.getPriceAt(as_of)
        swap_df["sqrt_P"] = sqrt_P

        swap_df["inRange0"] = 0.0
        swap_df["inRange1"] = 0.0

        # i dislike this fix - feels hacky
        swap_df = swap_df.drop_duplicates()
        # find the in-range tick
        inRangeCondition = (swap_df["p_a"] <= swap_df["sqrt_P"]) & (
            swap_df["p_b"] >= swap_df["sqrt_P"]
        )

        row = swap_df.loc[inRangeCondition]
        assert row.shape[0] == 1, "Duplicate in-range rows"

        p_a, p_b, sqrt_P, liquidity, tick = (
            row["p_a"].item(),
            row["p_b"].item(),
            row["sqrt_P"].item(),
            row["liquidity"].item(),
            row["ticks"].item(),
        )

        # get the values needed to trade in that range
        inRange0 = get_amount0_delta(p_a, sqrt_P, liquidity, self.bal_adj)
        inRangeToSwap0 = get_amount1_delta(p_a, sqrt_P, liquidity, self.bal_adj)

        inRange1 = get_amount1_delta(p_b, sqrt_P, liquidity, self.bal_adj)
        inRangeToSwap1 = get_amount0_delta(p_b, sqrt_P, liquidity, self.bal_adj)

        swap_df.loc[inRangeCondition, "inRange0"] = inRange0
        swap_df.loc[inRangeCondition, "inRange1"] = inRange1
        
        self.swap_df_asof = as_of
        
        return swap_df.copy(), (
            sqrt_P,
            inRange0,
            inRangeToSwap0,
            inRange1,
            inRangeToSwap1,
            liquidity,
            tick,
        )

    def swapIn(self, swapParams, swap_df="", inRangeValue=""):
        """
        TODO
        add heuristics to swaps
        add fees
        clean everything up

        Example:
        tokenIn = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        tokenOut = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
        swapIn = 100000 * 1e6 # 100k usdc
        as_of = 14814941

        swapParams = {'tokenIn': tokenIn,
                    'tokenOut': tokenOut,
                    'input': swapIn,
                    'as_of': as_of}

        out = pool.swapIn(swapParams)
        50.9929983049413 = out / 1e18
        """
        tokenIn = swapParams["tokenIn"]
        swapIn = swapParams["input"]
        as_of = swapParams["as_of"]
        
        # allows for pre-computing the swap_df
        if inRangeValue != "":
            assert not swap_df.empty, "Please give a valid swap_df"

        elif (not self.swapDF.empty) and (self.swap_df_asof == as_of):
            swap_df, inRangeValues = self.swapDF, self.inRangeValues

        else:
            swap_df, inRangeValues = self.createSwapDF(as_of)
        
        self.swapDF, self.inRangeValues = swap_df.copy(), inRangeValues
        # seperate the values tuple
        (
            sqrt_P,
            inRange0,
            inRangeToSwap0,
            inRange1,
            inRangeToSwap1,
            liquidity_in_range,
            tick_in_range,
        ) = inRangeValues
        bal_adj = self.v3Pool.bal_adj

        # this determines the direction of the swap
        if tokenIn == self.v3Pool.token1:
            zeroForOne = False
        else:
            zeroForOne = True

        # is there enough liquidity in the current tick?
        if zeroForOne:
            inRangeTest, inRangeToSwap = inRange0, inRangeToSwap0
        else:
            inRangeTest, inRangeToSwap = inRange1, inRangeToSwap1

        # account for fee
        swapInMinusFee = swapIn * (1 - self.v3Pool.fee)
        if inRangeTest > swapInMinusFee:
            # enough liquidity in range
            liquidity = liquidity_in_range

            # determine how far to push in-range
            if not zeroForOne:
                sqrtP_next = get_next_price_amount1(
                    sqrt_P, liquidity, swapInMinusFee, zeroForOne, self.bal_adj
                )
                amtOut = get_amount0_delta(sqrtP_next, sqrt_P, liquidity, self.bal_adj)
            else:
                sqrtP_next = get_next_price_amount0(
                    sqrt_P, liquidity, swapInMinusFee, zeroForOne, self.bal_adj
                )
                amtOut = get_amount1_delta(sqrtP_next, sqrt_P, liquidity, self.bal_adj)

            amtOutDelta = 0
            crossed_ticks = 0
            totalFee = swapIn * self.v3Pool.fee

        else:
            # find the first tick with enough liquidity in range
            leftToSwap = swapIn - inRangeTest

            # orient the liquidity distribution with the closest to middle at the top
            if zeroForOne:
                outOfRange = (
                    swap_df[swap_df["ticks"] < tick_in_range]
                    .sort_values(by="ticks", ascending=False)
                    .reset_index(drop=True)
                    .copy()
                )
            else:
                outOfRange = (
                    swap_df[swap_df["ticks"] > tick_in_range]
                    .sort_values(by="ticks", ascending=True)
                    .reset_index(drop=True)
                    .copy()
                )

            outOfRange["cumulativeY"] = outOfRange["yInTick"].cumsum()
            outOfRange["cumulativeX"] = outOfRange["xInTick"].cumsum()

            if zeroForOne:
                assert (
                    outOfRange["cumulativeX"].iloc[-1] > leftToSwap
                ), "Not enough liquidity in pool"
                liquid_tick = outOfRange[outOfRange["cumulativeX"] > leftToSwap].iloc[0]
                prev_ticks = outOfRange[outOfRange["ticks"] > liquid_tick["ticks"]]

            else:
                assert (
                    outOfRange["cumulativeY"].iloc[-1] > leftToSwap
                ), "Not enough liquidity in pool"
                liquid_tick = outOfRange[outOfRange["cumulativeY"] > leftToSwap].iloc[0]
                prev_ticks = outOfRange[outOfRange["ticks"] < liquid_tick["ticks"]]

            sqrt_P_last_top, sqrt_P_last_bottom = (
                liquid_tick["p_b"].item(),
                liquid_tick["p_a"].item(),
            )
            liquidity = liquid_tick["liquidity"].item()

            # calculate how much price impact the amount we are swapping in has
            # then based off that price impact, we calculate how much of the other side
            # we would have gotten
            if zeroForOne:
                amtInSwappedLeft = (leftToSwap - prev_ticks["xInTick"].sum()) 
                prevFee = (inRangeTest + prev_ticks["xInTick"].sum()) * self.v3Pool.fee

                amtInSwappedLeftMinusFee = amtInSwappedLeft * (1 - self.v3Pool.fee)
                amtOut = (inRangeToSwap + prev_ticks["yInTick"].sum()) * (1 - self.v3Pool.fee)

                sqrtP_next = get_next_price_amount0(
                    sqrt_P_last_top, liquidity, amtInSwappedLeftMinusFee, zeroForOne, bal_adj
                )
                amtOutDelta = get_amount1_delta(
                    sqrtP_next, sqrt_P_last_top, liquidity, bal_adj
                )

            else:
                amtInSwappedLeft = (leftToSwap - prev_ticks["yInTick"].sum())
                prevFee = (inRangeTest + prev_ticks["yInTick"].sum()) * self.v3Pool.fee
                
                amtInSwappedLeftMinusFee  = amtInSwappedLeft * (1 - self.v3Pool.fee)
                amtOut = (inRangeToSwap + prev_ticks["xInTick"].sum()) * (1 - self.v3Pool.fee)

                sqrtP_next = get_next_price_amount1(
                    sqrt_P_last_bottom, liquidity, amtInSwappedLeftMinusFee, zeroForOne, bal_adj
                )
                amtOutDelta = get_amount0_delta(
                    sqrtP_next, sqrt_P_last_bottom, liquidity, bal_adj
                )
            
            amtInFee = amtInSwappedLeft * self.v3Pool.fee
            totalFee = prevFee + amtInFee
            
            crossed_ticks = prev_ticks.shape[0]
            
        traded_out = amtOut + amtOutDelta
        traded_out = math.floor(traded_out)
        
        totalFee = math.ceil(totalFee)
      
        heur = [totalFee, crossed_ticks]

        return traded_out, heur 


    def swapToPrice(self, swapParams, swap_df = "", inRangeValue = ""):
        pcts = swapParams['pcts']
        tokenOut = swapParams['tokenOut']
        as_of = swapParams['as_of']

        # allows for pre-computing the swap_df
        if inRangeValue != "":
            assert not swap_df.empty, "Please give a valid swap_df"

        elif (not self.swapDF.empty) and (self.swap_df_asof == as_of):
            swap_df, inRangeValues = self.swapDF, self.inRangeValues

        else:
            swap_df, inRangeValues = self.createSwapDF(as_of)
        
        self.swapDF, self.inRangeValues = swap_df.copy(), inRangeValues

        (
            sqrt_P,
            inRange0,
            inRangeToSwap0,
            inRange1,
            inRangeToSwap1,
            liquidity_in_range,
            tick_in_range,
        ) = inRangeValues

        bal_adj = self.v3Pool.bal_adj
        px_cur = sqrt_P
        tick_cur = tick_in_range

        # not sure if this is needed
        cur_range = (tick_cur // self.v3Pool.ts) * self.v3Pool.ts
        
        inOrOutOfRange = ""
        data = []
        heur = []
        for pct in pcts:
            assert pct > -1, "Cannot be less than 100% drop"

            px_to = px_cur * np.sqrt(1 + pct)
            
            if pct < 0:
                lower = True
            else:
                lower = False
            zero = tokenOut == self.token0
            
            if lower:
                inRangeTest = np.sqrt(1.0001 ** cur_range)
                test_bool = inRangeTest <= px_to
            else:
                inRangeTest = np.sqrt(1.0001 ** (cur_range + self.v3Pool.ts))
                test_bool = inRangeTest >= px_to

            if test_bool:
                inOrOutOfRange = "IR"
                liquidity = liquidity_in_range
                if zero:
                    amtOut = get_amount0_delta(px_cur, px_to, liquidity, bal_adj)
                else:
                    amtOut = get_amount1_delta(px_cur, px_to, liquidity, bal_adj)

                amtPrev = 0
                inRangeAmt = 0

            else:
                inOrOutOfRange = "OOR"

                if lower:
                    min_tick = swap_df[(swap_df['p_a'] <= px_to)]['ticks'].max()

                    prev_ticks = swap_df[(swap_df['ticks'] > min_tick) &
                                        (swap_df['ticks'] < tick_in_range)]

                    liquid_tick = swap_df[swap_df['ticks'] == min_tick]
                    px_from = liquid_tick['p_b'].item()
                    liquidity = liquid_tick['liquidity'].item()
                    if zero:
                        amtOut = get_amount0_delta(px_from, px_to, liquidity, bal_adj)
                        amtPrev = prev_ticks['xInTick'].sum()
                        inRangeAmt = inRange0
                    else:
                        amtOut = get_amount1_delta(px_from, px_to, liquidity, bal_adj)
                        amtPrev = prev_ticks['yInTick'].sum()
                        inRangeAmt = inRangeToSwap0
                
                else:
                    max_tick = swap_df[(swap_df['p_b'] >= px_to)]['ticks'].min()

                    prev_ticks = swap_df[(swap_df['ticks'] < max_tick) &
                                        (swap_df['ticks'] > tick_in_range)]

                    liquid_tick = swap_df[swap_df['ticks'] == max_tick]
                    px_from = liquid_tick['p_a'].item()

                    if zero:
                        amtOut = get_amount0_delta(px_from, px_to, liquidity, bal_adj)
                        amtPrev = prev_ticks['xInTick'].sum()
                        inRangeAmt = inRangeToSwap1

                    else:
                        amtOut = get_amount1_delta(px_from, px_to, liquidity, bal_adj)
                        amtPrev = prev_ticks['yInTick'].sum()
                        inRangeAmt = inRange1

            swapToPct = amtOut + amtPrev + inRangeAmt

            data.append([swapToPct, pct])
            heur.append([inOrOutOfRange])
            
        return data, heur
           
