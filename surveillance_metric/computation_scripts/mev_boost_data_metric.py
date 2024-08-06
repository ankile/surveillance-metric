from glob import glob
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# current_path = sys.path[0]
# sys.path.append(current_path[: current_path.find("defi-measurement")] + "liquidity-distribution-history")

# sys.path.append("..")

import polars as pl

ROOT_PATH = Path(__file__).parent.parent.parent.absolute()
DATA_PATH = ROOT_PATH / "data_original"

print(f"ROOT_PATH: {ROOT_PATH}\nDATA_PATH: {DATA_PATH}")

from dataclasses import dataclass, field, asdict
from typing import List, Optional

from datetime import datetime, timezone
from itertools import permutations
from multiprocessing import Pool
from typing import Iterable, cast

from ipdb import set_trace as bp

from sim_original.pool_state import v3Pool

# from sqlalchemy import (
#     ARRAY,
#     CHAR,
#     Boolean,
#     Column,
#     DateTime,
#     Double,
#     Integer,
#     String,
#     create_engine,
# )
# from sqlalchemy.orm import declarative_base, sessionmaker
from tqdm import tqdm

import argparse

# If the `output` directory doesn't exist, create it
if not os.path.exists("output"):
    os.mkdir("output")

# load_dotenv()

# postgres_uri = os.environ["POSTGRESQL_URI"]
# azure_storage_uri = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

# Base = declarative_base()
# engine = create_engine(postgres_uri)
# SessionLocalMP = sessionmaker(bind=engine)


@dataclass
class BlockPoolMetrics:
    block_number: int
    pool_address: str
    num_transactions: int = 0
    n_buys: int = 0
    n_sells: int = 0
    baseline_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    mev_boost: bool = False
    mev_boost_amount: float = 0.0

    realized_order: str = None
    realized_prices: str = None
    realized_l1: float = 0.0
    realized_l2: float = 0.0
    realized_linf: float = 0.0

    volume_heur_order: str = None
    volume_heur_prices: str = None
    volume_heur_l1: float = 0.0
    volume_heur_l2: float = 0.0
    volume_heur_linf: float = 0.0

    tstar_l1: float = math.inf
    tstar_l2: float = math.inf
    tstar_linf: float = math.inf


def get_swaps_for_address(address: str, min_block: int, max_block: int) -> pl.DataFrame:
    # Create a lazy frame from all parquet files
    df = (
        pl.scan_parquet(DATA_PATH / "swaps", hive_partitioning=True)
        .filter((pl.col("address") == address))
        .filter(
            (pl.col("block_number") >= min_block)
            & (pl.col("block_number") <= max_block)
        )
    )

    # Collect and return the result
    return df.collect().to_pandas()


def get_token_info():
    # Load the data from the Parquet file
    df = pl.scan_parquet(DATA_PATH / "pool_token_info.parquet")

    # Select the required columns
    token_info = (
        df.select(
            [
                pl.col("pool"),
                pl.col("token0"),
                pl.col("token1"),
                pl.col("decimals0"),
                pl.col("decimals1"),
            ]
        )
        .collect()
        .to_pandas()
        .set_index("pool")[["token0", "token1", "decimals0", "decimals1"]]
    )

    return token_info.to_dict(orient="index")


def get_mev_boost_values() -> dict[int, float]:
    df = (
        pl.scan_parquet(DATA_PATH / "mev-boost" / "openethdata_eth_data_clean.parquet")
        .select(["block_number", "mevboost_value"])
        .collect()
        .to_pandas()
    )

    # Select the relevant columns
    return dict(zip(df.block_number, df.mevboost_value))


def get_pool_block_pairs(*, limit, offset, only_unprocessed) -> pd.DataFrame:
    # Load DataFrames lazily
    swap_counts = pl.scan_parquet(DATA_PATH / "swap_counts.parquet")
    token_info = pl.scan_parquet(DATA_PATH / "pool_token_info.parquet")

    # Check if block_pool_metrics.parquet exists
    block_pool_metrics_path = DATA_PATH / "pool_block_metrics/*.parquet"
    block_pool_metrics: Optional[pl.LazyFrame] = None

    if len(glob(str(block_pool_metrics_path))) > 0:
        block_pool_metrics = pl.scan_parquet(block_pool_metrics_path)
    elif only_unprocessed:
        print(
            "Warning: only_unprocessed is True, but pool_block_metrics/*.parquet doesn't exist. Returning all pool-block pairs."
        )

    # Start building the query
    query = (
        swap_counts.filter(
            (pl.col("block_number") >= 15537940) & (pl.col("block_number") <= 17959956)
        )
        .join(
            token_info.filter(
                pl.col("decimals0").is_not_null() & pl.col("decimals1").is_not_null()
            ),
            left_on="address",
            right_on="pool",
        )
        .select(["address", "block_number"])
    )

    if only_unprocessed and block_pool_metrics is not None:
        query = (
            query.join(
                block_pool_metrics,
                left_on=["address", "block_number"],
                right_on=["pool_address", "block_number"],
                how="left",
            )
            .filter(pl.col("num_transactions").is_null())
            .select(["address", "block_number"])
        )

    # Apply ordering, limit, and offset
    # query = query.sort(["address", "block_number"]).slice(offset, limit)

    # Count unique block numbers for each address
    address_block_counts = query.group_by("address").agg(
        pl.col("block_number").n_unique().alias("unique_block_count")
    )

    # Join the counts back to the original query
    query = query.join(address_block_counts, on="address", how="left")

    # Apply ordering (by unique block count descending, then address, then block number),
    # limit, and offset
    query = query.sort(
        ["address", "block_number"],
        descending=[False, False],
    ).slice(offset, limit)

    # Collect the results
    return query.collect().to_pandas()


def get_price(sqrt_price, pool_addr, token_info):
    return (
        1
        / (sqrt_price**2)
        / 10
        ** (token_info[pool_addr]["decimals0"] - token_info[pool_addr]["decimals1"])
    )


def get_pool(address):
    return v3Pool(
        poolAdd=address,
        initialize=False,
        verbose=True,
        bal_adj=1e9,
        update=False,
        chunk_length=5e5,
    )


def norm(prices, norm):
    if norm == 1:
        return float(np.sum(np.abs(prices)))
    elif norm == 2:
        return float(np.sqrt(np.sum(prices**2)))
    elif norm == np.inf:
        return float(np.max(np.abs(prices)))
    else:
        raise ValueError("Invalid norm")


def do_swap(swap, curr_price, pool, token_info):
    token_in = (
        token_info[swap.address]["token0"]
        if int(swap.amount0) > 0
        else token_info[swap.address]["token1"]
    )
    input_amount = int(swap.amount0) if int(swap.amount0) > 0 else int(swap.amount1)

    _, heur = pool.swapIn(
        {
            "tokenIn": token_in,
            "input": input_amount,
            "as_of": swap.block_number,
            "gas": True,
            "givenPrice": curr_price,
        }
    )

    return heur


def get_pool_block_count(*, only_unprocessed: bool) -> tuple[int, int]:
    # Load DataFrames lazily
    swap_counts = pl.scan_parquet(DATA_PATH / "swap_counts.parquet")
    token_info = pl.scan_parquet(DATA_PATH / "pool_token_info.parquet")

    # Check if block_pool_metrics.parquet exists
    block_pool_metrics_path = DATA_PATH / "pool_block_metrics/*.parquet"
    block_pool_metrics: Optional[pl.LazyFrame] = None

    if len(glob(str(block_pool_metrics_path))) > 0:
        block_pool_metrics = pl.scan_parquet(block_pool_metrics_path)
    else:
        print("pool_block_metrics/*.parquet not found. Proceeding without it.")

    # Start building the query
    query = swap_counts.filter(
        (pl.col("block_number") >= 15537940) & (pl.col("block_number") <= 17959956)
    ).join(
        token_info.filter(
            pl.col("decimals0").is_not_null() & pl.col("decimals1").is_not_null()
        ),
        left_on="address",
        right_on="pool",
    )

    total_count = query.select(pl.len()).collect().item()

    if only_unprocessed and block_pool_metrics is not None:
        query = query.join(
            block_pool_metrics,
            left_on=["address", "block_number"],
            right_on=["pool_address", "block_number"],
            how="left",
        ).filter(pl.col("num_transactions").is_null())
    elif only_unprocessed and block_pool_metrics is None:
        print(
            "Warning: only_unprocessed is True, but pool_block_metrics/*.parquet doesn't exist. Returning all pool-block pairs."
        )

    # Count the rows
    remaining_count = query.select(pl.len()).collect().item()

    return remaining_count, total_count


def set_metrics(
    blockpool_metric: BlockPoolMetrics, field: str, prices: list, ordering: list
):
    assert field in ["realized", "volume_heur"]
    setattr(blockpool_metric, f"{field}_prices", ",".join(map(str, prices)))  # type: ignore
    setattr(blockpool_metric, f"{field}_order", ",".join(ordering))  # type: ignore

    prices_np = np.array(prices) - blockpool_metric.baseline_price
    setattr(blockpool_metric, f"{field}_l1", norm(prices_np, 1))  # type: ignore
    setattr(blockpool_metric, f"{field}_l2", norm(prices_np, 2))  # type: ignore
    setattr(blockpool_metric, f"{field}_linf", norm(prices_np, np.inf))  # type: ignore


def run_swap_order(pool: v3Pool, swaps: Iterable, block_number: int, token_info):
    prices = []
    ordering = []
    curr_price_sqrt = pool.getPriceAt(block_number)

    for swap in swaps:
        heur = do_swap(swap, curr_price_sqrt, pool, token_info)

        prices.append(get_price(heur.sqrtP_next, swap.address, token_info))
        ordering.append(f"{swap.transaction_index:03}_{swap.log_index:03}")
        curr_price_sqrt = heur.sqrtP_next

    return prices, ordering


def realized_measurement(
    pool: v3Pool,
    swaps: pd.DataFrame,
    block_number: int,
    blockpool_metric: BlockPoolMetrics,
    token_info: dict,
):
    # Run the realized measurement
    prices, ordering = run_swap_order(
        pool, swaps.itertuples(index=False, name="Swap"), block_number, token_info
    )

    set_metrics(blockpool_metric, "realized", prices, ordering)


def volume_heuristic(
    pool: v3Pool,
    swaps: pd.DataFrame,
    block_number: int,
    blockpool_metric: BlockPoolMetrics,
    token_info: dict,
):
    pool_addr = blockpool_metric.pool_address
    baseline_price = blockpool_metric.baseline_price

    # Run the volume heuristic measurement
    curr_price_sqrt = cast(float, pool.getPriceAt(block_number))
    curr_price = get_price(curr_price_sqrt, pool_addr, token_info)

    prices = []
    ordering = []

    # Split the swaps into the set of buys and sells and order by volume ascending
    swaps = swaps.assign(
        amount0_float=swaps.amount0.astype(float),
        amount1_float=swaps.amount1.astype(float),
    )
    buy_df = swaps[~swaps.amount0.str.startswith("-")]
    sell_df = swaps[swaps.amount0.str.startswith("-")]
    buys = (
        [
            row
            for _, row in buy_df.sort_values(
                "amount0_float", ascending=False
            ).iterrows()
        ]
        if buy_df.shape[0] > 0
        else []
    )
    sells = (
        [
            row
            for _, row in sell_df.sort_values(
                "amount1_float", ascending=False
            ).iterrows()
        ]
        if sell_df.shape[0] > 0
        else []
    )

    # While we're still in the core
    while len(buys) > 0 and len(sells) > 0:
        if curr_price == baseline_price:
            # If we're at the baseline price, we can swap in either direction
            # Choose the one that moves the price the least
            buy_diff = (
                get_price(
                    do_swap(buys[-1], curr_price_sqrt, pool, token_info).sqrtP_next,
                    pool_addr,
                    token_info,
                )
                - baseline_price
            )
            sell_diff = (
                get_price(
                    do_swap(sells[-1], curr_price_sqrt, pool, token_info).sqrtP_next,
                    pool_addr,
                    token_info,
                )
                - baseline_price
            )

            if abs(buy_diff) < abs(sell_diff):
                swap = buys.pop(-1)
            else:
                swap = sells.pop(-1)
        elif curr_price <= baseline_price:
            swap = buys.pop(-1)
        else:
            swap = sells.pop(-1)

        heur = do_swap(swap, curr_price_sqrt, pool, token_info)

        curr_price_sqrt = heur.sqrtP_next
        curr_price = get_price(curr_price_sqrt, swap.address, token_info)
        prices.append(curr_price)
        ordering.append(f"{swap.transaction_index:03}_{swap.log_index:03}")

    # Process whatever is left in the tail
    for swap in (buys + sells)[::-1]:
        heur = do_swap(swap, curr_price_sqrt, pool, token_info)

        curr_price_sqrt = heur.sqrtP_next
        prices.append(get_price(curr_price_sqrt, swap.address, token_info))
        ordering.append(f"{swap.transaction_index:03}_{swap.log_index:03}")

    set_metrics(blockpool_metric, "volume_heur", prices, ordering)


def tstar(
    pool: v3Pool,
    swaps: pd.DataFrame,
    block_number: int,
    blockpool_metric: BlockPoolMetrics,
    token_info: dict,
):
    # Run the t* measurement if not more than 7 swaps
    if swaps.shape[0] > 7:
        return

    best_scores = {
        "l1": math.inf,
        "l2": math.inf,
        "linf": math.inf,
    }
    for swap_perm in permutations(swaps.itertuples(index=False, name="Swap")):
        prices, _ = run_swap_order(pool, swap_perm, block_number, token_info)
        best_scores["l1"] = min(
            best_scores["l1"],
            norm(np.array(prices) - blockpool_metric.baseline_price, 1),
        )
        best_scores["l2"] = min(
            best_scores["l2"],
            norm(np.array(prices) - blockpool_metric.baseline_price, 2),
        )
        best_scores["linf"] = min(
            best_scores["linf"],
            norm(np.array(prices) - blockpool_metric.baseline_price, np.inf),
        )

    blockpool_metric.tstar_l1 = best_scores["l1"]  # type: ignore
    blockpool_metric.tstar_l2 = best_scores["l2"]  # type: ignore
    blockpool_metric.tstar_linf = best_scores["linf"]  # type: ignore


def copy_over(blockpool_metric: BlockPoolMetrics, to: list[str]):
    for field in to:
        setattr(blockpool_metric, f"{field}_prices", blockpool_metric.realized_prices)
        setattr(blockpool_metric, f"{field}_order", blockpool_metric.realized_order)
        setattr(blockpool_metric, f"{field}_l1", blockpool_metric.realized_l1)
        setattr(blockpool_metric, f"{field}_l2", blockpool_metric.realized_l2)
        setattr(blockpool_metric, f"{field}_linf", blockpool_metric.realized_linf)


def run_metrics(
    limit,
    offset,
    process_id,
    token_info,
    mev_boost_values,
    only_unprocessed,
    t_star_max_swaps,
):

    def write_buffer():
        nonlocal buffer
        if buffer:
            df = pl.DataFrame(buffer)
            if os.path.exists(output_file):
                existing_df = pl.read_parquet(output_file)
                df = pl.concat([existing_df, df])
            df.write_parquet(output_file)
            buffer = []

    output_file = (
        DATA_PATH
        / "pool_block_metrics"
        / f"block_pool_metrics_{offset}-{offset+limit}_{datetime.now()}.parquet"
    )

    # Ensure the path exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pool_block_pairs = get_pool_block_pairs(
        limit=limit, offset=offset, only_unprocessed=only_unprocessed
    )

    it = tqdm(
        total=pool_block_pairs.shape[0],
        position=process_id,
        desc=f"[{process_id}] ({offset}-{offset+limit})",
        smoothing=0,
    )
    pool = None

    program_start = datetime.now()

    errors = 0
    successes = 0
    skipped_bc_empty = 0
    buffer = []

    for pool_addr, df in pool_block_pairs.groupby("address"):
        it.set_description(
            f"[{process_id}] ({offset}-{offset+limit}) Processing pool {pool_addr}"
        )

        if pool_addr not in token_info:
            continue

        try:
            if pool is None or pool_addr != pool.pool:
                pool = get_pool(pool_addr)

            swaps_for_pool = get_swaps_for_address(
                pool_addr, df.block_number.min(), df.block_number.max()
            )

            for block_number in df.block_number.unique():
                block_number = int(block_number)
                it.set_postfix(
                    errors=errors,
                    successes=successes,
                    skipped_bc_empty=skipped_bc_empty,
                )
                it.update(1)

                swaps = swaps_for_pool[
                    swaps_for_pool.block_number == block_number
                ].sort_values("transaction_index")

                if swaps.shape[0] == 0:
                    skipped_bc_empty += 1
                    continue

                curr_price_sqrt = pool.getPriceAt(block_number)

                blockpool_metric = BlockPoolMetrics(
                    block_number=block_number,
                    pool_address=pool_addr,
                    num_transactions=swaps.shape[0],
                    n_buys=swaps[~swaps.amount0.str.startswith("-")].shape[0],
                    n_sells=swaps[swaps.amount0.str.startswith("-")].shape[0],
                    mev_boost=block_number in mev_boost_values,
                    mev_boost_amount=mev_boost_values.get(block_number, 0),
                    baseline_price=get_price(curr_price_sqrt, pool_addr, token_info),
                )

                # Run the baseline measurement
                realized_measurement(
                    pool, swaps, block_number, blockpool_metric, token_info
                )

                if swaps.shape[0] > 1:
                    volume_heuristic(
                        pool, swaps, block_number, blockpool_metric, token_info
                    )

                    if swaps.shape[0] <= t_star_max_swaps:
                        tstar(pool, swaps, block_number, blockpool_metric, token_info)
                else:
                    copy_over(blockpool_metric, to=["volume_heur", "tstar"])

                buffer.append(asdict(blockpool_metric))

                # Write to Parquet file every 100 rows or at the end
                if len(buffer) >= 100:
                    write_buffer()

                successes += 1

        except Exception as e:
            errors += 1
            with open(f"output/error-{program_start}.log", "a") as f:
                f.write(
                    f"Error processing block {block_number} for pool {pool_addr}: {e}\n"
                )
            continue


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Calculate MEV Boost Data Metrics")
    # parser.add_argument("--n-cpus", type=int, default=1, help="Number of CPUs to use")
    # args = parser.parse_args()

    parser = argparse.ArgumentParser(description="Calculate MEV Boost Data Metrics")
    parser.add_argument(
        "--n-partitions",
        "-n",
        type=int,
        required=True,
        help="Total number of partitions",
    )
    parser.add_argument(
        "--partition-index",
        "-i",
        type=int,
        required=True,
        help="Index of this partition (0-based)",
    )
    args = parser.parse_args()

    only_unprocessed = True

    # print(f"Starting MEV Boost Data Metric Calculations with {args.n_cpus} CPUs")
    print(
        f"Starting MEV Boost Data Metric Calculations for partition {args.partition_index + 1} of {args.n_partitions}"
    )

    n_pool_block_pairs, total_pairs = get_pool_block_count(
        only_unprocessed=only_unprocessed
    )
    print(f"Processing {n_pool_block_pairs:,} pool-block pairs, out of {total_pairs:,}")

    mev_boost_values = get_mev_boost_values()
    token_info = get_token_info()

    # n_processes = args.n_cpus

    # # Calculate the chunk size
    # chunk_size = n_pool_block_pairs // n_processes

    # Calculate the chunk size and offset for this partition
    chunk_size = n_pool_block_pairs // args.n_partitions
    remainder = n_pool_block_pairs % args.n_partitions

    # Distribute the remainder across partitions
    if args.partition_index < remainder:
        chunk_size += 1
        offset = args.partition_index * chunk_size
    else:
        offset = (args.partition_index * chunk_size) + remainder

    print(f"Processing chunk size: {chunk_size}, offset: {offset}")

    run_metrics(
        limit=chunk_size,
        offset=offset,
        process_id=args.partition_index,
        token_info=token_info,
        mev_boost_values=mev_boost_values,
        only_unprocessed=only_unprocessed,
        t_star_max_swaps=4,
        # pull_latest_data=True,
        # reraise_exceptions=False,  # Set to True to debug
    )

    print(f"Partition {args.partition_index + 1} of {args.n_partitions} completed.")

    # # Define a function to be mapped
    # def run_chunk(i):
    #     offset = i * chunk_size
    #     run_metrics(
    #         limit=chunk_size,
    #         offset=offset,
    #         process_id=i,
    #         token_info=token_info,
    #         mev_boost_values=mev_boost_values,
    #         only_unprocessed=only_unprocessed,
    #         # pull_latest_data=True,
    #         # reraise_exceptions=False,  # Set to True to debug
    #     )

    # if n_processes == 1:
    #     run_chunk(0)
    # else:
    #     # Create a pool of workers and map the function across the input values
    #     with Pool(n_processes) as pool:
    #         pool.map(run_chunk, range(n_processes))

    # print("All processes completed.")
