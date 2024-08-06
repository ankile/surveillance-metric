from pathlib import Path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
import matplotlib.animation as animation
import os
from sim_original.swap_utils import *
import time
import polars as pl

from typing import Tuple

from ipdb import set_trace as bp

from collections import namedtuple

DATA_BASE_DIR = Path(__file__).absolute().parent.parent / "data_original"

Heur = namedtuple(
    "Heur",
    [
        "totalFee",
        "crossed_ticks",
        "liquidity_in_range",
        "sqrt_P",
        "sqrtP_next",
        "tickToFees",
        "inRangeTest",
        "swapInMinusFee",
        "zeroForOne",
        "gas_fee",
    ],
)

# import psycopg2
# from sqlalchemy import create_engine

# from google.cloud import bigquery
from collections import defaultdict


class v3Pool:
    """
    Creates and maintains a representation of a Uniswap v3 pool.

    TODO
    Current:
    Optimize data loading for swaps and mint/burns [Done]
    Automatic updating from big query [Done]
    Add swapping heuristics
    Add upToPrice order type [in progress]
    Optimize the speed of calculating upToPrice/swapIn
    Combine the redundant code between these types

    Eventual:
    Automatic boostrapping from google big query [Done]
    Fee accounting
    Changing state
    Mint/burns
    New order types
    """

    def __init__(
        self,
        poolAdd,
        connStr="postgres://postgres:postgrespw@localhost:55000",
        proj_id="mimetic-design-338620",
        initialize=True,
        mem_increase=2,
        load_data=True,
        verbose=True,
        bal_adj=1e9,
        update=False,
        chunk_length=5e5,
        delete_conn=False,
    ):
        # values to adjust behavior
        self.proj_id = proj_id
        self.verbose = verbose
        self.loadData = load_data
        self.update = update
        self.chunk_length = chunk_length

        assert not update, "Cannot update the database"
        assert not initialize, "Cannot initialize the database, already initialized"

        # make sure the pool is lower for gcp str matching
        lowerCasePool = poolAdd.lower()
        if lowerCasePool != poolAdd and self.verbose:
            print("Changing pool to lowercase")
        self.pool = lowerCasePool

        # self.engine = create_engine(connStr)
        # self.conn = psycopg2.connect(connStr)
        # self.cur = self.conn.cursor()

        # internal data warehousing
        self.bal_adj = bal_adj
        self.nfp = "0xc36442b4a4522e871399cd717abdd847ab11fe88"

        # if mem_increase != "" and initialize:
        #     """
        #     Since we are running on good machines, we can tune the postgres server to
        #     do all of its calculations in memory instead of on disk. this allows for
        #     faster sorting.

        #     We also want to create indexes so scanning the database happens quicker
        #     esp as the database grows in-size
        #     """
        #     self.cur.execute(f"SET work_mem TO '{mem_increase} GB';")
        #     self.conn.commit()

        #     self.cur.execute(f"SET maintenance_work_mem TO '1 GB';")
        #     self.conn.commit()

        # locks to accessing data that is not loaded correctly
        self.swapLoad = False
        self.mbLoad = False
        self.cooLoad = False
        self.infoLoad = False
        self.iniLoad = False

        self.coo = None
        self.swapDF = pd.DataFrame()
        self.swap_df_asof = 0

        # if initialize:
        #     self._establishSchema()

        if self.loadData:
            self.prepareData()

        self._createPoolState()

        # # Close the database connection after pool state is created
        # if delete_conn:
        #     self.conn.close()
        #     del self.conn
        #     self.cur.close()
        #     del self.cur
        #     self.engine.dispose()
        #     del self.engine

    # def _dropAll(self, subset=[]):
    #     if subset == []:
    #         print("Dropping all databases - waiting 15s")
    #         time.sleep(15)

    #         self.cur.execute("DROP TABLE swaps")
    #         self.conn.commit()

    #         self.cur.execute("DROP TABLE mb")
    #         self.conn.commit()

    #         self.cur.execute("DROP TABLE collects")
    #         self.conn.commit()

    #         self.cur.execute("DROP TABLE block_info")
    #         self.conn.commit()

    #         self.cur.execute("DROP TABLE factory")
    #         self.conn.commit()

    #         self.cur.execute("DROP TABLE initialize")
    #         self.conn.commit()

    #     else:
    #         for table in subset:
    #             print(f"Dropping {table} databases - waiting 5s")
    #             time.sleep(5)

    #             self.cur.execute(f"DROP TABLE {table}")
    #             self.conn.commit()

    def _verbosePrint(self, msg):
        if self.verbose:
            print(msg)

    def _createPoolState(self):
        """
        Clean and define the pool state from the factory
        """

        # factory = pd.read_sql(
        #     f"SELECT * from factory where pool = '{self.pool}';", self.engine
        # )
        factory_data = (
            pl.scan_parquet(DATA_BASE_DIR / "factory/*.parquet")
            .filter(pl.col("pool") == self.pool)
            .collect()
            .to_pandas()
        )

        # factory_data = factory[factory["pool"] == self.pool]

        assert not factory_data.empty, "Pool is missing from factory"

        # these are the state variables needed for the pool to function normally
        self.fee = float(factory_data["fee"].item()) / 1e6
        # bp()
        self.ts = float(factory_data["tickSpacing"].item())
        self.starting_block = int(factory_data["block_number"].item())
        self.starting_time = pd.to_datetime(factory_data["block_timestamp"].item())
        self.token0 = factory_data["token0"].item()
        self.token1 = factory_data["token1"].item()

    def _establishSchema(self):
        # creates the mb schema if it doesn't exist
        self._verbosePrint("Establishing schema")

        factory_schema = """CREATE TABLE IF NOT EXISTS factory
                          (block_timestamp timestamp, block_number integer, tx_hash text, log_index integer,
                           token0 text, token1 text, fee integer, tickSpacing integer, pool text);"""

        self.cur.execute(factory_schema)
        self.conn.commit()

        block_schema = """CREATE TABLE IF NOT EXISTS block_info
                          (block_number numeric, block_timestamp timestamp, min_gas numeric, pct_25_gas numeric,
                           median_gas numeric, mean_gas numeric, pct_75_gas numeric, max_gas numeric);"""

        self.cur.execute(block_schema)
        self.conn.commit()

        initialize_schema = """CREATE TABLE IF NOT EXISTS initialize
                          (block_timestamp timestamp, block_number numeric, tx_hash text, log_index numeric,
                          sqrtPriceX96 text, tick numeric, address text);"""

        self.cur.execute(initialize_schema)
        self.conn.commit()

        collects_schema = """CREATE TABLE IF NOT EXISTS collects
                          (block_timestamp timestamp, block_number numeric, tx_hash text, log_index numeric,
                          tokenID numeric, recipient text, amount0 text, amount1 text);"""

        self.cur.execute(collects_schema)
        self.conn.commit()

        mb_schema = """CREATE TABLE IF NOT EXISTS mb
                    (block_timestamp timestamp, block_number integer, tx_hash text, log_index integer,
                    address text, owner text, tickLower integer, tickUpper integer, amount text,
                    amount0 text, amount1 text, tokenID numeric, type integer, to_address text, from_address text,
                    transaction_index integer);"""

        self.cur.execute(mb_schema)
        self.conn.commit()

        swaps_schema = """CREATE TABLE IF NOT EXISTS swaps
                          (block_timestamp timestamp, block_number integer, tx_hash text, log_index integer,
                           sender text, recipient text, amount0 text, amount1 text, sqrtPriceX96 text,
                           liquidity text, tick integer, address text, to_address text, from_address text,
                           transaction_index integer);"""

        self.cur.execute(swaps_schema)
        self.conn.commit()

        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_add1 ON swaps (address);")
        self.conn.commit()

        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_add2 ON mb (address);")
        self.conn.commit()

        # fill the databases
        self._fillDatabases()

    def _pullFromBQ(self, hosted_db, local_db, fill):
        # TODO
        # break the swap/mint if into pieces
        # to work on the factory pulls

        if hosted_db in ["swap", "MintBurnV3-labeled"]:
            if fill:
                self._verbosePrint(f"Selecting from {hosted_db}")

                q = f"""select * FROM `{self.proj_id}.uniswap.{hosted_db}` where address = "{self.pool}" """
                df = pd.io.gbq.read_gbq(q, project_id=self.proj_id, dialect="standard")

            elif self.update:
                self._verbosePrint(f"Updating from {hosted_db}")

                self.cur.execute(
                    f"SELECT max(block_number) from {local_db} where address = '{self.pool}'"
                )
                max_block = self.cur.fetchone()
                max_block = max_block[0]
                assert max_block != None, f"Empty local database {local_db}"

                q = f"""select * FROM `{self.proj_id}.uniswap.{hosted_db}`
                        where address = "{self.pool}"
                        and block_number > {max_block} """

                df = pd.io.gbq.read_gbq(q, project_id=self.proj_id, dialect="standard")

                if df.empty:
                    self._verbosePrint(
                        f"Nothing to update for {local_db} above block {max_block}"
                    )
            else:
                return pd.DataFrame()

        elif hosted_db in [
            "V3Factory_PoolCreated",
            "ethereum_block_heuristics",
            "ethereum_uniswap_v3_pool_evt_initialize",
            "collect_all",
        ]:
            if fill:
                self._verbosePrint(f"Selecting from {hosted_db} for {self.pool}")

                q = f"""select * FROM `{self.proj_id}.uniswap.{hosted_db}`"""
                df = pd.io.gbq.read_gbq(q, project_id=self.proj_id, dialect="standard")

            elif self.update:
                self._verbosePrint(f"Updating from {hosted_db} for {self.pool}")

                self.cur.execute(f"SELECT max(block_number) from {local_db} ")
                max_block = self.cur.fetchone()
                max_block = max_block[0]
                assert max_block != None, f"Empty local database {local_db}"

                q = f"""select * FROM `{self.proj_id}.uniswap.{hosted_db}`
                        where block_number > {max_block} """
                df = pd.io.gbq.read_gbq(q, project_id=self.proj_id, dialect="standard")

                if df.empty:
                    self._verbosePrint(
                        f"Nothing to update for {local_db} above block {max_block}"
                    )
            else:
                return pd.DataFrame()

        else:
            if self.verbose:
                print("Returning empty df")

            return pd.DataFrame()

        return df

    def _insertIntoDB(self, df, db, insertStr):
        """
        heavily optimized (50-100x speed up) sql insert to by-pass some of the
        safety checks and the executemany command is just a for loop in python.

        this is a vectorized insert.
        """
        arr = [tuple(arr.tolist()) for arr in df.values]
        args = [self.cur.mogrify(insertStr, x).decode("utf-8") for x in arr]
        for count, sub_str in enumerate(chunks(args, int(self.chunk_length))):
            self._verbosePrint(
                f"Inserting chunk number {count} of size {self.chunk_length}"
            )
            args_str = ", ".join(sub_str)

            self.cur.execute(f"INSERT INTO {db} VALUES" + args_str)
            self.conn.commit()

    def _fillOrUpdateMB(self, fill):
        mb = self._pullFromBQ("MintBurnV3-labeled", "mb", fill)
        if mb.empty:
            print("Empty BigQuery")
            return

        # there is a possible non-overlap in the three databases updating
        # at the exact same time

        missing_nfp = (
            mb[(mb["owner"] == self.nfp) & (mb["tokenID"].isna())]["block_number"].min()
            - 1
        )

        # clean the data
        mb["block_timestamp"] = pd.to_datetime(mb["block_timestamp"])
        mb["block_timestamp"] = mb["block_timestamp"].dt.tz_localize(None)
        mb["tickLower"] = mb["tickLower"].astype(int)
        mb["tickUpper"] = mb["tickUpper"].astype(int)
        mb["amount"] = mb["amount"].astype(str)
        mb["amount0"] = mb["amount0"].astype(str)
        mb["amount1"] = mb["amount1"].astype(str)
        mb["block_timestamp"] = mb["block_timestamp"].dt.to_pydatetime()

        if not pd.isna(missing_nfp):
            self._verbosePrint(f"Selecting as-of {missing_nfp} block")
            mb = mb[mb["block_number"] <= missing_nfp].copy()

        self._verbosePrint("Inserting mintburn into database")

        self._insertIntoDB(
            mb, "mb", "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )

    def _fillOrUpdateSwaps(self, fill):
        swaps = self._pullFromBQ("swap", "swaps", fill)
        if swaps.empty:
            return

        # clean the data
        swaps["block_timestamp"] = pd.to_datetime(swaps["block_timestamp"])
        swaps["block_timestamp"] = swaps["block_timestamp"].dt.tz_localize(None)
        swaps["block_timestamp"] = swaps["block_timestamp"].dt.to_pydatetime()

        self._verbosePrint("Inserting swaps into database")

        self._insertIntoDB(
            swaps,
            "swaps",
            "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        )

    def _fillOrUpdateFactory(self, fill):
        factory = self._pullFromBQ("V3Factory_PoolCreated", "factory", fill)
        if factory.empty:
            return

        self._verbosePrint("Inserting factory into database")

        self._insertIntoDB(factory, "factory", "(%s, %s, %s, %s, %s, %s, %s, %s, %s)")

    def _fillOrUpdateIni(self, fill):
        pool_ini = self._pullFromBQ(
            "ethereum_uniswap_v3_pool_evt_initialize", "initialize", fill
        )
        if pool_ini.empty:
            return

        pool_ini["block_timestamp"] = pd.to_datetime(pool_ini["block_timestamp"])
        pool_ini["block_timestamp"] = pool_ini["block_timestamp"].dt.tz_localize(None)
        pool_ini["block_timestamp"] = pool_ini["block_timestamp"].dt.to_pydatetime()

        self._verbosePrint("Inserting initialization into database")

        self._insertIntoDB(pool_ini, "initialize", "(%s, %s, %s, %s, %s, %s, %s)")

    def _fillOrUpdateCollects(self, fill):
        collects = self._pullFromBQ("collect_all", "collects", fill)
        if collects.empty:
            return

        collects["block_timestamp"] = pd.to_datetime(collects["block_timestamp"])
        collects["block_timestamp"] = collects["block_timestamp"].dt.tz_localize(None)
        collects["block_timestamp"] = collects["block_timestamp"].dt.to_pydatetime()

        self._verbosePrint("Inserting collects into database")

        self._insertIntoDB(collects, "collects", "(%s, %s, %s, %s, %s, %s, %s, %s)")

    def _fillOrUpdateBlockInfo(self, fill):
        block_info = self._pullFromBQ("ethereum_block_heuristics", "block_info", fill)
        if block_info.empty:
            return

        block_info["block_timestamp"] = pd.to_datetime(block_info["block_timestamp"])
        block_info["block_timestamp"] = block_info["block_timestamp"].dt.tz_localize(
            None
        )
        block_info["block_timestamp"] = block_info["block_timestamp"].dt.to_pydatetime()
        self._verbosePrint("Inserting block_info into database")

        self._insertIntoDB(block_info, "block_info", "(%s, %s, %s, %s, %s, %s, %s, %s)")

    def _fillDatabases(self):
        # factory is the most basic values that are always needed
        # factory is always needed so we require it to be pulled
        # TODO
        # break these down to work for all the databases

        test = "factory"
        self.cur.execute(f"SELECT * from {test}")

        if self.cur.fetchone() == None:
            self._verbosePrint(f"Filling {test}")
            self._fillOrUpdateFactory(True)

        elif self.update:
            self._verbosePrint(f"Updating {test}")
            self._fillOrUpdateFactory(False)
        else:
            self._verbosePrint(f"Found {test} and update is {self.update}")

        test = "block_info"
        self.cur.execute(f"SELECT * from {test}")

        if self.cur.fetchone() == None and self.loadData:
            self._verbosePrint(f"Filling {test}")
            self._fillOrUpdateBlockInfo(True)
        elif self.update:
            self._verbosePrint(f"Updating {test}")
            self._fillOrUpdateBlockInfo(False)
        else:
            self._verbosePrint(f"Found {test} and update is {self.update}")

        test = "initialize"
        self.cur.execute(f"SELECT * from {test}")

        if self.cur.fetchone() == None and self.loadData:
            self._verbosePrint(f"Filling {test}")
            self._fillOrUpdateIni(True)
        elif self.update:
            self._verbosePrint(f"Updating {test}")
            self._fillOrUpdateIni(False)
        else:
            self._verbosePrint(f"Found {test} and update is {self.update}")

        test = "mb"
        self.cur.execute(f"SELECT * from {test} where address = '{self.pool}'")

        if self.cur.fetchone() == None and self.loadData:
            self._verbosePrint(f"Filling {test}")
            self._fillOrUpdateMB(True)
        elif self.update:
            self._verbosePrint(f"Updating {test}")
            self._fillOrUpdateMB(False)
        else:
            self._verbosePrint(f"Found {test} and update is {self.update}")

        test = "collects"
        self.cur.execute(f"SELECT * from {test}")

        if self.cur.fetchone() == None and self.loadData:
            self._verbosePrint(f"Filling {test}")
            self._fillOrUpdateCollects(True)
        elif self.update:
            self._verbosePrint(f"Updating {test}")
            self._fillOrUpdateCollects(False)
        else:
            self._verbosePrint(f"Found {test} and update is {self.update}")

        test = "swaps"
        self.cur.execute(f"SELECT * from {test} where address = '{self.pool}'")

        if self.cur.fetchone() == None and self.loadData:
            self._verbosePrint(f"Filling {test}")
            self._fillOrUpdateSwaps(True)
        elif self.update:
            self._verbosePrint(f"Updating {test}")
            self._fillOrUpdateSwaps(False)
        else:
            self._verbosePrint(f"Found {test} and update is {self.update}")

    def prepareSwaps(self):
        # q = f"SELECT * from swaps where address = '{self.pool}' order by block_number, transaction_index, log_index ;"
        # swaps = pd.read_sql(q, self.engine)
        # bp()
        swaps = (
            pl.scan_parquet(DATA_BASE_DIR / "swaps", hive_partitioning=True)
            .filter(pl.col("address") == self.pool)
            # .sort(by=["block_number", "transaction_index", "log_index"])
            .collect()
            .to_pandas()
        )

        # i really want to figure out a better way to do this
        swaps["block_idx"] = swaps["block_number"] + swaps["transaction_index"] / 10**4
        if "sqrtpricex96" in swaps.columns:
            swaps = swaps.rename(columns={"sqrtpricex96": "sqrtPriceX96"})

        # this is an extremely fast lookup to test for sorting
        assert swaps["block_idx"].is_monotonic_increasing, "swaps are not sorted"
        self.swaps = swaps.drop_duplicates().copy()
        self.swapLoad = True

    def prepareMB(self):
        # q = f"SELECT * from mb where address = '{self.pool}' order by block_number, transaction_index, log_index;"
        # mb = pd.read_sql(q, self.engine)
        mb = (
            pl.scan_parquet(DATA_BASE_DIR / "mb/*.parquet")
            .filter(pl.col("address") == self.pool)
            .sort(by=["block_number", "transaction_index", "log_index"])
            .collect()
            .to_pandas()
        )

        # i really want to figure out a better way to do this
        mb["block_idx"] = mb["block_number"] + mb["transaction_index"] / 10**4
        if "ticklower" in mb.columns:
            mb = mb.rename(columns={"ticklower": "tickLower"})

        if "tickupper" in mb.columns:
            mb = mb.rename(columns={"tickupper": "tickUpper"})

        if "tokenid" in mb.columns:
            mb = mb.rename(columns={"tokenid": "tokenID"})

        mb["tickLower"] = mb["tickLower"].astype(int)
        mb["tickUpper"] = mb["tickUpper"].astype(int)

        # this is an extremely fast lookup to test for sorting
        assert mb["block_idx"].is_monotonic_increasing, "swaps are not sorted"
        # there is a very small time period (1-2 minutes) when databases update,
        # so it is possible for a nft token id to not be saved. we want to ensure that is not happening
        # assert mb[
        #     (mb["owner"] == self.nfp) & (mb["tokenID"].isna())
        # ].empty, "Missing NFP tokenID label"

        if self.bal_adj != 1:
            mb["amount"] = mb["amount"].astype(np.float64) / self.bal_adj

        self.mb = mb.drop_duplicates().copy()
        self.mbLoad = True

    def prepareBlockInfo(self):
        # q = f"SELECT * from block_info order by block_number;"
        # block_info = pd.read_sql(q, self.engine)
        block_info = (
            pl.scan_parquet(DATA_BASE_DIR / "block_info/*.parquet")
            .sort(by=["block_number"])
            .collect()
            .to_pandas()
        )

        assert block_info[
            "block_number"
        ].is_monotonic_increasing, "block info is not sorted"

        # i really want to figure out a better way to do this
        block_info["block_number"] = block_info["block_number"].astype(int)

        self.block_info = block_info
        self.infoLoad = True

    def prepareCollects(self):
        # q = f"SELECT * from collects order by block_number, log_index ;"
        # collects = pd.read_sql(q, self.engine)
        collects = (
            pl.scan_parquet(DATA_BASE_DIR / "collects/*.parquet")
            .sort(by=["block_number", "log_index"])
            .collect()
            .to_pandas()
        )

        if "tokenid" in collects.columns:
            collects = collects.rename(columns={"tokenid": "tokenID"})

        # this is an extremely fast lookup to test for sorting
        assert collects[
            "block_number"
        ].is_monotonic_increasing, "collects are not sorted"

        self.collects = collects.drop_duplicates().copy()

    def prepareInitialization(self):
        # q = f"SELECT * from initialize order by block_number;"
        # pool_ini = pd.read_sql(q, self.engine)
        pool_ini = (
            pl.scan_parquet(DATA_BASE_DIR / "initialize/*.parquet")
            .sort(by=["block_number"])
            .collect()
            .to_pandas()
        )

        assert pool_ini[
            "block_number"
        ].is_monotonic_increasing, "block info is not sorted"

        # i really want to figure out a better way to do this
        pool_ini["block_number"] = pool_ini["block_number"].astype(int)

        self.pool_ini = pool_ini
        self.iniLoad = True

    def prepareData(self):
        """
        Helper function to load the two target datasets
        """
        self.prepareSwaps()
        self.prepareMB()
        self.prepareBlockInfo()

    def initializeLiquidity(self):
        # precompute the coo and kv values
        if not self.mbLoad:
            raise ValueError("Please create the pool with load_data = True")

        mb = self.mb[
            ["block_number", "transaction_index", "tickLower", "tickUpper", "amount"]
        ].copy()
        # precompute the liq distribution
        coo, kv = create_distribution(mb, self.ts)

        self.coo = coo
        self.kv = kv

        self.cooLoad = True

    def createLiq(self, as_of=""):
        if not self.cooLoad:
            self._verbosePrint("Initializing liquidity")
            self.initializeLiquidity()

        x, y = create_liq(self.ts, coordinate_payload=(self.coo, self.kv), as_of=as_of)
        return x, y

    def addMB(self, payload):
        reloadCOO = self.cooLoad

        payload["amount"] = payload["amount"] / self.bal_adj
        mb = self.mb.copy()
        mb = pd.concat([mb, payload])

        mb = mb.sort_values(by="block_idx")
        self.mb = mb

        self.swap_df_asof = -1
        if reloadCOO:
            self.initializeLiquidity()

    def reloadMB(self):
        reloadCOO = self.cooLoad
        self._verbosePrint("Reloading MB from state")

        self.prepareMB()

        self.swap_df_asof = -1
        if reloadCOO:
            self.initializeLiquidity()

    # get internal data
    def getCOO(self):
        return self.coo, self.kv

    def getFee(self):
        return self.fee

    def getTS(self):
        return self.ts

    def getToken0(self):
        return self.token0

    def getToken1(self):
        return self.token1

    def getSwaps(self):
        return self.swaps

    def getMB(self):
        return self.mb

    def getBlockInfo(self):
        return self.block_info

    def getPriceAt(self, as_of, return_block=False) -> float | tuple[float, int | None]:
        # as_of must be a block
        # looking for -1 - the last trade - 3 is the sqrtPriceX96
        if not self.swapLoad:
            raise ValueError("Please load the swap dataset")

        px = self.swaps.loc[self.swaps["block_idx"] < as_of, "sqrtPriceX96"]
        idx = self.swaps.loc[self.swaps["block_idx"] < as_of, "block_idx"]
        if px.empty:
            # this only happens if there have been no swaps in the pool yet
            # TODO
            # debate to see if throwing an exception and catching it is better
            if not self.iniLoad:
                self.prepareInitialization()
            ini = self.pool_ini
            ini = ini[ini["address"] == self.pool]["sqrtPriceX96"]

            assert not ini.empty, "Missing initialization from pool"
            if return_block:
                return (float(ini.item()) / (2**96), None)

            return float(ini.item()) / (2**96)

        else:
            if return_block:
                return (float(px.values[-1]) / (2**96), idx.values[-1])

            return float(px.values[-1]) / (2**96)

    def getTickAt(self, as_of, return_block=False):
        # as_of must be a block
        if not self.swapLoad:
            print("Please load the swap dataset")
            return

        tick = self.swaps.loc[self.swaps["block_idx"] < as_of, "tick"]
        idx = self.swaps.loc[self.swaps["block_idx"] < as_of, "block_idx"]

        if tick.empty:
            # this only happens if there have been no swaps in the pool yet
            # TODO
            # same debate on throwing
            if not self.iniLoad:
                self.prepareInitialization()

            ini = self.pool_ini
            ini = ini[ini["address"] == self.pool]["tick"]

            assert not ini.empty, "Missing initialization from pool"

            if return_block:
                return (int(ini.item()), None)
            else:
                return int(ini.item())
        else:
            if return_block:
                return (int(tick.iloc[-1]), idx.iloc[-1])
            else:
                return int(tick.iloc[-1])

    def getGasAt(self, as_of, type_of_gas="median_gas"):
        if not self.infoLoad:
            raise Exception("Please load the block_info dataset")

        return int(
            self.block_info.loc[
                self.block_info["block_number"] <= math.floor(as_of), type_of_gas
            ]
            .iloc[-1]
            .item()
        )

    def getBlockAtTS(self, ts):
        if not self.infoLoad:
            print("Please load the block_info dataset")
            return

        return self.block_info[self.block_info["block_timestamp"] < ts][
            "block_number"
        ].iloc[-1]

    def getTSAtBlock(self, block):
        if not self.infoLoad:
            print("Please load the block_info dataset")
            return

        return self.block_info[self.block_info["block_number"] < block][
            "block_timestamp"
        ].iloc[-1]

    def getLiqAtBlock(self, as_of):
        x, y = self.createLiq(as_of)
        swap_df = pd.DataFrame(x, columns=["ticks"])
        swap_df["liquidity"] = y

        # # fix floating point liquidity errors
        # swap_df.loc[
        #     swap_df["liquidity"].apply(lambda x: math.isclose(x, 0)), "liquidity"
        # ] = 0
        # fix floating point liquidity errors
        rel_tol = 1e-9  # default relative tolerance for math.isclose
        abs_tol = 0.0  # default absolute tolerance for math.isclose

        mask = np.isclose(swap_df["liquidity"], 0, rtol=rel_tol, atol=abs_tol)
        swap_df.loc[mask, "liquidity"] = 0
        swap_df = swap_df[swap_df["liquidity"] >= 0]

        in_range = (self.getTickAt(as_of) // self.ts) * self.ts
        liquidity = swap_df.loc[swap_df["ticks"] == in_range, "liquidity"]

        if len(liquidity) != 1:
            liquidity = 0
        else:
            liquidity = liquidity.item()

        return liquidity

    # start of the swapping code
    # will optimize and move this once Xin finishes amtOut
    def createSwapDF(self, as_of, given_price=""):
        """
        Precompute the swap dataframe for the given liquidity distribution

        This is needed to accurately know what is in/out of range.

        TODO:
        Create the ability to pass this to the swapIn/pctOut functions
        to precompute needed values
        """
        x, y = self.createLiq(as_of)
        x, y = np.array(x), np.array(y)

        swap_df = pd.DataFrame(x, columns=["ticks"])
        swap_df["liquidity"] = y

        # fix floating point liquidity errors
        # swap_df.loc[
        #     swap_df["liquidity"].apply(lambda x: math.isclose(x, 0)), "liquidity"
        # ] = 0
        # fix floating point liquidity errors
        rel_tol = 1e-9  # default relative tolerance for math.isclose
        abs_tol = 0.0  # default absolute tolerance for math.isclose

        mask = np.isclose(swap_df["liquidity"], 0, rtol=rel_tol, atol=abs_tol)
        swap_df.loc[mask, "liquidity"] = 0

        swap_df = swap_df[swap_df["liquidity"] >= 0]

        swap_df["ts"] = self.ts

        swap_df["p_b"] = np.sqrt(1.0001 ** (swap_df["ticks"] + swap_df["ts"]))
        swap_df["p_a"] = np.sqrt(1.0001 ** (swap_df["ticks"]))
        swap_df["yInTick"] = (
            swap_df["liquidity"] * (swap_df["p_b"] - swap_df["p_a"]) * self.bal_adj
        )

        swap_df["xInTick"] = (
            swap_df["liquidity"]
            * (swap_df["p_b"] - swap_df["p_a"])
            / (swap_df["p_b"] * swap_df["p_a"])
        ) * self.bal_adj

        if not given_price:
            sqrt_P = self.getPriceAt(as_of)
        else:
            sqrt_P = given_price

        swap_df["sqrt_P"] = sqrt_P

        swap_df["inRange0"] = 0.0
        swap_df["inRange1"] = 0.0

        swap_df = swap_df.drop_duplicates()
        # find the in-range tick
        inRangeCondition = (swap_df["p_a"] <= swap_df["sqrt_P"]) & (
            swap_df["p_b"] > swap_df["sqrt_P"]
        )

        row = swap_df.loc[inRangeCondition]
        assert (
            not row.shape[0] > 1
        ), f"Duplicate in-range rows {row} for sqrt_P {sqrt_P}"
        assert (
            not row.shape[0] == 0
        ), f"Missing data for in-range rows {row} for sqrt_P {sqrt_P}"

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

    def swapIn(
        self, swapParams, swap_df=None, inRangeValue="", fees=False, find_max=False
    ) -> Tuple[float, Heur]:
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

        gas = swapParams.get("gasFee", False)
        given_price = swapParams.get("givenPrice")

        # if we are given a current price, we precompute the swap_df
        reset_price = given_price is not None

        # allows for pre-computing the swap_df
        if inRangeValue != "":
            assert (
                swap_df is not None and not swap_df.empty
            ), "Please give a valid swap_df"

        if (
            (not self.swapDF.empty)
            and (self.swap_df_asof == as_of)
            and (not reset_price)
        ):
            swap_df, inRangeValues = self.swapDF, self.inRangeValues

        else:
            swap_df, inRangeValues = self.createSwapDF(as_of, given_price=given_price)

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
        bal_adj = self.bal_adj

        # this determines the direction of the swap
        if tokenIn == self.token1:
            zeroForOne = False
        else:
            zeroForOne = True

        # is there enough liquidity in the current tick?
        if zeroForOne:
            inRangeTest, inRangeToSwap = inRange0, inRangeToSwap0
        else:
            inRangeTest, inRangeToSwap = inRange1, inRangeToSwap1

        tickToFees = defaultdict(float)
        # account for fee
        swapInMinusFee = swapIn * (1 - self.fee)
        if inRangeTest > swapInMinusFee:
            # enough liquidity in range
            liquidity = liquidity_in_range

            # determine how far to push in-range
            if not zeroForOne:
                sqrtP_next = get_next_price_amount1(
                    sqrt_P, liquidity, swapInMinusFee, zeroForOne, bal_adj
                )
                amtOut = get_amount0_delta(sqrtP_next, sqrt_P, liquidity, bal_adj)
            else:
                sqrtP_next = get_next_price_amount0(
                    sqrt_P, liquidity, swapInMinusFee, zeroForOne, bal_adj
                )
                amtOut = get_amount1_delta(sqrtP_next, sqrt_P, liquidity, bal_adj)

            amtOutDelta = 0
            crossed_ticks = 0
            totalFee = swapIn * self.fee
            liquid_tick = tick_in_range

            if fees:
                if liquidity != 0:
                    tickToFees[tick_in_range] = totalFee / liquidity
                else:
                    tickToFees[tick_in_range] = 0
        else:
            # find the first tick with enough liquidity in range
            leftToSwap = swapIn - inRangeTest

            if fees:
                tickToFees[tick_in_range] = inRangeTest * self.fee / liquidity_in_range

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
                if find_max:
                    return outOfRange["cumulativeX"].max()
                assert (
                    outOfRange["cumulativeX"].max() > leftToSwap
                ), "Not enough liquidity in pool"
                liquid_tick = outOfRange[outOfRange["cumulativeX"] > leftToSwap].iloc[0]
                prev_ticks = outOfRange[outOfRange["ticks"] > liquid_tick["ticks"]]

            else:
                if find_max:
                    return outOfRange["cumulativeY"].max()
                assert (
                    outOfRange["cumulativeY"].max() > leftToSwap
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
                amtInSwappedLeft = leftToSwap - prev_ticks["xInTick"].sum()
                prevFee = (inRangeTest + prev_ticks["xInTick"].sum()) * self.fee
                amtInSwappedLeftMinusFee = amtInSwappedLeft * (1 - self.fee)

                if fees:
                    for tick, tickLiq, amt in prev_ticks[
                        ["ticks", "liquidity", "xInTick"]
                    ].values:
                        tickToFees[int(tick)] = amt * self.fee / tickLiq
                    tickToFees[int(liquid_tick["ticks"])] = (
                        amtInSwappedLeft * self.fee / liquid_tick["liquidity"]
                    )

                amtOut = (inRangeToSwap + prev_ticks["yInTick"].sum()) * (1 - self.fee)

                sqrtP_next = get_next_price_amount0(
                    sqrt_P_last_top,
                    liquidity,
                    amtInSwappedLeftMinusFee,
                    zeroForOne,
                    bal_adj,
                )
                amtOutDelta = get_amount1_delta(
                    sqrtP_next, sqrt_P_last_top, liquidity, bal_adj
                )

            else:
                amtInSwappedLeft = leftToSwap - prev_ticks["yInTick"].sum()
                prevFee = (inRangeTest + prev_ticks["yInTick"].sum()) * self.fee
                amtInSwappedLeftMinusFee = amtInSwappedLeft * (1 - self.fee)

                if fees:
                    for tick, tickLiq, amt in prev_ticks[
                        ["ticks", "liquidity", "yInTick"]
                    ].values:
                        tickToFees[int(tick)] = amt * self.fee / tickLiq
                    tickToFees[int(liquid_tick["ticks"])] = (
                        amtInSwappedLeft * self.fee / liquid_tick["liquidity"]
                    )

                amtOut = (inRangeToSwap + prev_ticks["xInTick"].sum()) * (1 - self.fee)
                sqrtP_next = get_next_price_amount1(
                    sqrt_P_last_bottom,
                    liquidity,
                    amtInSwappedLeftMinusFee,
                    zeroForOne,
                    bal_adj,
                )
                amtOutDelta = get_amount0_delta(
                    sqrtP_next, sqrt_P_last_bottom, liquidity, bal_adj
                )

            amtInFee = amtInSwappedLeft * self.fee
            totalFee = prevFee + amtInFee

            crossed_ticks = prev_ticks.shape[0]

            liquid_tick = liquid_tick["ticks"]

        traded_out = amtOut + amtOutDelta
        traded_out = math.floor(traded_out)

        totalFee = math.ceil(totalFee)

        if gas:
            # https://github.com/Uniswap/v3-periphery/blob/main/test/__snapshots__/SwapRouter.gas.spec.ts.snap
            base_swap_fee = 107_759
            median_block_gas = self.getGasAt(as_of)
            gas_fee = median_block_gas * base_swap_fee / 1e18
        else:
            gas_fee = 0

        heur = Heur(
            totalFee,
            crossed_ticks,
            liquidity_in_range,
            sqrt_P,
            sqrtP_next,
            tickToFees,
            inRangeTest,
            swapInMinusFee,
            zeroForOne,
            gas_fee,
        )

        return traded_out, heur

    def swapToPrice(self, swapParams, swap_df="", inRangeValue=""):
        pcts = swapParams["pcts"]
        tokenOut = swapParams["tokenOut"]
        as_of = swapParams["as_of"]

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

        bal_adj = self.bal_adj
        px_cur = sqrt_P
        tick_cur = tick_in_range

        # not sure if this is needed
        cur_range = (tick_cur // self.ts) * self.ts

        inOrOutOfRange = ""
        data = []
        heur = []
        for pct in pcts:
            if pct == "max":
                px_to = 1.0001**887250
                lower = False

            elif pct == "min":
                px_to = 1.0001**-887250
                lower = True
            else:
                assert pct > -1, "Cannot be less than 100% drop"
                px_to = px_cur * np.sqrt(1 + pct)

                if pct < 0:
                    lower = True
                else:
                    lower = False
            zero = tokenOut == self.token0

            if lower:
                inRangeTest = np.sqrt(1.0001**cur_range)
                test_bool = inRangeTest <= px_to
            else:
                inRangeTest = np.sqrt(1.0001 ** (cur_range + self.ts))
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
                    min_tick = swap_df[(swap_df["p_a"] <= px_to)]["ticks"].max()

                    prev_ticks = swap_df[
                        (swap_df["ticks"] > min_tick)
                        & (swap_df["ticks"] < tick_in_range)
                    ]

                    liquid_tick = swap_df[swap_df["ticks"] == min_tick]
                    px_from = liquid_tick["p_b"].item()

                    liquidity = liquid_tick["liquidity"].item()
                    assert liquidity is not np.nan, "Missing liquidity"

                    if zero:
                        amtOut = get_amount0_delta(px_from, px_to, liquidity, bal_adj)
                        amtPrev = prev_ticks["xInTick"].sum()
                        inRangeAmt = inRange0
                    else:
                        amtOut = get_amount1_delta(px_from, px_to, liquidity, bal_adj)
                        amtPrev = prev_ticks["yInTick"].sum()
                        inRangeAmt = inRangeToSwap0

                else:
                    max_tick = swap_df[(swap_df["p_b"] >= px_to)]["ticks"].min()

                    prev_ticks = swap_df[
                        (swap_df["ticks"] < max_tick)
                        & (swap_df["ticks"] > tick_in_range)
                    ]

                    liquid_tick = swap_df[swap_df["ticks"] == max_tick]

                    px_from = liquid_tick["p_a"].item()

                    liquidity = liquid_tick["liquidity"].item()
                    assert liquidity is not np.nan, "Missing liquidity"

                    if zero:
                        amtOut = get_amount0_delta(px_from, px_to, liquidity, bal_adj)
                        amtPrev = prev_ticks["xInTick"].sum()
                        inRangeAmt = inRangeToSwap1

                    else:
                        amtOut = get_amount1_delta(px_from, px_to, liquidity, bal_adj)
                        amtPrev = prev_ticks["yInTick"].sum()
                        inRangeAmt = inRange1

            swapToPct = amtOut + amtPrev + inRangeAmt

            data.append([swapToPct, pct])
            heur.append([inOrOutOfRange, prev_ticks, amtOut, inRangeAmt, px_cur, px_to])

        return data, heur
