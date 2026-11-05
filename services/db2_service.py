"""
JalDrishti-AI — Db2 on Cloud Database Service
Handles connection, table creation, and CRUD for the MIS 78th Round data.
"""

import os
import json
import ibm_db
import ibm_db_dbi
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def _dsn() -> str:
    return (
        f"DATABASE={os.getenv('IBM_DB2_DATABASE', 'bludb')};"
        f"HOSTNAME={os.getenv('IBM_DB2_HOST')};"
        f"PORT={os.getenv('IBM_DB2_PORT', '50001')};"
        f"PROTOCOL=TCPIP;"
        f"UID={os.getenv('IBM_DB2_USERNAME')};"
        f"PWD={os.getenv('IBM_DB2_PASSWORD')};"
        f"SECURITY=SSL;"
    )


def get_connection():
    return ibm_db.connect(_dsn(), "", "")


def get_pandas_connection():
    conn = get_connection()
    return ibm_db_dbi.Connection(conn)


def create_table():
    """Create the MIS_DRINKING_WATER table if it does not exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS MIS_DRINKING_WATER (
        HHID             VARCHAR(20) NOT NULL PRIMARY KEY,
        STATE            VARCHAR(50),
        SECTOR           VARCHAR(10),
        SOCIAL_GROUP     VARCHAR(20),
        RELIGION         VARCHAR(20),
        HOUSEHOLD_SIZE   INTEGER,
        WATER_SOURCE     VARCHAR(60),
        IMPROVED_SOURCE  SMALLINT,
        WATER_TIME_MIN   INTEGER,
        WATER_ON_PREMISES SMALLINT,
        CLEAN_FUEL       VARCHAR(30),
        HAS_CLEAN_FUEL   SMALLINT,
        MIGRATION        SMALLINT,
        SURVEY_YEAR      INTEGER,
        SURVEY_ROUND     INTEGER
    )
    """
    conn = get_connection()
    ibm_db.exec_immediate(conn, ddl)
    ibm_db.close(conn)
    print("[Db2] Table MIS_DRINKING_WATER ensured.")


def load_csv_to_db2(csv_path: str):
    """Bulk-load the cleaned CSV into Db2."""
    df = pd.read_csv(csv_path)
    pconn = get_pandas_connection()
    df.columns = [c.upper() for c in df.columns]
    df.rename(columns={
        "MAIN_DRINKING_WATER_SOURCE": "WATER_SOURCE",
        "WATER_COLLECTION_TIME_MIN":  "WATER_TIME_MIN",
        "MIGRATION_LAST_YEAR":        "MIGRATION",
        "SURVEY_ROUND":               "SURVEY_ROUND",
    }, inplace=True)

    cursor = pconn.cursor()
    insert_sql = """
        INSERT INTO MIS_DRINKING_WATER
        (HHID, STATE, SECTOR, SOCIAL_GROUP, RELIGION, HOUSEHOLD_SIZE,
         WATER_SOURCE, IMPROVED_SOURCE, WATER_TIME_MIN, WATER_ON_PREMISES,
         CLEAN_FUEL, HAS_CLEAN_FUEL, MIGRATION, SURVEY_YEAR, SURVEY_ROUND)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    rows = list(df.itertuples(index=False, name=None))
    cursor.executemany(insert_sql, rows)
    pconn.commit()
    print(f"[Db2] Loaded {len(rows)} rows into MIS_DRINKING_WATER.")


def query_state_summary() -> pd.DataFrame:
    """Return improved-water access % by state."""
    sql = """
        SELECT STATE,
               COUNT(*) AS TOTAL_HH,
               SUM(IMPROVED_SOURCE) AS IMPROVED_HH,
               ROUND(100.0 * SUM(IMPROVED_SOURCE) / COUNT(*), 2) AS IMPROVED_PCT
        FROM MIS_DRINKING_WATER
        GROUP BY STATE
        ORDER BY IMPROVED_PCT DESC
    """
    pconn = get_pandas_connection()
    return pd.read_sql(sql, pconn)


def query_sector_summary() -> pd.DataFrame:
    sql = """
        SELECT SECTOR,
               ROUND(100.0 * AVG(FLOAT(IMPROVED_SOURCE)), 2) AS IMPROVED_PCT,
               ROUND(100.0 * AVG(FLOAT(HAS_CLEAN_FUEL)),  2) AS CLEAN_FUEL_PCT,
               ROUND(100.0 * AVG(FLOAT(MIGRATION)),        2) AS MIGRATION_PCT
        FROM MIS_DRINKING_WATER
        GROUP BY SECTOR
    """
    pconn = get_pandas_connection()
    return pd.read_sql(sql, pconn)
