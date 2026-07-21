# WEEK 1 / DAY 1 — Bronze Layer Ingestion
# Run this in a Databricks notebook (Python) after uploading the 6 CSVs
# to a volume, e.g. /Volumes/main/payer_raw/landing/  (adjust path to your workspace)
# For local practice without Databricks, this also runs on plain PySpark.

from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, current_timestamp, lit

spark = SparkSession.builder.appName("bronze_ingestion").getOrCreate()

# In Databricks, replace this dict with dbutils volume/mount paths
landing_path = "data"   # local folder for this exercise

payer_files = {
    "payer_a_claims": f"{landing_path}/payer_A_claims_aetna_style.csv",
    "payer_b_claims": f"{landing_path}/payer_B_claims_uhc_style.csv",
    "payer_c_claims": f"{landing_path}/payer_C_claims_cigna_style.csv",
    "payer_a_members": f"{landing_path}/payer_A_members_aetna_style.csv",
    "payer_b_members": f"{landing_path}/payer_B_members_uhc_style.csv",
    "payer_c_members": f"{landing_path}/payer_C_members_cigna_style.csv",
}

bronze_tables = {}
for name, path in payer_files.items():
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv(path)
          # Bronze principle #1: NEVER transform the business data, only add metadata
          .withColumn("_ingest_ts", current_timestamp())
          .withColumn("_source_file", input_file_name())
          .withColumn("_source_system", lit(name.split("_")[0]))  # payer_a / payer_b / payer_c
         )
    bronze_tables[name] = df
    print(f"\n=== {name} ===")
    df.printSchema()
    print(f"Row count: {df.count()}")

# Key architectural point demonstrated here:
# Each payer keeps its ORIGINAL column names in Bronze. We do NOT rename yet.
# Bronze = "as the source sent it, plus lineage metadata." Standardization is a
# Silver-layer decision, not a Bronze one. This is what preserves your ability
# to re-derive Silver differently later without re-ingesting from source.
