# WEEK 1 / DAY 4 — Gold Layer: Claims, Eligibility, Risk Scoring
# Builds on Day 2 (Silver standardization) + Day 3 (DQ flags).
# Demonstrates THREE different grains from the same Silver source — the core
# Gold-layer design skill.

from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, when, concat_ws, lit, count, sum as spark_sum,
                                    avg, countDistinct, date_format, max as spark_max, round as sround)

spark = SparkSession.builder.appName("gold_layer").getOrCreate()
landing_path = "data"

# ---- Re-build Silver claims (same as Day 2/3) ----
PAYER_CLAIMS_MAPPING = {
    "payer_a": {"member_id":"member_id","claim_id":"claim_id","diagnosis_code":"diagnosis_code",
                "procedure_code":"procedure_code","service_date":"service_date","billed_amount":"billed_amount",
                "allowed_amount":"allowed_amount","provider_npi":"provider_npi","claim_status":"claim_status"},
    "payer_b": {"member_id":"MemberID","claim_id":"ClaimNbr","diagnosis_code":"DxCd",
                "procedure_code":"ProcCd","service_date":"SvcDt","billed_amount":"BilledAmt",
                "allowed_amount":"AllowedAmt","provider_npi":"ProviderNPI","claim_status":"ClmStatus"},
    "payer_c": {"member_id":"mbrid","claim_id":"clmid","diagnosis_code":"diagcd",
                "procedure_code":"proccd","service_date":"dos","billed_amount":"chgamt",
                "allowed_amount":"pdamt","provider_npi":"npinum","claim_status":"clmstat"},
}
CANONICAL_CLAIMS_COLS = ["member_id","claim_id","diagnosis_code","procedure_code","service_date",
                          "billed_amount","allowed_amount","provider_npi","claim_status"]

def standardize_claims(bronze_df, payer_key):
    mapping = PAYER_CLAIMS_MAPPING[payer_key]
    df = bronze_df.select(*[col(src).alias(canon) for canon, src in mapping.items()])
    return df.select(*CANONICAL_CLAIMS_COLS).withColumn("source_payer", lit(payer_key))

def load_standardized(payer_key, filename):
    bronze = spark.read.option("header", True).option("inferSchema", True).csv(f"{landing_path}/{filename}")
    return standardize_claims(bronze, payer_key)

silver_claims = (load_standardized("payer_a", "payer_A_claims_aetna_style.csv")
    .unionByName(load_standardized("payer_b", "payer_B_claims_uhc_style.csv"))
    .unionByName(load_standardized("payer_c", "payer_C_claims_cigna_style.csv")))

DQ_RULES = [
    ("null_diagnosis_code", col("diagnosis_code").isNull(), "diagnosis_code is missing"),
    ("null_provider_npi",   col("provider_npi").isNull(),   "provider_npi is missing"),
    ("negative_billed_amount", col("billed_amount") < 0,    "billed_amount is negative"),
    ("unrealistic_billed_amount", col("billed_amount") > 50000, "billed_amount exceeds realistic claim threshold"),
    ("invalid_claim_status", ~col("claim_status").isin("PAID","DENIED","PENDING"), "claim_status is not a recognized value"),
]
df = silver_claims
error_parts = []
for rule_name, fail_condition, message in DQ_RULES:
    flag_col = f"_flag_{rule_name}"
    df = df.withColumn(flag_col, when(fail_condition, lit(message)))
    error_parts.append(flag_col)
df = df.withColumn("error_description", concat_ws("; ", *error_parts))
df = df.withColumn("error_description", when(col("error_description") == "", lit(None)).otherwise(col("error_description")))
silver_claims_dq = df.drop(*error_parts)

# ================================================================
# GOLD TABLE 1: gold_claims — grain = one row per claim (clean subset)
# ================================================================
gold_claims = (silver_claims_dq
    .withColumn("net_paid", col("allowed_amount"))
    .withColumn("is_high_dollar_claim", col("billed_amount") > 5000)
    .withColumn("claim_month", date_format(col("service_date"), "yyyy-MM"))
)

print("=== GOLD: Claims (grain = 1 row per claim) ===")
gold_claims.select("claim_id","member_id","source_payer","service_date","claim_month",
                    "net_paid","is_high_dollar_claim","claim_status","error_description").show(5, truncate=False)
print(f"Total rows: {gold_claims.count()}")

# ================================================================
# GOLD TABLE 2: gold_member_eligibility — grain = one row per member per month
# (Derived here from claims presence as a simplified stand-in for a real
#  eligibility feed; in production this usually comes from a separate
#  enrollment file, not claims — noted explicitly because that's a common
#  design trap: don't infer eligibility purely from "did they file a claim.")
# ================================================================
gold_eligibility = (gold_claims
    .filter(col("error_description").isNull())  # only trust clean claims for this derived signal
    .groupBy("member_id", "claim_month")
    .agg(count("claim_id").alias("claims_filed_this_month"))
    .withColumn("has_activity_flag", lit(True))
)

print("\n=== GOLD: Member Eligibility-style monthly activity (grain = 1 row per member per month) ===")
gold_eligibility.orderBy("member_id", "claim_month").show(5, truncate=False)
print(f"Total rows: {gold_eligibility.count()}")

# ================================================================
# GOLD TABLE 3: gold_member_risk_score — grain = one row per member
# Simple illustrative risk score: NOT a real clinical risk model —
# demonstrates the aggregation pattern (spend + diagnosis diversity + frequency)
# ================================================================
member_agg = (gold_claims
    .filter(col("error_description").isNull())
    .groupBy("member_id")
    .agg(
        spark_sum("net_paid").alias("total_paid_amount"),
        count("claim_id").alias("total_claims"),
        countDistinct("diagnosis_code").alias("distinct_diagnoses"),
        spark_max("is_high_dollar_claim").cast("int").alias("has_high_dollar_claim")
    )
)

# Normalize into a 0-100 illustrative score (NOT clinically valid — teaching pattern only)
max_paid = member_agg.agg(spark_max("total_paid_amount")).collect()[0][0]
gold_risk_score = (member_agg
    .withColumn("spend_component", sround((col("total_paid_amount") / lit(max_paid)) * 50, 2))
    .withColumn("diversity_component", sround(col("distinct_diagnoses") * 5, 2))
    .withColumn("risk_score", sround(col("spend_component") + col("diversity_component"), 2))
)

print("\n=== GOLD: Member Risk Score (grain = 1 row per member) ===")
gold_risk_score.orderBy(col("risk_score").desc()).show(5, truncate=False)
print(f"Total rows: {gold_risk_score.count()}")

print("\n=== Grain check: does gold_risk_score have exactly 1 row per distinct member in gold_claims? ===")
distinct_members_in_claims = gold_claims.select("member_id").distinct().count()
print(f"Distinct members in gold_claims: {distinct_members_in_claims}")
print(f"Rows in gold_risk_score: {gold_risk_score.count()}")
