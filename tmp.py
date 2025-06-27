pk_analysis = (
    wass_df
    .agg(
        F.count("*").alias("total_records"),
        F.countDistinct(*pk_cols).alias("unique_keys"),
        F.countDistinct("pers_obj_id").alias("unique_employees")
    )
    .withColumn("duplicate_pct", 
        ((F.col("total_records") - F.col("unique_keys")) / F.col("total_records") * 100))
)
pk_analysis.show()

# 2. TIME COVERAGE & DATA ARTIFACTS
print("\n2. TIME COVERAGE ANALYSIS")
time_analysis = (
    wass_df
    .select(
        F.year("rec_eff_strt_dt").alias("start_year"),
        F.year("rec_eff_end_dt").alias("end_year")
    )
    .agg(
        # Start date patterns
        F.min("start_year").alias("min_start_year"),
        F.max("start_year").alias("max_start_year"),
        F.sum(F.when(F.col("start_year") < 2000, 1).otherwise(0)).alias("artifact_start_records"),
        
        # End date patterns  
        F.min("end_year").alias("min_end_year"),
        F.max("end_year").alias("max_end_year"),
        F.sum(F.when(F.col("end_year") > 2030, 1).otherwise(0)).alias("future_end_records"),
        F.sum(F.when(F.col("end_year").isNull(), 1).otherwise(0)).alias("null_end_records"),
        
        # Data quality
        F.sum(F.when(F.col("rec_eff_end_dt") < F.col("rec_eff_strt_dt"), 1).otherwise(0)).alias("invalid_date_logic")
    )
)
time_analysis.show()

# 3. EMPLOYEE DISTRIBUTION & EPISODE PATTERNS
print("\n3. EMPLOYEE EPISODE PATTERNS")
employee_stats = (
    wass_df
    .groupBy("pers_obj_id")
    .agg(F.count("*").alias("episode_count"))
    .agg(
        F.count("*").alias("total_employees"),
        F.min("episode_count").alias("min_episodes"),
        F.max("episode_count").alias("max_episodes"),
        F.mean("episode_count").alias("avg_episodes"),
        F.expr("percentile_approx(episode_count, 0.5)").alias("median_episodes"),
        F.expr("percentile_approx(episode_count, 0.95)").alias("p95_episodes"),
        F.sum(F.when(F.col("episode_count") > 100, 1).otherwise(0)).alias("employees_over_100_episodes")
    )
)
employee_stats.show()

# 4. EMPLOYMENT STATUS & BUSINESS LOGIC
print("\n4. EMPLOYMENT STATUS DISTRIBUTION")
status_analysis = (
    wass_df
    .groupBy("work_asgmt_stus_cd")
    .agg(
        F.count("*").alias("record_count"),
        F.countDistinct("pers_obj_id").alias("unique_employees"),
        F.sum(F.when(F.col("rec_eff_end_dt").isNull(), 1).otherwise(0)).alias("null_end_dates"),
        F.sum(F.when(F.year("rec_eff_end_dt") > 2030, 1).otherwise(0)).alias("future_end_dates")
    )
    .orderBy(F.desc("record_count"))
)
status_analysis.show()

# 5. DATA COMPLETENESS FOR CRITICAL FIELDS
print("\n5. DATA COMPLETENESS CHECK")
completeness_check = (
    wass_df
    .agg(
        F.count("*").alias("total_records"),
        
        # Critical fields for modeling
        (F.sum(F.when(F.col("job_cd").isin("UNKNOWN", ""), 1).when(F.col("job_cd").isNull(), 1).otherwise(0)) 
         / F.count("*") * 100).alias("job_cd_missing_pct"),
        
        (F.sum(F.when(F.col("mngr_pers_obj_id").isin("UNKNOWN", ""), 1).when(F.col("mngr_pers_obj_id").isNull(), 1).otherwise(0)) 
         / F.count("*") * 100).alias("manager_missing_pct"),
        
        (F.sum(F.when(F.col("annl_cmpn_amt").isNull(), 1).when(F.col("annl_cmpn_amt") == 0, 1).otherwise(0)) 
         / F.count("*") * 100).alias("salary_missing_pct"),
        
        (F.sum(F.when(F.col("pay_rt_type_cd").isin("UNKNOWN", ""), 1).when(F.col("pay_rt_type_cd").isNull(), 1).otherwise(0)) 
         / F.count("*") * 100).alias("paytype_missing_pct")
    )
)
completeness_check.show()

# 6. CLIENT CONCENTRATION ANALYSIS
print("\n6. CLIENT/COMPANY DISTRIBUTION")
client_stats = (
    wass_df
    .groupBy("clnt_obj_id")
    .agg(
        F.count("*").alias("record_count"),
        F.countDistinct("pers_obj_id").alias("employee_count")
    )
    .agg(
        F.count("*").alias("total_clients"),
        F.max("record_count").alias("max_records_per_client"),
        F.max("employee_count").alias("max_employees_per_client"),
        F.sum(F.when(F.col("employee_count") > 1000, 1).otherwise(0)).alias("large_clients_1000plus"),
        F.sum(F.when(F.col("employee_count") > 10000, 1).otherwise(0)).alias("xlarge_clients_10000plus")
    )
)
client_stats.show()

# 7. SOURCE SYSTEM COVERAGE
print("\n7. SOURCE SYSTEM DISTRIBUTION")
source_stats = (
    wass_df
    .groupBy("db_schema")
    .agg(
        F.count("*").alias("record_count"),
        F.countDistinct("pers_obj_id").alias("employee_count"),
        F.countDistinct("clnt_obj_id").alias("client_count")
    )
    .orderBy(F.desc("record_count"))
)
source_stats.show()

# 8. RECENT DATA QUALITY (Focus on 2020+)
print("\n8. RECENT DATA QUALITY (2020+)")
recent_quality = (
    wass_df
    .filter(F.year("rec_eff_strt_dt") >= 2020)
    .agg(
        F.count("*").alias("recent_records"),
        F.countDistinct("pers_obj_id").alias("recent_employees"),
        
        # Completeness in recent data
        (F.sum(F.when(F.col("job_cd").isin("UNKNOWN", ""), 1).when(F.col("job_cd").isNull(), 1).otherwise(0)) 
         / F.count("*") * 100).alias("recent_job_missing_pct"),
        
        (F.sum(F.when(F.col("annl_cmpn_amt").isNull(), 1).when(F.col("annl_cmpn_amt") == 0, 1).otherwise(0)) 
         / F.count("*") * 100).alias("recent_salary_missing_pct")
    )
)
recent_quality.show()

print("\n=== SUMMARY INSIGHTS ===")
print("✓ Check duplicate percentage - should be ~25%")
print("✓ Identify usable date range (exclude pre-2000 and post-2030)")
print("✓ Note employees with excessive episodes (data quality issues)")
print("✓ Assess field completeness for feature engineering")
print("✓ Understand client concentration for stratification")

# Unpersist cache
wass_df.unpersist()
