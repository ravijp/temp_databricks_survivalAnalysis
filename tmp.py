import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def check_duplicates(df: DataFrame, step: str) -> DataFrame:
    """Check for exact duplicate rows and print summary"""
    total = df.count()
    unique = df.distinct().count()
    duplicates = total - unique
    
    print(f"{step}: {total:,} records, {duplicates:,} duplicates")
    if duplicates > 0:
        print(f"  WARNING: {duplicates:,} exact duplicate rows detected")
    
    return df

def check_join_multiplier(before_df: DataFrame, after_df: DataFrame, join_name: str) -> DataFrame:
    """Check if join created record multiplication"""
    before_count = before_df.count()
    after_count = after_df.count()
    
    if after_count > before_count:
        multiplier = after_count / before_count
        increase = after_count - before_count
        print(f"{join_name}: {multiplier:.2f}x multiplier (+{increase:,} records)")
    
    return after_df

def check_key_duplicates(df: DataFrame, key_cols: list, step: str) -> DataFrame:
    """Check for duplicate keys in dataset"""
    total = df.count()
    unique_keys = df.select(*key_cols).distinct().count()
    
    if total > unique_keys:
        dup_count = total - unique_keys
        print(f"{step}: {dup_count:,} duplicate keys for {key_cols}")
        
        # Show top duplicates
        duplicates = (df.groupBy(*key_cols)
                     .count()
                     .filter(F.col("count") > 1)
                     .orderBy(F.desc("count"))
                     .limit(3))
        
        print("  Top duplicate keys:")
        duplicates.show(3, truncate=False)
    
    return df

def debug_manager_status(mgr_df: DataFrame) -> DataFrame:
    """Check manager status for duplicate manager-date combinations"""
    key_cols = ["db_schema", "clnt_obj_id", "mgr_sup_id", "rec_eff_strt_dt", "rec_eff_end_dt"]
    return check_key_duplicates(mgr_df, key_cols, "manager_status")

# Main debugging insertions for the ETL pipeline
def main() -> None:
    spark.conf.set("spark.sql.shuffle.partitions", 1200)
    
    # Load dimension tables
    clnt_mstr_df = load_clnt_mstr()
    pers_dln_df = load_pers_dln_df()
    job_dln_df = load_job_dln_df()
    ppfl_dln_df = load_ppfl_dln_df()
    
    # Load fact tables
    wass_base_df = load_wass_base_df()
    wass_base_df = check_duplicates(wass_base_df, "wass_base")
    
    wevt_term_df = load_wevt_term_df()
    
    # Join termination events
    wass_term_df = calc_wass_term_df(wass_base_df=wass_base_df, wevt_term_df=wevt_term_df)
    wass_term_df = check_duplicates(wass_term_df, "after_termination_join")
    wass_term_df = check_join_multiplier(wass_base_df, wass_term_df, "termination_join")
    
    # Load manager hierarchy
    mgr_status_df = load_mgr_status_df(wass_term_df=wass_term_df)
    mgr_status_df = debug_manager_status(mgr_status_df)
    
    # Main merge - critical checkpoint
    print("\nCRITICAL CHECKPOINT: Main data merge")
    wass_merged_df = calc_wass_merged_df(
        clnt_mstr_df=clnt_mstr_df,
        wass_term_df=wass_term_df,
        pers_dln_df=pers_dln_df,
        job_dln_df=job_dln_df,
        mgr_status_df=mgr_status_df,
        ppfl_dln_df=ppfl_dln_df,
    )
    
    wass_merged_df = check_duplicates(wass_merged_df, "after_main_merge")
    wass_merged_df = check_join_multiplier(wass_term_df, wass_merged_df, "main_merge")
    
    # Team aggregations
    team_age_df = calc_team_agg_df(wass_merged_df=wass_merged_df)
    
    # Final join
    wass_final_df = wass_merged_df.join(
        team_age_df, 
        on=["db_schema", "clnt_obj_id", "pers_obj_id", "work_asgmt_nbr", "rec_eff_strt_dt"], 
        how="left"
    )
    
    # Final validation
    wass_final_df = check_duplicates(wass_final_df, "final_output")
    
    print(f"\nFinal summary: {wass_final_df.count():,} total records")
    unique_employees = wass_final_df.select("clnt_obj_id", "pers_obj_id").distinct().count()
    print(f"Unique employees: {unique_employees:,}")
    
# Quick one-liner debugging functions
def dup_check(df, name): 
    return check_duplicates(df, name)

def join_check(before, after, name): 
    return check_join_multiplier(before, after, name)
