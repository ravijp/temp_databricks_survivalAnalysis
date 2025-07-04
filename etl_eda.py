"""
Employee Turnover Survival Analysis Data Preparation Pipeline
Handles multiple dataset formats with proper vantage date logic and no data leakage
Author: Ravi Prakash
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurvivalDataProcessor:
    """
    Comprehensive survival analysis data preparation with employment cycle handling
    Supports multiple dataset formats and proper temporal validation splits
    """
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.person_id_cols = ["db_schema", "clnt_obj_id", "pers_obj_id"]
        self.person_id_combined = "person_composite_id"
        
    def create_composite_person_id(self, df: DataFrame) -> DataFrame:
        """Create composite person identifier from multi-level keys"""
        return df.withColumn(
            self.person_id_combined,
            concat_ws("_", *self.person_id_cols)
        )
    

    def clean_termination_records(self, df: DataFrame,
                                status_col: str = "work_asgnmt_stus_cd",
                                start_date_col: str = "rec_eff_start_dt_mod",
                                end_date_col: str = "rec_eff_end_dt") -> DataFrame:
        """
        Consolidate consecutive termination records into single episodes
        Preserves earliest termination date and extends to final end date
        """
        
        df_with_id = self.create_composite_person_id(df)
        person_window = Window.partitionBy(self.person_id_combined).orderBy(start_date_col)
        
        # FIXED: Add col() wrapper around column references
        df_with_groups = df_with_id.withColumn(
            "is_termination", 
            when(col(status_col) == "T", 1).otherwise(0)
        ).withColumn(
            "prev_is_termination", 
            lag(col("is_termination")).over(person_window)
        ).withColumn(
            "termination_group_start",
            when((col("is_termination") == 1) & 
                (coalesce(col("prev_is_termination"), lit(0)) == 0), 1)
            .otherwise(0)
        ).withColumn(
            "termination_group_id",
            sum(col("termination_group_start")).over(
                person_window.rowsBetween(Window.unboundedPreceding, 0)
            )
        )
        
        # Calculate group boundaries for termination records
        termination_groups = df_with_groups.filter(
            col("is_termination") == 1
        ).groupBy(
            self.person_id_combined, "termination_group_id"
        ).agg(
            min(col(start_date_col)).alias("group_start_date"),
            max(col(end_date_col)).alias("group_end_date"),
            count("*").alias("records_in_group"),
            first(struct(*[c for c in df.columns if c not in [start_date_col, end_date_col]]))
            .alias("representative_record")
        )
        
        # Create consolidated termination records
        consolidated_terminations = termination_groups.select(
            col("representative_record.*"),
            col("group_start_date").alias(start_date_col),
            col("group_end_date").alias(end_date_col)
        )
        
        # Combine with non-termination records
        non_termination_records = df_with_groups.filter(col("is_termination") == 0)
        
        result = non_termination_records.select(df.columns).union(
            consolidated_terminations.select(df.columns)
        )
        
        logger.info(f"Termination record consolidation: {df.count():,} -> {result.count():,} records")# TODO commment
        return result
    
    def fill_short_gaps(self, df: DataFrame,
                    status_col: str = "work_asgnmt_stus_cd",
                    start_date_col: str = "rec_eff_start_dt_mod", 
                    end_date_col: str = "rec_eff_end_dt",
                    gap_threshold_days: int = 30) -> DataFrame:
        """
        Bridge short employment gaps by extending active periods
        Handles administrative terminations and brief unemployment periods
        """
        
        df_with_id = self.create_composite_person_id(df)
        person_window = Window.partitionBy(self.person_id_combined).orderBy(start_date_col)
        
        # FIXED: Add col() wrapper around column references
        df_with_context = df_with_id.withColumn(
            "prev_status", lag(col(status_col)).over(person_window)
        ).withColumn(
            "next_status", lead(col(status_col)).over(person_window)
        ).withColumn(
            "prev_end_date", lag(col(end_date_col)).over(person_window)
        ).withColumn(
            "next_start_date", lead(col(start_date_col)).over(person_window)
        ).withColumn(
            "gap_duration",
            when(col(status_col) == "T", 
                datediff(col(end_date_col), col(start_date_col)) + 1)
            .otherwise(0)
        )
        
        # Identify short gaps to fill (A -> T -> A pattern)
        df_with_flags = df_with_context.withColumn(
            "is_short_gap",
            when(
                (col(status_col) == "T") &
                (col("prev_status") == "A") &
                (col("next_status") == "A") &
                (col("gap_duration") <= gap_threshold_days),
                1
            ).otherwise(0)
        ).withColumn(
            "extend_previous_active",
            when(
                (col(status_col) == "A") &
                (lead(col("is_short_gap")).over(person_window) == 1),
                1
            ).otherwise(0)
        )
        
        # Extend active periods over short gaps
        df_extended = df_with_flags.withColumn(
            end_date_col,
            when(
                col("extend_previous_active") == 1,
                date_sub(lead(lead(col(start_date_col)).over(person_window)).over(person_window), 1)
            ).otherwise(col(end_date_col))
        )
        
        # Remove short gap records
        result = df_extended.filter(col("is_short_gap") == 0).select(df.columns)
        
        logger.info(f"Short gap filling: {df.count():,} -> {result.count():,} records")# TODO commment
        return result

    
    def identify_employment_cycles(self, df: DataFrame,
                                status_col: str = "work_asgnmt_stus_cd",
                                start_date_col: str = "rec_eff_start_dt_mod") -> DataFrame:
        """
        Identify employment cycles and flag latest cycle records
        Handles rehired employees by isolating current employment period
        """
        
        df_with_id = self.create_composite_person_id(df)
        person_window = Window.partitionBy(self.person_id_combined).orderBy(start_date_col)
        
        # Count termination events per person
        termination_counts = df_with_id.filter(
            col(status_col) == "T"
        ).groupBy(self.person_id_combined).agg(
            count("*").alias("total_employment_cycles"),
            max(col(start_date_col)).alias("latest_termination_date"),
            expr(f"sort_array(collect_list({start_date_col}), false)").alias("all_termination_dates")
        ).withColumn(
            "has_multiple_cycles", 
            when(col("total_employment_cycles") > 1, 1).otherwise(0)
        ).withColumn(
            "second_latest_termination_date",
            when(size(col("all_termination_dates")) >= 2, 
                col("all_termination_dates")[1]).otherwise(lit(None))
        )
        
        # Join cycle information back to main dataset
        df_with_cycles = df_with_id.join(
            termination_counts, self.person_id_combined, "left"
        ).withColumn(
            "total_employment_cycles", 
            coalesce(col("total_employment_cycles"), lit(1))
        ).withColumn(
            "has_multiple_cycles",
            coalesce(col("has_multiple_cycles"), lit(0))
        )
        
        # Determine current employment status
        current_status = df_with_cycles.withColumn(
            "record_rank",
            row_number().over(
                Window.partitionBy(self.person_id_combined)
                    .orderBy(desc(col(start_date_col)))
            )
        ).filter(col("record_rank") == 1).select(
            self.person_id_combined,
            col(status_col).alias("current_status")
        )
        
        # Define latest cycle boundaries
        df_final = df_with_cycles.join(current_status, self.person_id_combined).withColumn(
            "latest_cycle_start_date",
            when(
                # Currently terminated with multiple cycles: use records after second-latest termination
                (col("current_status") == "T") & 
                (col("total_employment_cycles") > 1),
                col("second_latest_termination_date")
            ).when(
                # Currently active with previous terminations: use records after latest termination
                (col("current_status") == "A") & 
                (col("total_employment_cycles") >= 1),
                col("latest_termination_date")
            ).otherwise(
                # Single cycle: use all records
                lit("1900-01-01").cast("date")
            )
        ).withColumn(
            "is_latest_cycle",
            when(
                col(start_date_col) > coalesce(col("latest_cycle_start_date"), 
                                            lit("1900-01-01").cast("date")),
                1
            ).otherwise(0)
        ).withColumn(
            "employment_cycle_info",
            struct(
                col("total_employment_cycles").alias("total_cycles"),
                col("has_multiple_cycles").alias("multiple_cycles_flag"),
                col("current_status").alias("current_employment_status"),
                col("is_latest_cycle").alias("in_current_cycle")
            )
        )
        
        return df_final
    
    def get_latest_cycle_records(self, df: DataFrame) -> DataFrame:
        """Filter to latest employment cycle records only"""
        
        df_with_cycles = self.identify_employment_cycles(df)
        latest_cycle_records = df_with_cycles.filter(col("is_latest_cycle") == 1)
        
        # TODO commment
        # Log employment cycle statistics
        cycle_stats = df_with_cycles.groupBy("total_employment_cycles").agg(
            countDistinct(self.person_id_combined).alias("employee_count")
        ).collect()
        
        logger.info("Employment cycle distribution:")
        for row in cycle_stats:
            logger.info(f"  {row['total_employment_cycles']} cycles: {row['employee_count']:,} employees")
        # TODO commment
        
        return latest_cycle_records
    
    def compress_episodes(self, df: DataFrame, 
                        compression_cols: list = None,
                        start_date_col: str = "rec_eff_start_dt_mod",
                        end_date_col: str = "rec_eff_end_dt") -> DataFrame:
        """
        Compress consecutive episodes with identical feature combinations
        Reduces start-stop dataset size while preserving temporal information
        """
        
        df_with_id = self.create_composite_person_id(df)
        
        # Default to all columns except dates and identifiers for compression
        if compression_cols is None:
            exclude_cols = [self.person_id_combined, start_date_col, end_date_col] + self.person_id_cols
            compression_cols = [c for c in df.columns if c not in exclude_cols]
        
        person_window = Window.partitionBy(self.person_id_combined).orderBy(start_date_col)
        
        # Create hash of compression columns to detect changes
        compression_hash = sha2(
            concat_ws("|", *[coalesce(col(c).cast("string"), lit("NULL")) 
                        for c in compression_cols]), 
            256
        )
        
        # Identify episode boundaries where feature combinations change
        df_with_changes = df_with_id.withColumn("compression_hash", compression_hash) \
            .withColumn("prev_hash", lag(col("compression_hash")).over(person_window)) \
            .withColumn(
                "episode_boundary",
                when(
                    (col("prev_hash").isNotNull()) & 
                    (col("compression_hash") != col("prev_hash")), 
                    1
                ).otherwise(0)
            ).withColumn(
                "episode_id",
                sum(col("episode_boundary")).over(
                    person_window.rowsBetween(Window.unboundedPreceding, 0)
                )
            )
        
        # Compress episodes by taking first record and extending date range
        compressed_episodes = df_with_changes.groupBy(
            self.person_id_combined, "episode_id"
        ).agg(
            min(col(start_date_col)).alias(start_date_col),
            max(col(end_date_col)).alias(end_date_col),
            count("*").alias("original_records_compressed"),
            first(struct(*[c for c in df.columns 
                        if c not in [start_date_col, end_date_col]])).alias("episode_data")
        ).select(
            col("episode_data.*"),
            col(start_date_col),
            col(end_date_col),
            col("original_records_compressed")
        )
        # TODO commment        
        compression_ratio = 1 - (compressed_episodes.count() / df_with_id.count())
        logger.info(f"Episode compression: {df_with_id.count():,} -> {compressed_episodes.count():,} "
                f"({compression_ratio:.2%} reduction)")
        # TODO commment
                
        return compressed_episodes    
    def create_baseline_features(self, df: DataFrame, vantage_date: str) -> DataFrame:
        """
        Create baseline employee features as of vantage date
        Features represent employee state at prediction time
        """
        
        df_with_id = self.create_composite_person_id(df)
        vantage_date_lit = lit(vantage_date).cast("date")
        
        # Get most recent record on or before vantage date
        baseline_window = Window.partitionBy(self.person_id_combined).orderBy(desc("rec_eff_start_dt_mod"))
        
        baseline_records = df_with_id.filter(
            col("rec_eff_start_dt_mod") <= vantage_date_lit
        ).withColumn(
            "baseline_rank", row_number().over(baseline_window)
        ).filter(col("baseline_rank") == 1)
        
        # Calculate tenure and experience features
        enhanced_features = baseline_records.withColumn(
            "tenure_at_vantage_days",
            datediff(vantage_date_lit, col("rec_eff_start_dt_mod"))
        ).withColumn(
            "baseline_salary", col("annl_cmpn_amt")
        ).withColumn(
            "baseline_job_title", col("job_dsc")
        )
        
        # Count job changes up to vantage date
        job_changes = df_with_id.filter(
            col("rec_eff_start_dt_mod") <= vantage_date_lit
        ).groupBy(self.person_id_combined).agg(
            countDistinct("job_dsc").alias("job_changes_count"),
            countDistinct("mngr_pers_obj_id").alias("manager_changes_count"),
            max("annl_cmpn_amt").alias("max_salary_to_date"),
            min("annl_cmpn_amt").alias("min_salary_to_date")
        ).withColumn(
            "salary_growth_ratio",
            col("max_salary_to_date") / col("min_salary_to_date")
        )
        
        # Combine baseline features with historical aggregations
        final_features = enhanced_features.join(job_changes, self.person_id_combined, "left")
        
        return final_features
    
    def get_survival_outcomes(self, df: DataFrame, 
                            active_employees: DataFrame,
                            vantage_date: str, 
                            follow_up_days: int) -> DataFrame:
        """
        Calculate survival outcomes within follow-up period
        Handles right-censoring for employees who don't terminate
        """
        
        df_with_id = self.create_composite_person_id(df)
        vantage_date_lit = lit(vantage_date).cast("date")
        follow_up_end_date = date_add(vantage_date_lit, follow_up_days)
        
        # Identify terminations within follow-up period
        future_terminations = df_with_id.join(active_employees, self.person_id_combined).filter(
            (col("work_asgnmt_stus_cd") == "T") &
            (col("rec_eff_start_dt_mod") > vantage_date_lit) &
            (col("rec_eff_start_dt_mod") <= follow_up_end_date)
        ).groupBy(self.person_id_combined).agg(
            min("rec_eff_start_dt_mod").alias("termination_date")
        ).withColumn(
            "survival_time_days",
            datediff(col("termination_date"), vantage_date_lit)
        ).withColumn(
            "event_indicator", lit(1)
        )
        
        # Add outcomes to active employee population
        outcomes = active_employees.join(future_terminations, self.person_id_combined, "left") \
            .withColumn(
                "survival_time_days",
                coalesce(col("survival_time_days"), lit(follow_up_days))
            ).withColumn(
                "event_indicator",
                coalesce(col("event_indicator"), lit(0))
            )
        
        return outcomes
    
    def create_dataset_splits(self, active_employees_2023: DataFrame,
                            active_employees_2024: DataFrame,
                            train_ratio: float = 0.7,
                            random_seed: int = 42) -> DataFrame:
        """
        Create temporal validation splits with proper person-level separation
        Ensures no person appears across different split types
        """
        
        # Train/Val split from 2023 population
        unique_persons_2023 = active_employees_2023.select(self.person_id_combined).distinct()
        
        # Sample persons for training
        train_persons = unique_persons_2023.sample(train_ratio, seed=random_seed)
        val_persons = unique_persons_2023.join(train_persons, self.person_id_combined, "left_anti")
        
        # OOT population from 2024
        oot_persons = active_employees_2024.select(self.person_id_combined).distinct()
        
        # Create split assignments
        split_assignments = train_persons.withColumn("dataset_split", lit("train")) \
            .union(val_persons.withColumn("dataset_split", lit("val")) ) \
            .union(oot_persons.withColumn("dataset_split", lit("oot"))  )
        # TODO commment
        # Verify no person leakage across splits
        person_counts = split_assignments.groupBy(self.person_id_combined).count() \
                                       .filter(col("count") > 1).count()
        if person_counts > 0:
            logger.warning(f"Person leakage detected: {person_counts} persons in multiple splits")
        
        logger.info("Dataset split summary:")
        split_summary = split_assignments.groupBy("dataset_split").count().collect()
        for row in split_summary:
            logger.info(f"  {row['dataset_split']}: {row['count']:,} persons")
        # TODO commment
        return split_assignments
    
    def create_start_stop_dataset(self, df: DataFrame, 
                                split_assignments: DataFrame,
                                vantage_dates: dict,
                                follow_up_days: int = 365,
                                compressed: bool = False) -> DataFrame:
        """
        Create start-stop format dataset with proper temporal boundaries
        Uses only historical episodes up to respective vantage dates
        """
        
        df_with_id = self.create_composite_person_id(df)
        results = []
        
        for split_type, vantage_date in vantage_dates.items():
            vantage_date_lit = lit(vantage_date).cast("date")
            
            # Get persons in this split
            split_persons = split_assignments.filter(col("dataset_split") == split_type) \
                                           .select(self.person_id_combined)
            
            # Historical episodes up to vantage date only
            historical_episodes = df_with_id.join(split_persons, self.person_id_combined) \
                .filter(col("rec_eff_start_dt_mod") <= vantage_date_lit)
            
            # Apply compression if requested
            if compressed:
                historical_episodes = self.compress_episodes(historical_episodes)
            
            # Get hire dates for time reference
            hire_dates = historical_episodes.groupBy(self.person_id_combined).agg(
                min("rec_eff_start_dt_mod").alias("hire_date")
            )
            
            # Create start-stop time variables relative to hire date
            start_stop_episodes = historical_episodes.join(hire_dates, self.person_id_combined) \
                .withColumn(
                    "start_time",
                    datediff(col("rec_eff_start_dt_mod"), col("hire_date"))
                ).withColumn(
                    "stop_time",
                    when(
                        col("rec_eff_end_dt") <= vantage_date_lit,
                        datediff(col("rec_eff_end_dt"), col("hire_date")) + 1
                    ).otherwise(
                        datediff(vantage_date_lit, col("hire_date"))
                    )
                ).filter(col("start_time") < col("stop_time"))
            
            # Calculate survival outcomes for target variable
            active_employees = split_persons.join(
                df_with_id.filter(
                    (col("rec_eff_start_dt_mod") <= vantage_date_lit) &
                    (col("rec_eff_end_dt") >= vantage_date_lit) &
                    (col("work_asgnmt_stus_cd") == "A")
                ).select(self.person_id_combined).distinct(),
                self.person_id_combined
            )
            
            # Adjust follow-up period for data availability
            actual_follow_up = follow_up_days
            if split_type == "oot":
                # For 2024-01-01 vantage, only have data till end of 2024
                actual_follow_up = min(follow_up_days, 364)
            
            outcomes = self.get_survival_outcomes(
                df_with_id, active_employees, vantage_date, actual_follow_up
            )
            
            # Add event indicator to final episode for each person
            final_episodes = start_stop_episodes.withColumn(
                "episode_rank",
                row_number().over(
                    Window.partitionBy(self.person_id_combined)
                          .orderBy(desc("stop_time"))
                )
            ).join(outcomes, self.person_id_combined, "left") \
             .withColumn(
                "event_in_episode",
                when(col("episode_rank") == 1, coalesce(col("event_indicator"), lit(0)))
                .otherwise(0)
            ).withColumn("dataset_split", lit(split_type)) \
             .withColumn("vantage_date", lit(vantage_date)) \
             .withColumn("compressed", lit(compressed))
            
            results.append(final_episodes)
        
        # Combine all splits
        combined_dataset = results[0]
        for result in results[1:]:
            combined_dataset = combined_dataset.union(result)
        
        return combined_dataset
    
    def create_employee_level_dataset(self, df: DataFrame,
                                    split_assignments: DataFrame,
                                    vantage_dates: dict,
                                    follow_up_days: int = 365) -> DataFrame:
        """
        Create employee-level dataset with one row per employee per split
        Captures baseline features and survival outcomes
        """
        
        df_with_id = self.create_composite_person_id(df)
        results = []
        
        for split_type, vantage_date in vantage_dates.items():
            # Get persons in this split
            split_persons = split_assignments.filter(col("dataset_split") == split_type) \
                                           .select(self.person_id_combined)
            
            # Get active employees on vantage date
            active_employees = split_persons.join(
                df_with_id.filter(
                    (col("rec_eff_start_dt_mod") <= lit(vantage_date).cast("date")) &
                    (col("rec_eff_end_dt") >= lit(vantage_date).cast("date")) &
                    (col("work_asgnmt_stus_cd") == "A")
                ).select(self.person_id_combined).distinct(),
                self.person_id_combined
            )
            
            # Create baseline features
            baseline_features = self.create_baseline_features(df_with_id, vantage_date)
            
            # Calculate survival outcomes
            actual_follow_up = follow_up_days
            if split_type == "oot":
                actual_follow_up = min(follow_up_days, 364)
            
            outcomes = self.get_survival_outcomes(
                df_with_id, active_employees, vantage_date, actual_follow_up
            )
            
            # Combine features and outcomes
            employee_dataset = baseline_features.join(active_employees, self.person_id_combined) \
                .join(outcomes, self.person_id_combined, "left") \
                .withColumn("dataset_split", lit(split_type)) \
                .withColumn("vantage_date", lit(vantage_date))
            
            results.append(employee_dataset)
        
        # Combine all splits
        combined_dataset = results[0]
        for result in results[1:]:
            combined_dataset = combined_dataset.union(result)
        
        return combined_dataset

def run_comprehensive_survival_pipeline(spark: SparkSession,
                                       input_df: DataFrame,
                                       gap_threshold_days: int = 30,
                                       use_latest_cycle_only: bool = True,
                                       train_ratio: float = 0.7,
                                       random_seed: int = 42) -> dict:
    """
    Execute comprehensive survival analysis pipeline
    Creates four dataset types with proper temporal validation
    """
    
    logger.info("Starting comprehensive survival analysis pipeline")
    
    processor = SurvivalDataProcessor(spark)
    
    # Phase 1: Data cleaning and preprocessing
    logger.info("Phase 1: Cleaning and preprocessing data")
    cleaned_terminations = processor.clean_termination_records(input_df)
    gap_filled_data = processor.fill_short_gaps(cleaned_terminations, gap_threshold_days=gap_threshold_days)
    
    if use_latest_cycle_only:
        modeling_data = processor.get_latest_cycle_records(gap_filled_data)
    else:
        modeling_data = processor.identify_employment_cycles(gap_filled_data)
    
    # Phase 2: Define vantage dates and populations
    vantage_dates = {
        "train": "2023-01-01",
        "val": "2023-01-01", 
        "oot": "2024-01-01"
    }
    
    df_with_id = processor.create_composite_person_id(modeling_data)
    
    # Get active populations for each vantage date
    active_2023 = df_with_id.filter(
        (col("rec_eff_start_dt_mod") <= lit("2023-01-01").cast("date")) &
        (col("rec_eff_end_dt") >= lit("2023-01-01").cast("date")) &
        (col("work_asgnmt_stus_cd") == "A")
    ).select(processor.person_id_combined).distinct()
    
    active_2024 = df_with_id.filter(
        (col("rec_eff_start_dt_mod") <= lit("2024-01-01").cast("date")) &
        (col("rec_eff_end_dt") >= lit("2024-01-01").cast("date")) &
        (col("work_asgnmt_stus_cd") == "A")
    ).select(processor.person_id_combined).distinct()
    
    # Phase 3: Create dataset splits
    logger.info("Phase 3: Creating dataset splits")
    split_assignments = processor.create_dataset_splits(
        active_2023, active_2024, train_ratio, random_seed
    )
    
    # Phase 4: Generate four dataset formats
    logger.info("Phase 4: Creating survival datasets")
    
    # Dataset 1: Start-stop without compression
    start_stop_uncompressed = processor.create_start_stop_dataset(
        modeling_data, split_assignments, vantage_dates, compressed=False
    )
    
    # Dataset 2: Start-stop with compression
    start_stop_compressed = processor.create_start_stop_dataset(
        modeling_data, split_assignments, vantage_dates, compressed=True
    )
    
    # Dataset 3: Employee-level
    employee_level = processor.create_employee_level_dataset(
        modeling_data, split_assignments, vantage_dates
    )
    
    logger.info("Pipeline completed successfully")
    
    return {
        "start_stop_uncompressed": start_stop_uncompressed,
        "start_stop_compressed": start_stop_compressed,
        "employee_level": employee_level,
        "split_assignments": split_assignments,
        "cleaned_data": modeling_data,
        "metadata": {
            "vantage_dates": vantage_dates,
            "train_ratio": train_ratio,
            "gap_threshold_days": gap_threshold_days,
            "use_latest_cycle_only": use_latest_cycle_only
        }
    }

# Usage example
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("ComprehensiveSurvivalAnalysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Load data and run pipeline
    # input_df = spark.read.table("your_schema.employee_episodes")
    # results = run_comprehensive_survival_pipeline(spark, input_df)
    
    spark.stop()
# 
# GET SPECIFIC PERSON IDS
# 
from pyspark.sql.functions import col, row_number, asc, desc, rand
from pyspark.sql.window import Window
import random

# Get unique person_composite_id across splits
unique_ids = results['split_assignments'].select('person_composite_id').distinct()

# Join with cleaned data
joined_df = unique_ids.join(results['cleaned_data'], on='person_composite_id', how='inner')

# Initialize list to track selected IDs and final results
selected_ids = []
final_results = []

# 1. Get 1 record of top most annual_campaign_amount
remaining_df = joined_df
top_annual = remaining_df.orderBy(desc('annual_campaign_amount')).limit(1)
top_annual_ids = [row.person_composite_id for row in top_annual.collect()]
selected_ids.extend(top_annual_ids)
final_results.append(top_annual)

# 2. Get 1 record of top most monthly_campaign_amount (excluding already selected)
remaining_df = joined_df.filter(~col('person_composite_id').isin(selected_ids))
top_monthly = remaining_df.orderBy(desc('monthly_campaign_amount')).limit(1)
top_monthly_ids = [row.person_composite_id for row in top_monthly.collect()]
selected_ids.extend(top_monthly_ids)
final_results.append(top_monthly)

# 3. Get 2 records: 1 with top most episode_duration_days, 1 with least episode_duration_days
remaining_df = joined_df.filter(~col('person_composite_id').isin(selected_ids))

# Get top episode duration
top_episode = remaining_df.orderBy(desc('episode_duration_days')).limit(1)
top_episode_ids = [row.person_composite_id for row in top_episode.collect()]
selected_ids.extend(top_episode_ids)
final_results.append(top_episode)

# Get least episode duration (excluding the top one we just selected)
remaining_df = joined_df.filter(~col('person_composite_id').isin(selected_ids))
least_episode = remaining_df.orderBy(asc('episode_duration_days')).limit(1)
least_episode_ids = [row.person_composite_id for row in least_episode.collect()]
selected_ids.extend(least_episode_ids)
final_results.append(least_episode)

# 4. Get 2 records each for youngest and oldest based on birth_dt (total 4 records)
remaining_df = joined_df.filter(~col('person_composite_id').isin(selected_ids))

# Get 2 youngest (highest birth_dt values)
youngest = remaining_df.orderBy(desc('birth_dt')).limit(2)
youngest_ids = [row.person_composite_id for row in youngest.collect()]
selected_ids.extend(youngest_ids)
final_results.append(youngest)

# Get 2 oldest (lowest birth_dt values) - excluding already selected
remaining_df = joined_df.filter(~col('person_composite_id').isin(selected_ids))
oldest = remaining_df.orderBy(asc('birth_dt')).limit(2)
oldest_ids = [row.person_composite_id for row in oldest.collect()]
selected_ids.extend(oldest_ids)
final_results.append(oldest)

# 5. Get 4 random IDs from remaining
remaining_df = joined_df.filter(~col('person_composite_id').isin(selected_ids))
random_ids = remaining_df.orderBy(rand()).limit(4)
random_ids_list = [row.person_composite_id for row in random_ids.collect()]
selected_ids.extend(random_ids_list)
final_results.append(random_ids)

# Combine all results
from functools import reduce
person_composite_id_df = reduce(lambda df1, df2: df1.union(df2), final_results)

# Verify we have the right number of records
print(f"Total selected IDs: {person_composite_id_df.count()}")
print(f"Expected: 12 (1+1+2+2+2+4)")

# Show the final results
display(person_composite_id_df.select('person_composite_id').distinct().orderBy('person_composite_id'))

# Optional: Show breakdown by category
print("\nBreakdown:")
print(f"Top Annual: {len(top_annual_ids)} IDs")
print(f"Top Monthly: {len(top_monthly_ids)} IDs") 
print(f"Top Episode Duration: {len(top_episode_ids)} IDs")
print(f"Least Episode Duration: {len(least_episode_ids)} IDs")
print(f"Youngest: {len(youngest_ids)} IDs")
print(f"Oldest: {len(oldest_ids)} IDs")
print(f"Random: {len(random_ids_list)} IDs")
print(f"Total unique IDs: {len(set(selected_ids))}")