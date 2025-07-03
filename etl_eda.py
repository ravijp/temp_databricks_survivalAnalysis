"""
Employee Turnover Survival Analysis Data Preparation Pipeline
Handles start-stop format data transformation for Cox PH and XGBoost AFT models
Author: Senior ML Engineer
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurvivalDataProcessor:
    """
    Comprehensive survival analysis data preparation pipeline
    Handles multi-level employee identification and time-varying covariate compression
    """
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.person_id_cols = ["db_schema", "clnt_obj_id", "pers_obj_id"]
        self.person_id_combined = "person_composite_id"
        
    def create_composite_person_id(self, df: DataFrame) -> DataFrame:
        """
        Create composite person identifier from schema, client, and person IDs
        Ensures unique identification across database schemas and clients
        """
        return df.withColumn(
            self.person_id_combined,
            concat_ws("_", *self.person_id_cols)
        )
    
    def analyze_field_changes_grouped(self, df: DataFrame, 
                                    fields_to_analyze: list = None,
                                    date_col: str = "rec_eff_strt_dt_mod") -> DataFrame:
        """
        Analyze field changes at episode level rather than individual field level
        Groups consecutive records with identical field combinations
        This determines optimal compression strategy for start-stop format
        """
        
        if fields_to_analyze is None:
            fields_to_analyze = [
                "work_assgmt_stus_cd", "mngr_pers_obj_id", "hrly_cmpn_amt", 
                "annl_cmpn_amt", "naics_cd", "job_dsc", "eeo1_job_cat_cd", 
                "team_size", "team_avg_comp"
            ]
        
        # Filter to only include fields that exist in the dataframe
        existing_fields = [f for f in fields_to_analyze if f in df.columns]
        
        if not existing_fields:
            logger.warning("No analyzable fields found in dataframe")
            return self.spark.createDataFrame([], StructType([]))
        
        logger.info(f"Analyzing field changes for: {existing_fields}")
        
        df_with_id = self.create_composite_person_id(df)
        person_window = Window.partitionBy(self.person_id_combined).orderBy(date_col)
        
        # Create hash of all field values to detect when ANY field changes
        field_concat = concat_ws("|", *[coalesce(col(f).cast("string"), lit("NULL")) 
                                       for f in existing_fields])
        
        df_with_hash = df_with_id.withColumn("field_combination_hash", 
                                           sha2(field_concat, 256))
        
        # Detect changes in field combinations
        df_with_changes = df_with_hash.withColumn(
            "prev_hash", 
            lag("field_combination_hash").over(person_window)
        ).withColumn(
            "field_combo_changed",
            when(col("prev_hash").isNotNull() & 
                 (col("field_combination_hash") != col("prev_hash")), 1)
            .otherwise(0)
        )
        
        # Create episode groups based on field combination changes
        df_with_episodes = df_with_changes.withColumn(
            "episode_start",
            when(col("field_combo_changed") == 1, 1).otherwise(0)
        ).withColumn(
            "episode_id",
            sum("episode_start").over(person_window.rowsBetween(Window.unboundedPreceding, 0))
        )
        
        # Analyze compression potential
        episode_stats = df_with_episodes.groupBy(
            self.person_id_combined, "episode_id"
        ).agg(
            count("*").alias("records_in_episode"),
            min(date_col).alias("episode_start_date"),
            max(date_col).alias("episode_end_date"),
            first("field_combination_hash").alias("episode_hash")
        )
        
        # Calculate compression metrics
        compression_analysis = episode_stats.agg(
            count("*").alias("total_episodes"),
            sum("records_in_episode").alias("total_records"),
            avg("records_in_episode").alias("avg_records_per_episode"),
            expr("percentile_approx(records_in_episode, array(0.5, 0.75, 0.9, 0.95))").alias("record_percentiles")
        )
        
        # Person-level statistics
        person_stats = episode_stats.groupBy(self.person_id_combined).agg(
            count("*").alias("episodes_per_person"),
            sum("records_in_episode").alias("total_records_per_person")
        ).withColumn(
            "compression_ratio_per_person",
            1.0 - (col("episodes_per_person") / col("total_records_per_person"))
        )
        
        person_summary = person_stats.agg(
            count("*").alias("total_employees"),
            avg("episodes_per_person").alias("avg_episodes_per_person"),
            avg("compression_ratio_per_person").alias("avg_compression_ratio"),
            expr("percentile_approx(compression_ratio_per_person, 0.5)").alias("median_compression_ratio")
        )
        
        # Combine results
        compression_summary = compression_analysis.crossJoin(person_summary)
        
        logger.info("Field change analysis completed")
        return compression_summary
    
    def compression_impact_analysis(self, df: DataFrame) -> dict:
        """
        Comprehensive analysis of data compression potential
        Considers termination record patterns and episode grouping
        """
        
        df_with_id = self.create_composite_person_id(df)
        
        # Original record statistics
        original_stats = df_with_id.agg(
            count("*").alias("total_records"),
            countDistinct(self.person_id_combined).alias("unique_employees")
        ).collect()[0]
        
        # Termination pattern analysis
        termination_patterns = df_with_id.groupBy(
            self.person_id_combined, "work_assgmt_stus_cd"
        ).agg(
            count("*").alias("status_count")
        ).groupBy(self.person_id_combined).agg(
            collect_list("work_assgmt_stus_cd").alias("status_sequence"),
            sum("status_count").alias("total_records_per_person")
        )
        
        # Count consecutive termination groups
        termination_compression = termination_patterns.withColumn(
            "status_string", 
            array_join(col("status_sequence"), "")
        ).withColumn(
            "consecutive_t_groups",
            size(split(regexp_replace(col("status_string"), "T+", "T"), "T")) - 1
        ).agg(
            sum("consecutive_t_groups").alias("total_t_groups"),
            avg("consecutive_t_groups").alias("avg_t_groups_per_person")
        ).collect()[0]
        
        # Calculate potential compression
        potential_savings = original_stats["total_records"] - termination_compression["total_t_groups"]
        
        return {
            "original_records": original_stats["total_records"],
            "unique_employees": original_stats["unique_employees"],
            "avg_records_per_employee": original_stats["total_records"] / original_stats["unique_employees"],
            "potential_t_record_savings": termination_compression["total_t_groups"],
            "estimated_compression_ratio": potential_savings / original_stats["total_records"],
            "avg_t_groups_per_person": float(termination_compression["avg_t_groups_per_person"])
        }
    
    def clean_termination_records(self, df: DataFrame,
                                status_col: str = "work_assgmt_stus_cd",
                                start_date_col: str = "rec_eff_strt_dt_mod",
                                end_date_col: str = "rec_eff_end_dt") -> DataFrame:
        """
        Consolidate consecutive termination records into single episodes
        Preserves earliest termination date and extends to final end date
        Critical for accurate survival time calculation
        """
        
        df_with_id = self.create_composite_person_id(df)
        person_window = Window.partitionBy(self.person_id_combined).orderBy(start_date_col)
        
        # Identify termination record groups
        df_with_groups = df_with_id.withColumn(
            "is_termination", 
            when(col(status_col) == "T", 1).otherwise(0)
        ).withColumn(
            "prev_is_termination", 
            lag("is_termination").over(person_window)
        ).withColumn(
            "termination_group_start",
            when((col("is_termination") == 1) & 
                 (coalesce(col("prev_is_termination"), lit(0)) == 0), 1)
            .otherwise(0)
        ).withColumn(
            "termination_group_id",
            sum("termination_group_start").over(
                person_window.rowsBetween(Window.unboundedPreceding, 0)
            )
        )
        
        # Calculate group boundaries for termination records
        termination_groups = df_with_groups.filter(
            col("is_termination") == 1
        ).groupBy(
            self.person_id_combined, "termination_group_id"
        ).agg(
            min(start_date_col).alias("group_start_date"),
            max(end_date_col).alias("group_end_date"),
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
        
        logger.info(f"Termination record consolidation: {df.count()} -> {result.count()} records")
        return result
    
    def fill_short_gaps(self, df: DataFrame,
                       status_col: str = "work_assgmt_stus_cd",
                       start_date_col: str = "rec_eff_strt_dt_mod", 
                       end_date_col: str = "rec_eff_end_dt",
                       gap_threshold_days: int = 30) -> DataFrame:
        """
        Bridge short employment gaps by extending active periods
        Handles administrative terminations and brief unemployment periods
        """
        
        df_with_id = self.create_composite_person_id(df)
        person_window = Window.partitionBy(self.person_id_combined).orderBy(start_date_col)
        
        # Add context from adjacent records
        df_with_context = df_with_id.withColumn(
            "prev_status", lag(status_col).over(person_window)
        ).withColumn(
            "next_status", lead(status_col).over(person_window)
        ).withColumn(
            "prev_end_date", lag(end_date_col).over(person_window)
        ).withColumn(
            "next_start_date", lead(start_date_col).over(person_window)
        ).withColumn(
            "gap_duration",
            when(col(status_col) == "T", 
                 datediff(col(end_date_col), col(start_date_col)) + 1)
            .otherwise(0)
        )
        
        # Identify short gaps to fill (A -> T -> A pattern with short T duration)
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
                (lead("is_short_gap").over(person_window) == 1),
                1
            ).otherwise(0)
        )
        
        # Extend active periods over short gaps
        df_extended = df_with_flags.withColumn(
            end_date_col,
            when(
                col("extend_previous_active") == 1,
                date_sub(lead(lead(start_date_col).over(person_window)).over(person_window), 1)
            ).otherwise(col(end_date_col))
        )
        
        # Remove short gap records
        result = df_extended.filter(col("is_short_gap") == 0).select(df.columns)
        
        logger.info(f"Short gap filling completed: {df.count()} -> {result.count()} records")
        return result
    
    def create_survival_datasets(self, df: DataFrame,
                               vantage_date: str = "2022-01-01",
                               follow_up_days: int = 365) -> dict:
        """
        Generate model-ready datasets for different survival analysis approaches
        Handles right-censoring and time-varying covariate formatting
        """
        
        vantage_date_lit = lit(vantage_date).cast("date")
        follow_up_end_date = date_add(vantage_date_lit, follow_up_days)
        
        df_with_id = self.create_composite_person_id(df)
        
        # Identify employees active on vantage date
        active_employees = df_with_id.filter(
            (col("rec_eff_strt_dt_mod") <= vantage_date_lit) &
            (col("rec_eff_end_dt") >= vantage_date_lit)
        ).select(self.person_id_combined).distinct()
        
        logger.info(f"Found {active_employees.count()} employees active on {vantage_date}")
        
        # Generate different dataset formats
        employee_level_data = self._create_employee_level_dataset(
            df_with_id, active_employees, vantage_date, follow_up_days
        )
        
        start_stop_data = self._create_start_stop_dataset(
            df_with_id, active_employees, vantage_date, follow_up_days
        )
        
        landmark_data = self._create_landmark_datasets(
            df_with_id, active_employees, vantage_date, [90, 180, 270]
        )
        
        return {
            "employee_level": employee_level_data,
            "start_stop": start_stop_data,
            "landmark": landmark_data,
            "metadata": {
                "vantage_date": vantage_date,
                "follow_up_days": follow_up_days,
                "active_employees": active_employees.count()
            }
        }
    
    def _create_employee_level_dataset(self, df: DataFrame, 
                                     active_employees: DataFrame,
                                     vantage_date: str, 
                                     follow_up_days: int) -> DataFrame:
        """
        Create one-row-per-employee dataset for Kaplan-Meier, standard Cox, and XGBoost AFT
        Captures baseline characteristics and survival outcome
        """
        
        vantage_date_lit = lit(vantage_date).cast("date")
        follow_up_end_date = date_add(vantage_date_lit, follow_up_days)
        
        # Get baseline characteristics as of vantage date
        baseline_window = Window.partitionBy(self.person_id_combined).orderBy(desc("rec_eff_strt_dt_mod"))
        
        baseline_features = df.join(active_employees, self.person_id_combined).filter(
            col("rec_eff_strt_dt_mod") <= vantage_date_lit
        ).withColumn(
            "baseline_rank", row_number().over(baseline_window)
        ).filter(col("baseline_rank") == 1)
        
        # Identify termination events within follow-up period
        termination_events = df.filter(
            (col("work_assgmt_stus_cd") == "T") &
            (col("rec_eff_strt_dt_mod") > vantage_date_lit) &
            (col("rec_eff_strt_dt_mod") <= follow_up_end_date)
        ).groupBy(self.person_id_combined).agg(
            min("rec_eff_strt_dt_mod").alias("termination_date")
        ).withColumn(
            "days_to_termination",
            datediff(col("termination_date"), vantage_date_lit)
        ).withColumn("event_occurred", lit(1))
        
        # Combine baseline features with outcomes
        survival_dataset = baseline_features.join(
            termination_events, self.person_id_combined, "left"
        ).withColumn(
            "survival_time_days",
            coalesce(col("days_to_termination"), lit(follow_up_days))
        ).withColumn(
            "event_indicator",
            coalesce(col("event_occurred"), lit(0))
        ).withColumn(
            "tenure_at_vantage_days",
            datediff(vantage_date_lit, col("rec_eff_strt_dt_mod"))
        )
        
        return survival_dataset
    
    def _create_start_stop_dataset(self, df: DataFrame,
                                 active_employees: DataFrame,
                                 vantage_date: str,
                                 follow_up_days: int) -> DataFrame:
        """
        Create start-stop format for time-varying Cox models
        Each row represents an episode with constant covariate values
        """
        
        vantage_date_lit = lit(vantage_date).cast("date")
        follow_up_end_date = date_add(vantage_date_lit, follow_up_days)
        
        # Get all episodes within follow-up period for active employees
        episodes = df.join(active_employees, self.person_id_combined).filter(
            (col("rec_eff_strt_dt_mod") >= vantage_date_lit) &
            (col("rec_eff_strt_dt_mod") <= follow_up_end_date)
        )
        
        # Convert to start-stop format with relative time
        start_stop_episodes = episodes.withColumn(
            "start_time",
            greatest(lit(0), datediff(col("rec_eff_strt_dt_mod"), vantage_date_lit))
        ).withColumn(
            "stop_time", 
            least(lit(follow_up_days), 
                  datediff(col("rec_eff_end_dt"), vantage_date_lit) + 1)
        ).withColumn(
            "event_in_episode",
            when((col("work_assgmt_stus_cd") == "T") & 
                 (col("rec_eff_end_dt") <= follow_up_end_date), 1)
            .otherwise(0)
        ).filter(col("start_time") < col("stop_time"))
        
        return start_stop_episodes
    
    def _create_landmark_datasets(self, df: DataFrame,
                                active_employees: DataFrame,
                                vantage_date: str,
                                landmark_days: list) -> dict:
        """
        Create landmark analysis datasets for time-dependent risk assessment
        
        Landmark analysis: Among employees who survive to time t, 
        what's the probability of surviving to time t+s?
        
        Example: Among employees who survive 90 days, who survives to 365 days?
        This captures changing risk profiles over employment tenure
        """
        
        landmark_datasets = {}
        vantage_date_lit = lit(vantage_date).cast("date")
        
        for landmark_day in landmark_days:
            landmark_date = date_add(vantage_date_lit, landmark_day)
            
            # Employees who survived to landmark date (conditional survivors)
            survivors_at_landmark = df.join(active_employees, self.person_id_combined).filter(
                (col("work_assgmt_stus_cd") == "T") &
                (col("rec_eff_strt_dt_mod") <= landmark_date)
            ).select(self.person_id_combined).distinct()
            
            # Remove employees who terminated before landmark
            conditional_survivors = active_employees.join(
                survivors_at_landmark, self.person_id_combined, "left_anti"
            )
            
            # Get features as of landmark date
            landmark_features = df.filter(
                col("rec_eff_strt_dt_mod") <= landmark_date
            ).withColumn(
                "rank_at_landmark",
                row_number().over(
                    Window.partitionBy(self.person_id_combined)
                          .orderBy(desc("rec_eff_strt_dt_mod"))
                )
            ).filter(col("rank_at_landmark") == 1)
            
            # Create survival dataset from landmark forward
            landmark_survival = self._create_employee_level_dataset(
                df, conditional_survivors, 
                (datetime.strptime(vantage_date, "%Y-%m-%d") + 
                 timedelta(days=landmark_day)).strftime("%Y-%m-%d"),
                365 - landmark_day
            )
            
            landmark_datasets[f"landmark_{landmark_day}d"] = landmark_survival
            
            logger.info(f"Landmark {landmark_day}d: {landmark_survival.count()} conditional survivors")
        
        return landmark_datasets
    
    def validate_survival_datasets(self, datasets: dict) -> dict:
        """
        Comprehensive validation of survival analysis datasets
        Checks for common data quality issues that affect model performance
        """
        
        validation_results = {}
        
        for dataset_name, dataset in datasets.items():
            if dataset_name == "metadata":
                continue
                
            if dataset_name == "landmark":
                landmark_validation = {}
                for landmark_name, landmark_df in dataset.items():
                    landmark_validation[landmark_name] = self._validate_single_dataset(
                        landmark_df, f"landmark_{landmark_name}"
                    )
                validation_results[dataset_name] = landmark_validation
            else:
                validation_results[dataset_name] = self._validate_single_dataset(
                    dataset, dataset_name
                )
        
        return validation_results
    
    def _validate_single_dataset(self, df: DataFrame, dataset_name: str) -> dict:
        """
        Validate individual dataset for survival analysis requirements
        """
        
        total_records = df.count()
        unique_employees = df.select(self.person_id_combined).distinct().count()
        
        validation_report = {
            "dataset_name": dataset_name,
            "total_records": total_records,
            "unique_employees": unique_employees,
            "validation_errors": [],
            "validation_warnings": []
        }
        
        # Check survival time validity
        if "survival_time_days" in df.columns:
            invalid_survival_times = df.filter(
                col("survival_time_days") <= 0
            ).count()
            
            if invalid_survival_times > 0:
                validation_report["validation_errors"].append(
                    f"Found {invalid_survival_times} records with invalid survival times"
                )
            
            # Check event rate
            if "event_indicator" in df.columns:
                event_stats = df.groupBy("event_indicator").count().collect()
                event_distribution = {str(row[0]): row[1] for row in event_stats}
                
                total_events = event_distribution.get("1", 0)
                event_rate = total_events / total_records
                
                validation_report["event_rate"] = event_rate
                validation_report["event_distribution"] = event_distribution
                
                if event_rate < 0.01:
                    validation_report["validation_warnings"].append(
                        f"Very low event rate: {event_rate:.3f}"
                    )
                elif event_rate > 0.9:
                    validation_report["validation_warnings"].append(
                        f"Very high event rate: {event_rate:.3f}"
                    )
        
        # Check start-stop format validity
        if "start_time" in df.columns and "stop_time" in df.columns:
            invalid_intervals = df.filter(
                col("start_time") >= col("stop_time")
            ).count()
            
            if invalid_intervals > 0:
                validation_report["validation_errors"].append(
                    f"Found {invalid_intervals} invalid time intervals"
                )
            
            # Check for overlapping episodes
            person_window = Window.partitionBy(self.person_id_combined).orderBy("start_time")
            overlapping_episodes = df.withColumn(
                "prev_stop_time", lag("stop_time").over(person_window)
            ).filter(
                col("start_time") < col("prev_stop_time")
            ).count()
            
            if overlapping_episodes > 0:
                validation_report["validation_errors"].append(
                    f"Found {overlapping_episodes} overlapping episodes"
                )
        
        return validation_report
    
    def create_train_validation_splits(self, employee_df: DataFrame,
                                     train_ratio: float = 0.7,
                                     stratify_by: list = None,
                                     random_seed: int = 42) -> dict:
        """
        Create stratified train/validation splits ensuring no employee leakage
        Maintains representativeness across key dimensions
        """
        
        if stratify_by is None:
            stratify_by = ["event_indicator", "naics_cd"]
        
        # Create stratification variables
        stratified_df = employee_df
        
        # Add tenure quartiles for stratification
        if "tenure_at_vantage_days" in employee_df.columns:
            tenure_quartiles = employee_df.approxQuantile(
                "tenure_at_vantage_days", [0.25, 0.5, 0.75], 0.01
            )
            
            stratified_df = stratified_df.withColumn(
                "tenure_quartile",
                when(col("tenure_at_vantage_days") <= tenure_quartiles[0], "Q1")
                .when(col("tenure_at_vantage_days") <= tenure_quartiles[1], "Q2")
                .when(col("tenure_at_vantage_days") <= tenure_quartiles[2], "Q3")
                .otherwise("Q4")
            )
            
            stratify_by.append("tenure_quartile")
        
        # Create composite stratification key
        strata_key = concat_ws("_", *[coalesce(col(c).cast("string"), lit("NULL")) 
                                     for c in stratify_by if c in stratified_df.columns])
        
        stratified_df = stratified_df.withColumn("strata", strata_key)
        
        # Get unique strata for sampling
        unique_strata = [row[0] for row in stratified_df.select("strata").distinct().collect()]
        
        # Create sampling fractions
        sampling_fractions = {stratum: train_ratio for stratum in unique_strata}
        
        # Sample for training set
        train_df = stratified_df.sampleBy("strata", sampling_fractions, seed=random_seed)
        
        # Validation set is the remainder
        validation_df = stratified_df.join(
            train_df.select(self.person_id_combined), 
            self.person_id_combined, 
            "left_anti"
        )
        
        # Calculate split statistics
        split_stats = {
            "total_employees": stratified_df.count(),
            "train_employees": train_df.count(),
            "validation_employees": validation_df.count(),
            "train_events": train_df.filter(col("event_indicator") == 1).count(),
            "validation_events": validation_df.filter(col("event_indicator") == 1).count()
        }
        
        split_stats["train_event_rate"] = split_stats["train_events"] / split_stats["train_employees"]
        split_stats["validation_event_rate"] = split_stats["validation_events"] / split_stats["validation_employees"]
        
        return {
            "train": train_df,
            "validation": validation_df,
            "split_statistics": split_stats
        }

def run_survival_data_pipeline(spark: SparkSession,
                              input_df: DataFrame,
                              vantage_date: str = "2022-01-01",
                              follow_up_days: int = 365,
                              gap_threshold_days: int = 30,
                              output_path: str = None) -> dict:
    """
    Execute complete survival analysis data preparation pipeline
    Production-ready implementation with comprehensive logging and validation
    """
    
    logger.info("Starting survival analysis data preparation pipeline")
    logger.info(f"Vantage date: {vantage_date}, Follow-up: {follow_up_days} days")
    
    processor = SurvivalDataProcessor(spark)
    
    # Phase 1: Data exploration and compression analysis
    logger.info("Phase 1: Analyzing data compression potential")
    
    field_change_analysis = processor.analyze_field_changes_grouped(
        input_df, 
        fields_to_analyze=["work_assgmt_stus_cd", "mngr_pers_obj_id", "annl_cmpn_amt", 
                          "job_dsc", "naics_cd", "team_size", "team_avg_comp"]
    )
    
    compression_stats = processor.compression_impact_analysis(input_df)
    
    logger.info(f"Compression analysis complete. Potential savings: {compression_stats['estimated_compression_ratio']:.2%}")
    
    # Phase 2: Data cleaning and preprocessing
    logger.info("Phase 2: Cleaning and preprocessing data")
    
    cleaned_terminations = processor.clean_termination_records(input_df)
    final_cleaned_data = processor.fill_short_gaps(
        cleaned_terminations, 
        gap_threshold_days=gap_threshold_days
    )
    
    logger.info(f"Data cleaning complete: {input_df.count()} -> {final_cleaned_data.count()} records")
    
    # Phase 3: Survival dataset creation
    logger.info("Phase 3: Creating survival analysis datasets")
    
    survival_datasets = processor.create_survival_datasets(
        final_cleaned_data,
        vantage_date=vantage_date,
        follow_up_days=follow_up_days
    )
    
    # Phase 4: Data validation
    logger.info("Phase 4: Validating datasets")
    
    validation_results = processor.validate_survival_datasets(survival_datasets)
    
    # Phase 5: Train/validation splits
    logger.info("Phase 5: Creating train/validation splits")
    
    data_splits = processor.create_train_validation_splits(
        survival_datasets["employee_level"],
        train_ratio=0.7,
        random_seed=42
    )
    
    # Phase 6: Output and logging
    logger.info("Phase 6: Finalizing pipeline results")
    
    pipeline_results = {
        "cleaned_data": final_cleaned_data,
        "survival_datasets": survival_datasets,
        "data_splits": data_splits,
        "validation_results": validation_results,
        "compression_analysis": {
            "field_changes": field_change_analysis,
            "compression_stats": compression_stats
        }
    }
    
    # Log summary statistics
    logger.info("Pipeline Summary:")
    logger.info(f"  - Original records: {input_df.count()}")
    logger.info(f"  - Cleaned records: {final_cleaned_data.count()}")
    logger.info(f"  - Active employees: {survival_datasets['metadata']['active_employees']}")
    logger.info(f"  - Train employees: {data_splits['split_statistics']['train_employees']}")
    logger.info(f"  - Validation employees: {data_splits['split_statistics']['validation_employees']}")
    
    # Save results if output path provided
    if output_path:
        logger.info(f"Saving results to {output_path}")
        # Implementation of save logic would go here
        
    logger.info("Survival analysis data preparation pipeline completed successfully")
    
    return pipeline_results

# Example usage:
if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("SurvivalAnalysisDataPrep") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Load your data
    # input_dataframe = spark.read.table("your_schema.employee_episodes")
    
    # Run pipeline
    # results = run_survival_data_pipeline(
    #     spark=spark,
    #     input_df=input_dataframe,
    #     vantage_date="2022-01-01",
    #     follow_up_days=365,
    #     gap_threshold_days=30
    # )
    
    spark.stop()

# Cell 1: Setup and Validation Framework
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataValidationSuite:
    """
    Efficient validation suite for large-scale survival analysis data preparation
    Optimized for Databricks notebooks with sampling and caching strategies
    """
    
    def __init__(self, spark_session, sample_size=1000000):
        self.spark = spark_session
        self.sample_size = sample_size
        self.validation_history = []
        
    def quick_profile(self, df, step_name="Unknown", sample_ratio=0.01):
        """
        Quick statistical profile suitable for 500M+ records
        Uses sampling for efficiency
        """
        print(f"\n{'='*60}")
        print(f"VALIDATION: {step_name}")
        print(f"{'='*60}")
        
        # Basic counts (fast operations)
        total_records = df.count()
        total_columns = len(df.columns)
        
        # Sample for detailed analysis
        if total_records > self.sample_size:
            sample_df = df.sample(False, sample_ratio, seed=42)
            print(f"Using sample of {sample_df.count():,} records ({sample_ratio*100:.1f}% of total)")
        else:
            sample_df = df
        
        # Core statistics
        unique_employees = df.select("pers_obj_id").distinct().count()
        
        profile = {
            "step_name": step_name,
            "total_records": total_records,
            "total_columns": total_columns,
            "unique_employees": unique_employees,
            "avg_records_per_employee": total_records / unique_employees,
            "timestamp": F.current_timestamp()
        }
        
        # Display summary
        print(f"Total Records: {total_records:,}")
        print(f"Unique Employees: {unique_employees:,}")
        print(f"Avg Records/Employee: {profile['avg_records_per_employee']:.2f}")
        print(f"Columns: {total_columns}")
        
        # Store for comparison
        self.validation_history.append(profile)
        
        return profile, sample_df
    
    def validate_termination_cleaning(self, original_df, cleaned_df):
        """
        Validate termination record cleaning results
        """
        print(f"\n{'='*60}")
        print("VALIDATION: Termination Record Cleaning")
        print(f"{'='*60}")
        
        # Record count changes
        original_count = original_df.count()
        cleaned_count = cleaned_df.count()
        records_removed = original_count - cleaned_count
        
        print(f"Original records: {original_count:,}")
        print(f"Cleaned records: {cleaned_count:,}")
        print(f"Records removed: {records_removed:,} ({records_removed/original_count*100:.2f}%)")
        
        # Termination pattern analysis
        original_t_patterns = original_df.filter(F.col("work_assgmt_stus_cd") == "T") \
                                       .groupBy("pers_obj_id") \
                                       .agg(F.count("*").alias("t_records")) \
                                       .groupBy("t_records") \
                                       .count() \
                                       .orderBy("t_records") \
                                       .collect()
        
        cleaned_t_patterns = cleaned_df.filter(F.col("work_assgmt_stus_cd") == "T") \
                                     .groupBy("pers_obj_id") \
                                     .agg(F.count("*").alias("t_records")) \
                                     .groupBy("t_records") \
                                     .count() \
                                     .orderBy("t_records") \
                                     .collect()
        
        print("\nTermination Records per Employee (Before vs After):")
        print("T-Records | Original | Cleaned")
        print("-" * 35)
        
        orig_dict = {row['t_records']: row['count'] for row in original_t_patterns}
        clean_dict = {row['t_records']: row['count'] for row in cleaned_t_patterns}
        
        for t_count in range(1, 6):  # Show patterns 1-5
            orig_val = orig_dict.get(t_count, 0)
            clean_val = clean_dict.get(t_count, 0)
            print(f"    {t_count}     | {orig_val:,}    | {clean_val:,}")
        
        # Check for data integrity issues
        integrity_checks = self._check_data_integrity(cleaned_df)
        if integrity_checks:
            print("\n🚨 DATA INTEGRITY ISSUES:")
            for issue in integrity_checks:
                print(f"  - {issue}")
        else:
            print("\n✅ No data integrity issues detected")
    
    def validate_gap_filling(self, before_df, after_df, gap_threshold=30):
        """
        Validate short gap filling results
        """
        print(f"\n{'='*60}")
        print("VALIDATION: Gap Filling")
        print(f"{'='*60}")
        
        # Record count changes
        before_count = before_df.count()
        after_count = after_df.count()
        gaps_filled = before_count - after_count
        
        print(f"Before gap filling: {before_count:,}")
        print(f"After gap filling: {after_count:,}")
        print(f"Gaps filled: {gaps_filled:,} ({gaps_filled/before_count*100:.2f}%)")
        
        # Analyze gap durations that were filled
        sample_df = before_df.sample(False, 0.01, seed=42)
        
        person_window = Window.partitionBy("pers_obj_id").orderBy("rec_eff_strt_dt_mod")
        
        gap_analysis = sample_df.withColumn(
            "prev_status", F.lag("work_assgmt_stus_cd").over(person_window)
        ).withColumn(
            "next_status", F.lead("work_assgmt_stus_cd").over(person_window)
        ).withColumn(
            "gap_days",
            F.when(F.col("work_assgmt_stus_cd") == "T",
                   F.datediff(F.col("rec_eff_end_dt"), F.col("rec_eff_strt_dt_mod")) + 1)
        ).filter(
            (F.col("work_assgmt_stus_cd") == "T") &
            (F.col("prev_status") == "A") &
            (F.col("next_status") == "A") &
            (F.col("gap_days") <= gap_threshold)
        )
        
        gap_stats = gap_analysis.agg(
            F.count("*").alias("total_fillable_gaps"),
            F.avg("gap_days").alias("avg_gap_duration"),
            F.min("gap_days").alias("min_gap_duration"),
            F.max("gap_days").alias("max_gap_duration")
        ).collect()[0]
        
        print(f"\nGap Analysis (sampled):")
        print(f"Fillable gaps found: {gap_stats['total_fillable_gaps']:,}")
        print(f"Average gap duration: {gap_stats['avg_gap_duration']:.1f} days")
        print(f"Gap duration range: {gap_stats['min_gap_duration']} - {gap_stats['max_gap_duration']} days")
    
    def validate_survival_datasets(self, datasets_dict):
        """
        Validate survival analysis datasets
        """
        print(f"\n{'='*60}")
        print("VALIDATION: Survival Datasets")
        print(f"{'='*60}")
        
        for dataset_name, dataset in datasets_dict.items():
            if dataset_name == "metadata":
                continue
                
            print(f"\n--- {dataset_name.upper()} DATASET ---")
            
            if dataset_name == "landmark":
                for landmark_name, landmark_df in dataset.items():
                    self._validate_single_survival_dataset(landmark_df, f"Landmark {landmark_name}")
            else:
                self._validate_single_survival_dataset(dataset, dataset_name)
    
    def _validate_single_survival_dataset(self, df, dataset_name):
        """
        Validate individual survival dataset
        """
        total_records = df.count()
        unique_employees = df.select("pers_obj_id").distinct().count()
        
        print(f"\n{dataset_name}:")
        print(f"  Records: {total_records:,}")
        print(f"  Unique Employees: {unique_employees:,}")
        
        # Check survival-specific fields
        if "survival_time_days" in df.columns:
            survival_stats = df.agg(
                F.min("survival_time_days").alias("min_survival"),
                F.max("survival_time_days").alias("max_survival"),
                F.avg("survival_time_days").alias("avg_survival")
            ).collect()[0]
            
            print(f"  Survival Time - Min: {survival_stats['min_survival']}, Max: {survival_stats['max_survival']}, Avg: {survival_stats['avg_survival']:.1f}")
            
            # Check for invalid survival times
            invalid_survival = df.filter(F.col("survival_time_days") <= 0).count()
            if invalid_survival > 0:
                print(f"  ⚠️  Invalid survival times: {invalid_survival}")
        
        # Check event rates
        if "event_indicator" in df.columns:
            event_distribution = df.groupBy("event_indicator").count().collect()
            events = {row['event_indicator']: row['count'] for row in event_distribution}
            
            event_rate = events.get(1, 0) / total_records
            print(f"  Event Rate: {event_rate:.3f} ({events.get(1, 0):,} events)")
            
            if event_rate < 0.01:
                print(f"  ⚠️  Very low event rate: {event_rate:.3f}")
        
        # Check start-stop format
        if "start_time" in df.columns and "stop_time" in df.columns:
            invalid_intervals = df.filter(F.col("start_time") >= F.col("stop_time")).count()
            if invalid_intervals > 0:
                print(f"  ⚠️  Invalid intervals: {invalid_intervals}")
    
    def _check_data_integrity(self, df):
        """
        Check for common data integrity issues
        """
        issues = []
        
        # Sample for efficiency
        sample_df = df.sample(False, 0.01, seed=42)
        
        # Check for negative date differences
        negative_duration = sample_df.filter(
            F.col("rec_eff_strt_dt_mod") > F.col("rec_eff_end_dt")
        ).count()
        
        if negative_duration > 0:
            issues.append(f"Found {negative_duration} records with start_date > end_date")
        
        # Check for null critical fields
        for col_name in ["pers_obj_id", "work_assgmt_stus_cd", "rec_eff_strt_dt_mod"]:
            if col_name in df.columns:
                null_count = df.filter(F.col(col_name).isNull()).count()
                if null_count > 0:
                    issues.append(f"Found {null_count} null values in {col_name}")
        
        return issues
    
    def create_validation_dashboard(self):
        """
        Create visual dashboard of validation results
        """
        if not self.validation_history:
            print("No validation history available")
            return
        
        # Convert to pandas for visualization
        history_df = pd.DataFrame(self.validation_history)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Record counts by step
        axes[0, 0].bar(history_df['step_name'], history_df['total_records'])
        axes[0, 0].set_title('Total Records by Step')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Records per employee
        axes[0, 1].bar(history_df['step_name'], history_df['avg_records_per_employee'])
        axes[0, 1].set_title('Avg Records per Employee')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Employee counts
        axes[1, 0].bar(history_df['step_name'], history_df['unique_employees'])
        axes[1, 0].set_title('Unique Employees')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Compression ratio
        if len(history_df) > 1:
            baseline_records = history_df.iloc[0]['total_records']
            compression_ratios = [(baseline_records - row['total_records']) / baseline_records 
                                for _, row in history_df.iterrows()]
            axes[1, 1].plot(history_df['step_name'], compression_ratios, marker='o')
            axes[1, 1].set_title('Compression Ratio by Step')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Initialize validation suite
validator = DataValidationSuite(spark, sample_size=1000000)
