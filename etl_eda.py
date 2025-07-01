import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window
from datetime import datetime, timedelta
from typing import Dict, List


def analyze_etl_output(df: DataFrame) -> Dict[str, any]:
    """
    Analyze ETL output to provide insights for survival analysis design.
    
    Args:
        df: ETL output DataFrame
        
    Returns:
        Dictionary containing data insights for modeling decisions
    """
    
    insights = {}
    
    # Basic data dimensions
    insights['data_volume'] = get_data_volume_insights(df)
    
    # Time series characteristics
    insights['temporal_patterns'] = get_temporal_patterns(df)
    
    # Event analysis for survival modeling
    insights['event_analysis'] = get_event_analysis(df)
    
    # Client and industry distribution
    insights['client_distribution'] = get_client_distribution(df)
    
    # Feature availability and quality
    insights['feature_quality'] = get_feature_quality(df)
    
    # Data completeness patterns
    insights['completeness_patterns'] = get_completeness_patterns(df)
    
    # Survival analysis specific insights
    insights['survival_characteristics'] = get_survival_characteristics(df)
    
    return insights


def get_data_volume_insights(df: DataFrame) -> Dict[str, any]:
    """Analyze data volume and key entity counts."""
    
    total_records = df.count()
    unique_persons = df.select("pers_obj_id").distinct().count()
    unique_clients = df.select("clnt_obj_id").distinct().count()
    unique_work_assignments = df.select("pers_obj_id", "work_asgmt_nbr").distinct().count()
    
    # Records per person distribution
    records_per_person = (df.groupBy("pers_obj_id")
                         .count()
                         .select(
                             F.min("count").alias("min_records"),
                             F.max("count").alias("max_records"),
                             F.mean("count").alias("avg_records"),
                             F.expr("percentile_approx(count, 0.5)").alias("median_records"),
                             F.expr("percentile_approx(count, 0.9)").alias("p90_records")
                         )
                         .collect()[0])
    
    # Client size distribution
    client_sizes = (df.groupBy("clnt_obj_id")
                   .agg(F.countDistinct("pers_obj_id").alias("employee_count"))
                   .select(
                       F.min("employee_count").alias("min_employees"),
                       F.max("employee_count").alias("max_employees"),
                       F.mean("employee_count").alias("avg_employees"),
                       F.expr("percentile_approx(employee_count, 0.5)").alias("median_employees"),
                       F.expr("percentile_approx(employee_count, 0.9)").alias("p90_employees")
                   )
                   .collect()[0])
    
    return {
        'total_records': total_records,
        'unique_persons': unique_persons,
        'unique_clients': unique_clients,
        'unique_work_assignments': unique_work_assignments,
        'avg_records_per_person': records_per_person['avg_records'],
        'median_records_per_person': records_per_person['median_records'],
        'records_per_person_p90': records_per_person['p90_records'],
        'avg_employees_per_client': client_sizes['avg_employees'],
        'median_employees_per_client': client_sizes['median_employees'],
        'largest_client_size': client_sizes['max_employees']
    }


def get_temporal_patterns(df: DataFrame) -> Dict[str, any]:
    """Analyze temporal patterns in the data."""
    
    # Overall date range
    date_range = df.select(
        F.min("rec_eff_strt_dt").alias("earliest_start"),
        F.max("rec_eff_strt_dt").alias("latest_start"),
        F.min("rec_eff_end_dt").alias("earliest_end"),
        F.max("rec_eff_end_dt").alias("latest_end")
    ).collect()[0]
    
    # Records by year
    yearly_distribution = (df.withColumn("start_year", F.year("rec_eff_strt_dt"))
                          .groupBy("start_year")
                          .count()
                          .orderBy("start_year")
                          .collect())
    
    # Recent data availability (last 12 months)
    recent_cutoff = datetime.now() - timedelta(days=365)
    recent_records = df.filter(F.col("rec_eff_strt_dt") >= F.lit(recent_cutoff.date())).count()
    total_records = df.count()
    recent_percentage = (recent_records / total_records * 100) if total_records > 0 else 0
    
    # Episode duration analysis
    episode_durations = (df.filter(F.col("rec_eff_end_dt").isNotNull())
                        .withColumn("duration_days", F.datediff("rec_eff_end_dt", "rec_eff_strt_dt"))
                        .select(
                            F.min("duration_days").alias("min_duration"),
                            F.max("duration_days").alias("max_duration"),
                            F.mean("duration_days").alias("avg_duration"),
                            F.expr("percentile_approx(duration_days, 0.5)").alias("median_duration"),
                            F.expr("percentile_approx(duration_days, 0.9)").alias("p90_duration")
                        )
                        .collect()[0])
    
    return {
        'date_range': {
            'earliest_start': date_range['earliest_start'],
            'latest_start': date_range['latest_start'],
            'earliest_end': date_range['earliest_end'],
            'latest_end': date_range['latest_end']
        },
        'yearly_distribution': yearly_distribution,
        'recent_data_percentage': recent_percentage,
        'episode_durations': {
            'min_days': episode_durations['min_duration'],
            'max_days': episode_durations['max_duration'],
            'avg_days': episode_durations['avg_duration'],
            'median_days': episode_durations['median_duration'],
            'p90_days': episode_durations['p90_duration']
        }
    }


def get_event_analysis(df: DataFrame) -> Dict[str, any]:
    """Analyze termination events for survival modeling."""
    
    total_records = df.count()
    
    # Event type distribution
    event_distribution = (df.groupBy("term_type_cd")
                         .count()
                         .withColumn("percentage", F.col("count") / total_records * 100)
                         .collect())
    
    # Active vs terminated records
    active_records = df.filter(F.col("rec_eff_end_dt").isNull()).count()
    terminated_records = df.filter(F.col("term_type_cd").isNotNull()).count()
    
    # Time to event analysis for terminated records
    time_to_event = (df.filter(F.col("term_type_cd").isNotNull())
                    .withColumn("tenure_days", F.datediff("rec_eff_end_dt", "rec_eff_strt_dt"))
                    .select(
                        F.min("tenure_days").alias("min_tenure"),
                        F.max("tenure_days").alias("max_tenure"),
                        F.mean("tenure_days").alias("avg_tenure"),
                        F.expr("percentile_approx(tenure_days, 0.25)").alias("p25_tenure"),
                        F.expr("percentile_approx(tenure_days, 0.5)").alias("median_tenure"),
                        F.expr("percentile_approx(tenure_days, 0.75)").alias("p75_tenure")
                    )
                    .collect()[0])
    
    # Event rates by year
    event_rates_by_year = (df.withColumn("start_year", F.year("rec_eff_strt_dt"))
                          .groupBy("start_year")
                          .agg(
                              F.count("*").alias("total_records"),
                              F.sum(F.when(F.col("term_type_cd").isNotNull(), 1).otherwise(0)).alias("events"),
                              F.sum(F.when(F.col("term_type_cd") == "Vol", 1).otherwise(0)).alias("voluntary_events")
                          )
                          .withColumn("event_rate", F.col("events") / F.col("total_records") * 100)
                          .withColumn("voluntary_rate", F.col("voluntary_events") / F.col("total_records") * 100)
                          .orderBy("start_year")
                          .collect())
    
    return {
        'total_records': total_records,
        'active_records': active_records,
        'terminated_records': terminated_records,
        'event_rate_overall': (terminated_records / total_records * 100) if total_records > 0 else 0,
        'event_distribution': event_distribution,
        'time_to_event_stats': time_to_event,
        'event_rates_by_year': event_rates_by_year
    }


def get_client_distribution(df: DataFrame) -> Dict[str, any]:
    """Analyze client and industry distribution."""
    
    # Top clients by volume
    top_clients = (df.groupBy("clnt_obj_id")
                  .agg(
                      F.countDistinct("pers_obj_id").alias("unique_employees"),
                      F.count("*").alias("total_records")
                  )
                  .orderBy(F.col("unique_employees").desc())
                  .limit(20)
                  .collect())
    
    # NAICS industry distribution
    naics_distribution = (df.filter(F.col("naics_cd").isNotNull())
                         .groupBy("naics_cd")
                         .agg(
                             F.countDistinct("clnt_obj_id").alias("client_count"),
                             F.countDistinct("pers_obj_id").alias("employee_count"),
                             F.count("*").alias("record_count")
                         )
                         .orderBy(F.col("employee_count").desc())
                         .limit(15)
                         .collect())
    
    # Data source distribution
    source_distribution = (df.groupBy("db_schema")
                          .agg(
                              F.countDistinct("clnt_obj_id").alias("client_count"),
                              F.countDistinct("pers_obj_id").alias("employee_count"),
                              F.count("*").alias("record_count")
                          )
                          .collect())
    
    return {
        'top_clients': top_clients,
        'naics_distribution': naics_distribution,
        'source_distribution': source_distribution,
        'total_unique_naics': df.select("naics_cd").distinct().count(),
        'records_with_naics': df.filter(F.col("naics_cd").isNotNull()).count()
    }


def get_feature_quality(df: DataFrame) -> Dict[str, any]:
    """Analyze feature availability and quality for modeling."""
    
    total_records = df.count()
    
    # Key modeling features
    modeling_features = [
        "annl_cmpn_amt", "job_cd", "mgr_sup_id", "pay_rate_type_cd",
        "ftm_ptm_cd", "reg_tmp_cd", "gender_cd", "birth_dt",
        "eeo1_job_cat_cd", "mngr_lvl"
    ]
    
    feature_stats = {}
    for feature in modeling_features:
        if feature in df.columns:
            null_count = df.filter(F.col(feature).isNull()).count()
            unknown_count = df.filter(F.col(feature) == "UNKNOWN").count()
            
            feature_stats[feature] = {
                'null_count': null_count,
                'null_percentage': (null_count / total_records * 100),
                'unknown_count': unknown_count,
                'unknown_percentage': (unknown_count / total_records * 100),
                'available_records': total_records - null_count - unknown_count,
                'availability_percentage': ((total_records - null_count - unknown_count) / total_records * 100)
            }
            
            # Value distribution for categorical features
            if feature in ["pay_rate_type_cd", "ftm_ptm_cd", "reg_tmp_cd", "gender_cd", "eeo1_job_cat_cd"]:
                value_dist = (df.filter(F.col(feature).isNotNull() & (F.col(feature) != "UNKNOWN"))
                             .groupBy(feature)
                             .count()
                             .orderBy(F.col("count").desc())
                             .limit(10)
                             .collect())
                feature_stats[feature]['value_distribution'] = value_dist
    
    # Compensation analysis
    comp_stats = (df.filter(F.col("annl_cmpn_amt").isNotNull() & (F.col("annl_cmpn_amt") > 0))
                 .select(
                     F.min("annl_cmpn_amt").alias("min_comp"),
                     F.max("annl_cmpn_amt").alias("max_comp"),
                     F.mean("annl_cmpn_amt").alias("avg_comp"),
                     F.expr("percentile_approx(annl_cmpn_amt, 0.25)").alias("p25_comp"),
                     F.expr("percentile_approx(annl_cmpn_amt, 0.5)").alias("median_comp"),
                     F.expr("percentile_approx(annl_cmpn_amt, 0.75)").alias("p75_comp"),
                     F.expr("percentile_approx(annl_cmpn_amt, 0.9)").alias("p90_comp")
                 )
                 .collect()[0])
    
    return {
        'feature_availability': feature_stats,
        'compensation_distribution': comp_stats
    }


def get_completeness_patterns(df: DataFrame) -> Dict[str, any]:
    """Analyze data completeness patterns across dimensions."""
    
    # Completeness by data source
    completeness_by_source = (df.groupBy("db_schema")
                             .agg(
                                 F.count("*").alias("total_records"),
                                 F.sum(F.when(F.col("job_cd") != "UNKNOWN", 1).otherwise(0)).alias("has_job"),
                                 F.sum(F.when(F.col("mgr_sup_id").isNotNull(), 1).otherwise(0)).alias("has_manager"),
                                 F.sum(F.when(F.col("annl_cmpn_amt").isNotNull(), 1).otherwise(0)).alias("has_compensation")
                             )
                             .collect())
    
    # Completeness by year
    completeness_by_year = (df.withColumn("start_year", F.year("rec_eff_strt_dt"))
                           .groupBy("start_year")
                           .agg(
                               F.count("*").alias("total_records"),
                               F.sum(F.when(F.col("job_cd") != "UNKNOWN", 1).otherwise(0)).alias("has_job"),
                               F.sum(F.when(F.col("mgr_sup_id").isNotNull(), 1).otherwise(0)).alias("has_manager"),
                               F.sum(F.when(F.col("annl_cmpn_amt").isNotNull(), 1).otherwise(0)).alias("has_compensation")
                           )
                           .orderBy("start_year")
                           .collect())
    
    # Manager hierarchy coverage
    manager_coverage = (df.filter(F.col("mgr_sup_id").isNotNull())
                       .groupBy("mgr_sup_id")
                       .count()
                       .select(
                           F.count("*").alias("total_managers"),
                           F.min("count").alias("min_reports"),
                           F.max("count").alias("max_reports"),
                           F.mean("count").alias("avg_reports"),
                           F.expr("percentile_approx(count, 0.9)").alias("p90_reports")
                       )
                       .collect()[0])
    
    return {
        'completeness_by_source': completeness_by_source,
        'completeness_by_year': completeness_by_year,
        'manager_hierarchy': manager_coverage
    }


def get_survival_characteristics(df: DataFrame) -> Dict[str, any]:
    """Analyze characteristics specific to survival analysis setup."""
    
    # Censoring patterns
    censoring_analysis = (df.groupBy("term_type_cd")
                         .count()
                         .collect())
    
    # Follow-up time analysis
    followup_stats = (df.withColumn(
                         "followup_time",
                         F.datediff(
                             F.coalesce(F.col("rec_eff_end_dt"), F.current_date()),
                             F.col("rec_eff_strt_dt")
                         )
                     )
                     .select(
                         F.min("followup_time").alias("min_followup"),
                         F.max("followup_time").alias("max_followup"),
                         F.mean("followup_time").alias("avg_followup"),
                         F.expr("percentile_approx(followup_time, 0.25)").alias("p25_followup"),
                         F.expr("percentile_approx(followup_time, 0.5)").alias("median_followup"),
                         F.expr("percentile_approx(followup_time, 0.75)").alias("p75_followup")
                     )
                     .collect()[0])
    
    # Potential observation windows for validation
    observation_windows = [180, 365, 730]  # 6 months, 1 year, 2 years
    window_analysis = {}
    
    for window in observation_windows:
        cutoff_date = datetime.now() - timedelta(days=window)
        eligible_records = df.filter(F.col("rec_eff_strt_dt") <= F.lit(cutoff_date.date())).count()
        observed_events = df.filter(
            (F.col("rec_eff_strt_dt") <= F.lit(cutoff_date.date())) &
            (F.col("term_type_cd").isNotNull()) &
            (F.col("rec_eff_end_dt") <= F.lit(cutoff_date.date()))
        ).count()
        
        window_analysis[f"{window}_days"] = {
            'eligible_records': eligible_records,
            'observed_events': observed_events,
            'event_rate': (observed_events / eligible_records * 100) if eligible_records > 0 else 0
        }
    
    return {
        'censoring_distribution': censoring_analysis,
        'followup_time_stats': followup_stats,
        'observation_windows': window_analysis
    }


def print_data_insights(insights: Dict):
    """Print comprehensive data insights for modeling decisions."""
    
    print("ETL DATA INSIGHTS FOR SURVIVAL ANALYSIS")
    print("=" * 60)
    
    # Data Volume
    vol = insights['data_volume']
    print(f"\nDATA VOLUME SUMMARY")
    print(f"Total Records: {vol['total_records']:,}")
    print(f"Unique Persons: {vol['unique_persons']:,}")
    print(f"Unique Clients: {vol['unique_clients']:,}")
    print(f"Avg Records per Person: {vol['avg_records_per_person']:.1f}")
    print(f"Largest Client: {vol['largest_client_size']:,} employees")
    
    # Temporal Patterns
    temp = insights['temporal_patterns']
    print(f"\nTEMPORAL COVERAGE")
    print(f"Date Range: {temp['date_range']['earliest_start']} to {temp['date_range']['latest_start']}")
    print(f"Recent Data (last 12 months): {temp['recent_data_percentage']:.1f}%")
    print(f"Median Episode Duration: {temp['episode_durations']['median_days']:.0f} days")
    
    # Event Analysis
    events = insights['event_analysis']
    print(f"\nEVENT ANALYSIS (SURVIVAL MODELING)")
    print(f"Overall Event Rate: {events['event_rate_overall']:.1f}%")
    print(f"Active Records: {events['active_records']:,}")
    print(f"Terminated Records: {events['terminated_records']:,}")
    print(f"Median Tenure at Termination: {events['time_to_event_stats']['median_tenure']:.0f} days")
    
    print("\nEvent Type Distribution:")
    for event in events['event_distribution']:
        event_type = event['term_type_cd'] if event['term_type_cd'] else 'Active/Censored'
        print(f"  {event_type}: {event['count']:,} ({event['percentage']:.1f}%)")
    
    # Industry Distribution
    clients = insights['client_distribution']
    print(f"\nINDUSTRY COVERAGE")
    print(f"Total NAICS Industries: {clients['total_unique_naics']}")
    print(f"Records with NAICS: {clients['records_with_naics']:,}")
    
    print("\nTop Industries by Employee Count:")
    for naics in clients['naics_distribution'][:5]:
        print(f"  NAICS {naics['naics_cd']}: {naics['employee_count']:,} employees")
    
    # Feature Quality
    features = insights['feature_quality']
    print(f"\nFEATURE AVAILABILITY FOR MODELING")
    
    key_features = ['annl_cmpn_amt', 'job_cd', 'mgr_sup_id', 'gender_cd']
    for feature in key_features:
        if feature in features['feature_availability']:
            stats = features['feature_availability'][feature]
            print(f"  {feature}: {stats['availability_percentage']:.1f}% available")
    
    # Compensation insights
    comp = features['compensation_distribution']
    print(f"\nCOMPENSATION DISTRIBUTION")
    print(f"  Median: ${comp['median_comp']:,.0f}")
    print(f"  P25-P75: ${comp['p25_comp']:,.0f} - ${comp['p75_comp']:,.0f}")
    print(f"  Range: ${comp['min_comp']:,.0f} - ${comp['max_comp']:,.0f}")
    
    # Survival Analysis Insights
    survival = insights['survival_characteristics']
    print(f"\nSURVIVAL ANALYSIS SETUP INSIGHTS")
    print(f"Median Follow-up Time: {survival['followup_time_stats']['median_followup']:.0f} days")
    
    print("\nValidation Window Analysis:")
    for window, stats in survival['observation_windows'].items():
        days = window.replace('_days', '')
        print(f"  {days} days: {stats['eligible_records']:,} records, {stats['event_rate']:.1f}% event rate")
    
    # Recommendations
    print(f"\n" + "=" * 60)
    print(f"MODELING RECOMMENDATIONS")
    print(f"=" * 60)
    
    # Training split recommendation
    if temp['recent_data_percentage'] > 20:
        print("[OK] Sufficient recent data for temporal validation split")
    else:
        print("[Caution] Limited recent data - consider cross-validation approach")
    
    # Event rate assessment
    if 15 <= events['event_rate_overall'] <= 30:
        print("[OK] Good event rate for survival analysis")
    elif events['event_rate_overall'] < 10:
        print("[Caution] Low event rate - may need longer observation periods")
    else:
        print("[Caution] High event rate - validate data quality")
    
    # Feature readiness
    usable_features = sum(1 for f in features['feature_availability'].values() 
                         if f['availability_percentage'] > 70)
    print(f"[OK] {usable_features} features with >70% availability")
    
    # Industry stratification
    if clients['total_unique_naics'] > 50:
        print("[OK] Sufficient industry diversity for NAICS stratification")
    else:
        print("[Caution] Limited industry diversity - consider broader groupings")


# Usage
if __name__ == "__main__":
    insights = analyze_etl_output(df)
    print_data_insights(insights)
