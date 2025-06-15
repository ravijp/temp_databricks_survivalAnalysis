# Databricks notebook source
# MAGIC %md
# MAGIC # ADP Employee Turnover - Databricks Starter Template
# MAGIC ### For team members new to Databricks/PySpark
# MAGIC 
# MAGIC **Key Databricks Concepts for Our Project:**
# MAGIC - Databricks notebooks run on Spark clusters (distributed computing)
# MAGIC - Use `spark.sql()` for SQL queries or DataFrame API for Python
# MAGIC - Data is typically stored in Delta Lake format
# MAGIC - Use `display()` instead of `print()` for better table visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup & Imports

# COMMAND ----------

# Standard imports for survival analysis project
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Survival analysis specific imports
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Initialize Spark session (usually auto-initialized in Databricks)
spark = SparkSession.builder.getOrCreate()

print("✅ Environment setup complete!")
print(f"Spark version: {spark.version}")
print(f"Available cores: {spark.sparkContext.defaultParallelism}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Connection Templates

# COMMAND ----------

# Template for connecting to Analytics Warehouse
def connect_to_analytics_warehouse():
    """
    Template function to connect to ADP Analytics Warehouse
    Replace with actual connection details once catalog access is confirmed
    """
    try:
        # Test connection
        test_query = """
        SELECT COUNT(*) as row_count 
        FROM your_catalog.your_schema.employee_data 
        LIMIT 5
        """
        
        result = spark.sql(test_query)
        display(result)
        print("✅ Analytics Warehouse connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("Check catalog permissions and table names")
        return False

# Test the connection
connection_status = connect_to_analytics_warehouse()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Spark DataFrame Basics for Survival Analysis

# COMMAND ----------

# Essential Spark DataFrame operations for our project
def spark_dataframe_survival_basics():
    """
    Key Spark operations you'll need for survival analysis data prep
    """
    
    # Example: Creating sample employee data (replace with real data)
    sample_data = [
        (1, "2020-01-01", "2021-06-15", 1, 75000, "Engineering", "Manufacturing"),
        (2, "2020-02-01", "2023-12-31", 0, 68000, "Sales", "Technology"), 
        (3, "2020-03-01", "2021-12-01", 1, 82000, "Marketing", "Healthcare")
    ]
    
    schema = StructType([
        StructField("employee_id", IntegerType(), True),
        StructField("hire_date", StringType(), True),
        StructField("termination_date", StringType(), True),
        StructField("terminated", IntegerType(), True),  # 1=terminated, 0=censored
        StructField("salary", IntegerType(), True),
        StructField("department", StringType(), True),
        StructField("industry", StringType(), True)
    ])
    
    df = spark.createDataFrame(sample_data, schema)
    
    # Convert string dates to proper date types
    df = df.withColumn("hire_date", to_date(col("hire_date"), "yyyy-MM-dd")) \
          .withColumn("termination_date", to_date(col("termination_date"), "yyyy-MM-dd"))
    
    # Calculate tenure (survival time) in days
    df = df.withColumn("tenure_days", 
                      when(col("terminated") == 1, 
                           datediff(col("termination_date"), col("hire_date")))
                      .otherwise(datediff(current_date(), col("hire_date"))))
    
    print("Sample employee survival data:")
    display(df)
    
    return df

# Create sample dataset
sample_df = spark_dataframe_survival_basics()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Start-Stop Format Conversion Template

# COMMAND ----------

def convert_to_start_stop_format(df):
    """
    Convert employee data to start-stop format for time-varying covariates
    This is crucial for Cox PH models with time-varying covariates
    """
    
    # This is a simplified example - real implementation will be more complex
    # based on actual Analytics Warehouse schema
    
    print("🔄 Converting to start-stop format...")
    print("Key concepts:")
    print("- Each row represents a time period where covariates are constant")
    print("- Multiple rows per employee for different time periods")
    print("- Event indicator only '1' in final row if termination occurred")
    
    # Example transformation (customize based on real data structure)
    start_stop_df = df.select(
        col("employee_id"),
        lit(0).alias("start_time"),  # Start of observation period
        col("tenure_days").alias("stop_time"),  # End of observation/event
        col("terminated").alias("event"),  # Event indicator
        col("salary"),
        col("department"),
        col("industry")
    )
    
    print("Start-stop format sample:")
    display(start_stop_df)
    
    return start_stop_df

# Convert sample data
start_stop_sample = convert_to_start_stop_format(sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Basic Survival Analysis Pipeline

# COMMAND ----------

def run_basic_survival_analysis(df):
    """
    Basic survival analysis pipeline using lifelines
    This will serve as your starting point for more complex models
    """
    
    # Convert Spark DataFrame to Pandas for lifelines
    pandas_df = df.toPandas()
    
    # Kaplan-Meier survival curve
    kmf = KaplanMeierFitter()
    kmf.fit(durations=pandas_df['tenure_days'], 
            event_observed=pandas_df['terminated'],
            label='Overall Survival')
    
    # Plot survival curve
    plt.figure(figsize=(10, 6))
    kmf.plot()
    plt.title('Employee Survival Curve - Kaplan-Meier Estimate')
    plt.xlabel('Days Since Hire')
    plt.ylabel('Survival Probability')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Basic statistics
    print(f"Median survival time: {kmf.median_survival_time_:.0f} days")
    print(f"12-month survival probability: {kmf.survival_function_at_times(365).iloc[0]:.3f}")
    
    return kmf

# Run basic analysis
survival_model = run_basic_survival_analysis(sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Data Quality Assessment Template

# COMMAND ----------

def assess_data_quality(df):
    """
    Comprehensive data quality assessment specific to survival analysis
    Focus on issues identified in ADP project context
    """
    
    print("🔍 Data Quality Assessment for Survival Analysis")
    print("=" * 50)
    
    # Basic data info
    print(f"Total records: {df.count():,}")
    print(f"Total employees: {df.select('employee_id').distinct().count():,}")
    
    # Missing value analysis
    print("\n📊 Missing Value Analysis:")
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        null_pct = (null_count / df.count()) * 100
        print(f"  {column}: {null_count:,} ({null_pct:.1f}%)")
    
    # Survival-specific quality checks
    print("\n⚠️  Survival Analysis Specific Checks:")
    
    # Check for negative tenure
    negative_tenure = df.filter(col("tenure_days") < 0).count()
    print(f"  Negative tenure records: {negative_tenure}")
    
    # Check for zero tenure (same-day hires/terms)
    zero_tenure = df.filter(col("tenure_days") == 0).count()
    print(f"  Zero tenure records: {zero_tenure}")
    
    # Event rate analysis
    event_rate = df.filter(col("terminated") == 1).count() / df.count()
    print(f"  Overall termination rate: {event_rate:.3f}")
    
    # Industry distribution (for NAICS stratification)
    print("\n🏭 Industry Distribution:")
    industry_dist = df.groupBy("industry").count().orderBy(desc("count"))
    display(industry_dist)
    
    return {
        'total_records': df.count(),
        'total_employees': df.select('employee_id').distinct().count(),
        'event_rate': event_rate,
        'negative_tenure': negative_tenure,
        'zero_tenure': zero_tenure
    }

# Run data quality assessment
quality_metrics = assess_data_quality(sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Performance Tracking Template

# COMMAND ----------

def track_model_performance(y_true, y_pred, model_name):
    """
    Template for tracking survival model performance
    Focus on metrics relevant to ADP business requirements
    """
    
    from lifelines.utils import concordance_index
    from sklearn.metrics import brier_score_loss
    
    print(f"📈 Performance Metrics for {model_name}")
    print("=" * 40)
    
    # Concordance Index (C-index) - primary metric for survival models
    c_index = concordance_index(y_true['duration'], y_pred, y_true['event'])
    print(f"C-index: {c_index:.3f}")
    
    # Business interpretation
    if c_index > 0.6:
        print("✅ Acceptable predictive performance")
    else:
        print("⚠️  Below minimum threshold (0.6)")
    
    # Log metrics for tracking across experiments
    metrics = {
        'model_name': model_name,
        'c_index': c_index,
        'timestamp': pd.Timestamp.now(),
        'business_acceptable': c_index > 0.6
    }
    
    return metrics

# Example usage (with dummy data)
dummy_metrics = {
    'model_name': 'Baseline Cox PH',
    'c_index': 0.65,
    'timestamp': pd.Timestamp.now(),
    'business_acceptable': True
}

print("Example model tracking:")
print(dummy_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Quick Reference: Common Operations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Databricks Quick Reference for Survival Analysis Team
# MAGIC 
# MAGIC **Data Loading:**
# MAGIC ```python
# MAGIC # From catalog
# MAGIC df = spark.table("catalog.schema.table_name")
# MAGIC 
# MAGIC # From SQL query
# MAGIC df = spark.sql("SELECT * FROM catalog.schema.table_name WHERE condition")
# MAGIC ```
# MAGIC 
# MAGIC **Common Transformations:**
# MAGIC ```python
# MAGIC # Add calculated columns
# MAGIC df = df.withColumn("new_column", col("existing_column") * 2)
# MAGIC 
# MAGIC # Filter data
# MAGIC df = df.filter(col("column_name") > value)
# MAGIC 
# MAGIC # Group and aggregate
# MAGIC df = df.groupBy("group_column").agg(avg("value_column").alias("avg_value"))
# MAGIC ```
# MAGIC 
# MAGIC **Survival Analysis Specific:**
# MAGIC ```python
# MAGIC # Calculate time differences
# MAGIC df = df.withColumn("tenure_days", datediff(col("end_date"), col("start_date")))
# MAGIC 
# MAGIC # Create event indicators
# MAGIC df = df.withColumn("event", when(col("status") == "terminated", 1).otherwise(0))
# MAGIC ```
# MAGIC 
# MAGIC **Performance Tips:**
# MAGIC - Use `display()` instead of `show()` for better visualization
# MAGIC - Cache frequently used DataFrames: `df.cache()`
# MAGIC - Use `.toPandas()` only for final analysis, not large datasets
# MAGIC - Partition data appropriately for time-series analysis

# COMMAND ----------

print("🎉 Databricks Survival Analysis Starter Template Complete!")
print("\nNext Steps:")
print("1. Test connection to Analytics Warehouse")
print("2. Explore actual employee data schema") 
print("3. Adapt templates to real data structure")
print("4. Begin Week 1 data discovery tasks")
print("\n💡 Remember: Ask for help if you get stuck - we're in sprint mode!")
