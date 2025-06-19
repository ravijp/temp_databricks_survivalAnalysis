from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def comprehensive_table_eda(spark, table_name, sample_size=100000, show_distributions=True, max_categories=50):
    """
    Comprehensive EDA function optimized for Databricks/PySpark
    
    Parameters:
    -----------
    spark : SparkSession
        Active Spark session
    table_name : str
        Name of the table to analyze (can include database.table_name)
    sample_size : int
        Sample size for detailed analysis (default 100k for speed)
    show_distributions : bool
        Whether to show categorical distributions
    max_categories : int
        Maximum number of categories to show for categorical variables
    
    Returns:
    --------
    dict : Dictionary containing all EDA results
    """
    
    print("="*80)
    print(f"COMPREHENSIVE EDA REPORT: {table_name}")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Read the table
    df = spark.table(table_name)
    
    # 1. BASIC TABLE INFORMATION
    print("📊 BASIC TABLE INFORMATION")
    print("-" * 40)
    
    total_rows = df.count()
    total_cols = len(df.columns)
    
    print(f"Total Rows: {total_rows:,}")
    print(f"Total Columns: {total_cols}")
    print(f"Table Size: {total_rows:,} x {total_cols}")
    print()
    
    # 2. SCHEMA INFORMATION
    print("🏗️  SCHEMA INFORMATION")
    print("-" * 40)
    
    schema_info = []
    for field in df.schema.fields:
        schema_info.append({
            'Column': field.name,
            'Data_Type': str(field.dataType),
            'Nullable': field.nullable
        })
    
    schema_df = pd.DataFrame(schema_info)
    print(schema_df.to_string(index=False))
    print()
    
    # Type categorization
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, LongType, FloatType, DoubleType, DecimalType))]
    string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    date_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (DateType, TimestampType))]
    boolean_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, BooleanType)]
    
    print(f"📈 Numeric Columns ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
    print(f"📝 String Columns ({len(string_cols)}): {string_cols[:10]}{'...' if len(string_cols) > 10 else ''}")
    print(f"📅 Date Columns ({len(date_cols)}): {date_cols}")
    print(f"☑️  Boolean Columns ({len(boolean_cols)}): {boolean_cols}")
    print()
    
    # 3. SAMPLE DATA
    print("🔍 SAMPLE DATA (First 5 rows)")
    print("-" * 40)
    sample_data = df.limit(5).toPandas()
    print(sample_data.to_string(index=False, max_cols=10))
    print()
    
    # 4. NULL VALUE ANALYSIS
    print("❌ NULL VALUE ANALYSIS")
    print("-" * 40)
    
    # Calculate null counts efficiently
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
    
    null_analysis = []
    for col_name in df.columns:
        null_count = null_counts[col_name]
        null_percentage = (null_count / total_rows) * 100
        null_analysis.append({
            'Column': col_name,
            'Null_Count': null_count,
            'Null_Percentage': round(null_percentage, 2),
            'Data_Type': str(dict(df.dtypes)[col_name])
        })
    
    null_df = pd.DataFrame(null_analysis)
    null_df = null_df.sort_values('Null_Percentage', ascending=False)
    
    # Show columns with nulls
    columns_with_nulls = null_df[null_df['Null_Percentage'] > 0]
    if len(columns_with_nulls) > 0:
        print("Columns with NULL values:")
        print(columns_with_nulls.to_string(index=False))
    else:
        print("✅ No NULL values found in any column!")
    print()
    
    # 5. NUMERIC COLUMN ANALYSIS
    if numeric_cols:
        print("📊 NUMERIC COLUMNS ANALYSIS")
        print("-" * 40)
        
        # Get basic statistics for all numeric columns
        numeric_stats = df.select(numeric_cols).describe().toPandas()
        numeric_stats = numeric_stats.set_index('summary').T
        
        # Add additional statistics
        for col_name in numeric_cols:
            try:
                # Get percentiles (approximate for speed)
                percentiles = df.select(col_name).na.drop().approxQuantile(col_name, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], 0.01)
                
                # Get unique count (approximate)
                unique_count = df.select(col_name).distinct().count()
                
                print(f"\n📈 {col_name}:")
                print(f"   Count: {int(float(numeric_stats.loc[col_name, 'count'])):,}")
                print(f"   Mean: {float(numeric_stats.loc[col_name, 'mean']):.4f}")
                print(f"   Std: {float(numeric_stats.loc[col_name, 'stddev']):.4f}")
                print(f"   Min: {float(numeric_stats.loc[col_name, 'min']):.4f}")
                print(f"   Max: {float(numeric_stats.loc[col_name, 'max']):.4f}")
                print(f"   Unique Values: {unique_count:,}")
                print(f"   Percentiles:")
                print(f"     1%: {percentiles[0]:.4f}, 5%: {percentiles[1]:.4f}")
                print(f"     25%: {percentiles[2]:.4f}, 50%: {percentiles[3]:.4f}, 75%: {percentiles[4]:.4f}")
                print(f"     95%: {percentiles[5]:.4f}, 99%: {percentiles[6]:.4f}")
                
                # Check for potential issues
                if percentiles[0] == percentiles[6]:  # All percentiles same
                    print(f"   ⚠️  Warning: Very low variance detected")
                if unique_count < 10:
                    print(f"   💡 Note: Low cardinality - might be categorical")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {col_name}: {str(e)}")
        
        print()
    
    # 6. CATEGORICAL COLUMN ANALYSIS
    if string_cols and show_distributions:
        print("📝 CATEGORICAL COLUMNS ANALYSIS")
        print("-" * 40)
        
        for col_name in string_cols[:10]:  # Limit to first 10 to avoid overwhelming output
            try:
                print(f"\n📊 {col_name}:")
                
                # Get unique count
                unique_count = df.select(col_name).na.drop().distinct().count()
                non_null_count = df.select(col_name).na.drop().count()
                
                print(f"   Non-null Count: {non_null_count:,}")
                print(f"   Unique Values: {unique_count:,}")
                
                if unique_count <= max_categories and unique_count > 0:
                    # Get value distribution
                    value_counts = (df.select(col_name)
                                   .na.drop()
                                   .groupBy(col_name)
                                   .count()
                                   .orderBy(desc("count"))
                                   .limit(20)
                                   .collect())
                    
                    print(f"   Top Values:")
                    for row in value_counts[:10]:
                        percentage = (row['count'] / non_null_count) * 100
                        print(f"     '{row[col_name]}': {row['count']:,} ({percentage:.2f}%)")
                    
                    # Check for imbalance
                    if len(value_counts) > 1:
                        top_percentage = (value_counts[0]['count'] / non_null_count) * 100
                        if top_percentage > 80:
                            print(f"   ⚠️  High imbalance: Top category represents {top_percentage:.1f}% of data")
                        elif top_percentage < 5:
                            print(f"   📊 Well balanced distribution")
                            
                else:
                    print(f"   💡 High cardinality column - showing sample values only")
                    sample_values = (df.select(col_name)
                                   .na.drop()
                                   .distinct()
                                   .limit(10)
                                   .collect())
                    sample_list = [row[col_name] for row in sample_values]
                    print(f"   Sample Values: {sample_list}")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {col_name}: {str(e)}")
        
        if len(string_cols) > 10:
            print(f"\n💡 Showing first 10 categorical columns. Total categorical columns: {len(string_cols)}")
        
        print()
    
    # 7. DATE COLUMN ANALYSIS
    if date_cols:
        print("📅 DATE COLUMNS ANALYSIS")
        print("-" * 40)
        
        for col_name in date_cols:
            try:
                print(f"\n📅 {col_name}:")
                
                # Get date range and basic stats
                date_stats = df.select(
                    min(col_name).alias('min_date'),
                    max(col_name).alias('max_date'),
                    count(col_name).alias('non_null_count')
                ).collect()[0]
                
                print(f"   Non-null Count: {date_stats['non_null_count']:,}")
                print(f"   Date Range: {date_stats['min_date']} to {date_stats['max_date']}")
                
                # Calculate date span
                if date_stats['min_date'] and date_stats['max_date']:
                    date_span = (date_stats['max_date'] - date_stats['min_date']).days
                    print(f"   Span: {date_span:,} days ({date_span/365.25:.1f} years)")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {col_name}: {str(e)}")
        
        print()
    
    # 8. DATA QUALITY SUMMARY
    print("🔍 DATA QUALITY SUMMARY")
    print("-" * 40)
    
    high_null_cols = [col for col in df.columns if null_counts[col] / total_rows > 0.5]
    low_variance_cols = []
    high_cardinality_cols = [col for col in string_cols if df.select(col).distinct().count() > total_rows * 0.8]
    
    print(f"✅ Complete Analysis of {total_rows:,} rows and {total_cols} columns")
    print(f"📊 Numeric Columns: {len(numeric_cols)}")
    print(f"📝 Categorical Columns: {len(string_cols)}")
    print(f"📅 Date Columns: {len(date_cols)}")
    print(f"☑️  Boolean Columns: {len(boolean_cols)}")
    
    if high_null_cols:
        print(f"⚠️  High NULL columns (>50%): {high_null_cols}")
    if high_cardinality_cols:
        print(f"🔢 High cardinality columns (>80% unique): {high_cardinality_cols}")
    
    print()
    print("="*80)
    print(f"EDA completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Return summary dictionary for programmatic use
    return {
        'table_name': table_name,
        'total_rows': total_rows,
        'total_cols': total_cols,
        'numeric_cols': numeric_cols,
        'string_cols': string_cols,
        'date_cols': date_cols,
        'boolean_cols': boolean_cols,
        'null_analysis': null_df.to_dict('records'),
        'high_null_cols': high_null_cols,
        'high_cardinality_cols': high_cardinality_cols
    }

# USAGE EXAMPLE FOR employee_main_monthly TABLE
# ================================================

def analyze_employee_main_monthly(spark):
    """
    Specific analysis function for employee_main_monthly table
    """
    print("🎯 ANALYZING employee_main_monthly TABLE FOR ADP TURNOVER PROJECT")
    print("="*80)
    
    # Run comprehensive EDA
    eda_results = comprehensive_table_eda(
        spark=spark,
        table_name="employee_main_monthly",
        sample_size=100000,
        show_distributions=True,
        max_categories=20
    )
    
    # Additional survival analysis specific checks
    df = spark.table("employee_main_monthly")
    
    print("\n🎯 SURVIVAL ANALYSIS SPECIFIC CHECKS")
    print("-" * 40)
    
    # Check for employee ID column
    potential_id_cols = [col for col in df.columns if 'id' in col.lower() or 'employee' in col.lower()]
    print(f"Potential Employee ID columns: {potential_id_cols}")
    
    # Check for date columns (crucial for survival analysis)
    date_related_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'month', 'year'])]
    print(f"Date-related columns: {date_related_cols}")
    
    # Check for termination/status columns
    status_cols = [col for col in df.columns if any(term in col.lower() for term in ['status', 'term', 'active', 'left', 'quit'])]
    print(f"Potential Status/Termination columns: {status_cols}")
    
    # Check for demographic/feature columns
    demo_cols = [col for col in df.columns if any(term in col.lower() for term in ['age', 'gender', 'dept', 'manager', 'salary', 'title', 'location'])]
    print(f"Demographic/Feature columns: {demo_cols}")
    
    return eda_results

# QUICK EXECUTION FUNCTIONS
# =========================

def quick_table_summary(spark, table_name):
    """Quick 30-second table summary"""
    df = spark.table(table_name)
    print(f"Table: {table_name}")
    print(f"Rows: {df.count():,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Schema: {[(f.name, str(f.dataType)) for f in df.schema.fields[:5]]}...")
    df.show(3, truncate=True)

def get_column_list(spark, table_name):
    """Get formatted column list"""
    df = spark.table(table_name)
    for i, field in enumerate(df.schema.fields, 1):
        print(f"{i:2d}. {field.name:<30} | {str(field.dataType):<20} | Nullable: {field.nullable}")

# Execute the analysis
if __name__ == "__main__":
    # Initialize Spark session (in Databricks, spark is already available)
    # spark = SparkSession.builder.appName("TableEDA").getOrCreate()
    
    # Run the analysis
    results = analyze_employee_main_monthly(spark)
