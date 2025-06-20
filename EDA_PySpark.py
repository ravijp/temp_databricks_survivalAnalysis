from pyspark.sql.functions import col as spark_col, count as spark_count, when, desc, approx_count_distinct
import pandas as pd
import os
import time

def generate_edd_fast(table_name, output_path="/tmp/edd", filter_condition=None, sample_rate=None):
    """
    Fast EDD generation using batch operations
    
    Parameters:
    table_name: Name of the table to analyze
    output_path: Directory to save CSV file
    filter_condition: Optional SQL WHERE condition
    sample_rate: Optional sampling rate between 0 and 1
    """
    
    print(f"Generating EDD for: {table_name}")
    start_time = time.time()
    
    # Load and prepare data
    df = spark.table(table_name)
    
    if filter_condition:
        df = df.filter(filter_condition)
        print(f"Applied filter: {filter_condition}")
    
    if sample_rate:
        df = df.sample(sample_rate, seed=42)
        print(f"Using {sample_rate*100}% sample")
    
    # Cache the dataframe
    df.cache()
    
    total_rows = df.count()
    print(f"Analyzing {total_rows:,} rows")
    
    column_names = df.columns
    print(f"Processing {len(column_names)} columns using batch operations...")
    
    # Step 1: Get column types
    column_types = dict(df.dtypes)
    numeric_columns = [column_name for column_name, dtype in column_types.items() 
                      if dtype in ['int', 'bigint', 'float', 'double', 'decimal']]
    categorical_columns = [column_name for column_name in column_names if column_name not in numeric_columns]
    
    print(f"Numeric columns: {len(numeric_columns)}, Categorical columns: {len(categorical_columns)}")
    
    # Step 2: Batch compute null counts for ALL columns at once
    print("Computing null counts for all columns...")
    null_count_expressions = [spark_count(when(spark_col(column_name).isNull(), column_name)).alias(f"null_{column_name}") 
                             for column_name in column_names]
    null_results_row = df.select(null_count_expressions).collect()[0]
    null_counts_dict = {column_name: null_results_row[f"null_{column_name}"] for column_name in column_names}
    
    # Step 3: Batch compute unique counts for ALL columns at once
    print("Computing unique counts for all columns...")
    unique_count_expressions = [approx_count_distinct(spark_col(column_name)).alias(column_name) 
                               for column_name in column_names]
    unique_results_row = df.select(unique_count_expressions).collect()[0]
    unique_counts_dict = {column_name: unique_results_row[column_name] for column_name in column_names}
    
    # Step 4: Process numeric columns
    numeric_statistics = {}
    if numeric_columns:
        print(f"Computing statistics for {len(numeric_columns)} numeric columns...")
        
        # Get basic stats for all numeric columns in one operation
        stats_df = df.select(numeric_columns).describe()
        stats_pandas_df = stats_df.toPandas().set_index('summary')
        
        # Get percentiles for numeric columns
        print("Computing percentiles for numeric columns...")
        percentile_values = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        
        for column_name in numeric_columns:
            try:
                # Extract basic stats
                basic_stats = {
                    'mean': float(stats_pandas_df.loc['mean', column_name]),
                    'stddev': float(stats_pandas_df.loc['stddev', column_name]),
                    'min': float(stats_pandas_df.loc['min', column_name]),
                    'max': float(stats_pandas_df.loc['max', column_name])
                }
                
                # Get percentiles
                percentiles_list = df.select(column_name).na.drop().approxQuantile(column_name, percentile_values, 0.05)
                
                numeric_statistics[column_name] = {
                    **basic_stats,
                    'percentiles': percentiles_list
                }
            except Exception as error:
                numeric_statistics[column_name] = {'error': str(error)}
    
    # Step 5: Process categorical columns
    categorical_statistics = {}
    if categorical_columns:
        print(f"Computing top values for {len(categorical_columns)} categorical columns...")
        
        for column_name in categorical_columns:
            try:
                # Get top 10 values for categorical columns
                top_values_list = (df.select(column_name)
                                 .filter(spark_col(column_name).isNotNull())
                                 .groupBy(column_name)
                                 .count()
                                 .orderBy(desc("count"))
                                 .limit(10)
                                 .collect())
                
                categorical_statistics[column_name] = [(row[column_name], row['count']) for row in top_values_list]
            except Exception as error:
                categorical_statistics[column_name] = []
    
    # Step 6: Build results in EDD format
    print("Building results...")
    edd_results = []
    
    for field_index, column_name in enumerate(column_names, 1):
        non_null_count = total_rows - null_counts_dict[column_name]
        unique_count = unique_counts_dict[column_name]
        
        if column_name in numeric_columns:
            # Numeric column result - matching original EDD format
            if column_name in numeric_statistics and 'error' not in numeric_statistics[column_name]:
                stats = numeric_statistics[column_name]
                percentiles_list = stats.get('percentiles', [])
                
                edd_result = {
                    'Field_Num': field_index,
                    'Field_Name': column_name,
                    'Type': 'Numeric',
                    'Num_Blanks': null_counts_dict[column_name],
                    'Num_Entries': non_null_count,
                    'Num_Unique': unique_count,
                    'Stddev': round(stats['stddev'], 6),
                    'Mean_or_Top1': round(stats['mean'], 6),
                    'Min_or_Top2': stats['min'],
                    'P1_or_Top3': percentiles_list[0] if len(percentiles_list) > 0 else None,
                    'P5_or_Top4': percentiles_list[1] if len(percentiles_list) > 1 else None,
                    'P25_or_Top5': percentiles_list[2] if len(percentiles_list) > 2 else None,
                    'Median_or_Bot5': percentiles_list[3] if len(percentiles_list) > 3 else None,
                    'P75_or_Bot4': percentiles_list[4] if len(percentiles_list) > 4 else None,
                    'P95_or_Bot3': percentiles_list[5] if len(percentiles_list) > 5 else None,
                    'P99_or_Bot2': percentiles_list[6] if len(percentiles_list) > 6 else None,
                    'Max_or_Bot1': stats['max']
                }
            else:
                # Numeric column with error
                edd_result = {
                    'Field_Num': field_index,
                    'Field_Name': column_name,
                    'Type': 'Numeric',
                    'Num_Blanks': null_counts_dict[column_name],
                    'Num_Entries': non_null_count,
                    'Num_Unique': unique_count,
                    'Stddev': None,
                    'Mean_or_Top1': 'Error',
                    'Min_or_Top2': None,
                    'P1_or_Top3': None,
                    'P5_or_Top4': None,
                    'P25_or_Top5': None,
                    'Median_or_Bot5': None,
                    'P75_or_Bot4': None,
                    'P95_or_Bot3': None,
                    'P99_or_Bot2': None,
                    'Max_or_Bot1': None
                }
        else:
            # Categorical column result - matching original EDD format
            edd_result = {
                'Field_Num': field_index,
                'Field_Name': column_name,
                'Type': 'Categorical',
                'Num_Blanks': null_counts_dict[column_name],
                'Num_Entries': non_null_count,
                'Num_Unique': unique_count,
                'Stddev': None
            }
            
            # Add top values in EDD format
            if column_name in categorical_statistics and len(categorical_statistics[column_name]) > 0:
                top_values_list = categorical_statistics[column_name]
                
                # Create 17 positions for categorical values (like original EDD)
                categorical_values = []
                for value, count in top_values_list:
                    categorical_values.append(f"{value}:{count}")
                
                # Pad to ensure we have enough values
                while len(categorical_values) < 17:
                    categorical_values.append('')
                
                # Map to EDD column names
                edd_result.update({
                    'Mean_or_Top1': categorical_values[0] if len(categorical_values) > 0 else '',
                    'Min_or_Top2': categorical_values[1] if len(categorical_values) > 1 else '',
                    'P1_or_Top3': categorical_values[2] if len(categorical_values) > 2 else '',
                    'P5_or_Top4': categorical_values[3] if len(categorical_values) > 3 else '',
                    'P25_or_Top5': categorical_values[4] if len(categorical_values) > 4 else '',
                    'Median_or_Bot5': categorical_values[12] if len(categorical_values) > 12 else '',
                    'P75_or_Bot4': categorical_values[13] if len(categorical_values) > 13 else '',
                    'P95_or_Bot3': categorical_values[14] if len(categorical_values) > 14 else '',
                    'P99_or_Bot2': categorical_values[15] if len(categorical_values) > 15 else '',
                    'Max_or_Bot1': categorical_values[16] if len(categorical_values) > 16 else ''
                })
            else:
                # No data or high cardinality
                if unique_count == 0:
                    default_value = 'No_Data'
                elif unique_count == 1:
                    default_value = 'All_Same'
                elif unique_count == non_null_count:
                    default_value = 'All_Unique'
                else:
                    default_value = 'High_Cardinality'
                
                edd_result.update({
                    'Mean_or_Top1': default_value,
                    'Min_or_Top2': default_value,
                    'P1_or_Top3': default_value,
                    'P5_or_Top4': default_value,
                    'P25_or_Top5': default_value,
                    'Median_or_Bot5': default_value,
                    'P75_or_Bot4': default_value,
                    'P95_or_Bot3': default_value,
                    'P99_or_Bot2': default_value,
                    'Max_or_Bot1': default_value
                })
        
        edd_results.append(edd_result)
    
    # Unpersist cache
    df.unpersist()
    
    # Save results
    try:
        os.makedirs(output_path, exist_ok=True)
    except:
        output_path = "/tmp/edd"
        os.makedirs(output_path, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{table_name.replace('.', '_')}_edd_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    # Create DataFrame and save
    edd_dataframe = pd.DataFrame(edd_results)
    edd_dataframe.to_csv(filepath, index=False)
    
    runtime_minutes = (time.time() - start_time) / 60
    print(f"EDD completed in {runtime_minutes:.1f} minutes")
    print(f"EDD saved: {filepath}")
    
    return filepath

def generate_edd_ultra_fast(table_name, output_path="/tmp/edd", filter_condition=None, sample_rate=None):
    """
    Ultra-fast EDD for very large tables - basic statistics only
    """
    
    print(f"Ultra-fast EDD for: {table_name}")
    start_time = time.time()
    
    # Load data
    df = spark.table(table_name)
    
    if filter_condition:
        df = df.filter(filter_condition)
        print(f"Applied filter: {filter_condition}")
    
    if sample_rate:
        df = df.sample(sample_rate, seed=42)
        print(f"Using {sample_rate*100}% sample")
    
    df.cache()
    total_rows = df.count()
    print(f"Analyzing {total_rows:,} rows with minimal statistics")
    
    column_names = df.columns
    column_types = dict(df.dtypes)
    numeric_columns = [column_name for column_name, dtype in column_types.items() 
                      if dtype in ['int', 'bigint', 'float', 'double', 'decimal']]
    
    # Batch null counts
    null_count_expressions = [spark_count(when(spark_col(column_name).isNull(), column_name)).alias(f"null_{column_name}") 
                             for column_name in column_names]
    null_results_row = df.select(null_count_expressions).collect()[0]
    null_counts_dict = {column_name: null_results_row[f"null_{column_name}"] for column_name in column_names}
    
    # Batch unique counts
    unique_count_expressions = [approx_count_distinct(spark_col(column_name)).alias(column_name) 
                               for column_name in column_names]
    unique_results_row = df.select(unique_count_expressions).collect()[0]
    unique_counts_dict = {column_name: unique_results_row[column_name] for column_name in column_names}
    
    # Basic stats for numeric columns only
    numeric_statistics = {}
    if numeric_columns:
        stats_df = df.select(numeric_columns).describe()
        stats_pandas_df = stats_df.toPandas().set_index('summary')
        
        for column_name in numeric_columns:
            try:
                numeric_statistics[column_name] = {
                    'mean': round(float(stats_pandas_df.loc['mean', column_name]), 4),
                    'min': float(stats_pandas_df.loc['min', column_name]),
                    'max': float(stats_pandas_df.loc['max', column_name]),
                    'stddev': round(float(stats_pandas_df.loc['stddev', column_name]), 4)
                }
            except:
                numeric_statistics[column_name] = {'error': 'Stats_Error'}
    
    # Build results
    edd_results = []
    for field_index, column_name in enumerate(column_names, 1):
        non_null_count = total_rows - null_counts_dict[column_name]
        
        edd_result = {
            'Field_Num': field_index,
            'Field_Name': column_name,
            'Type': 'Numeric' if column_name in numeric_columns else 'Categorical',
            'Num_Blanks': null_counts_dict[column_name],
            'Num_Entries': non_null_count,
            'Num_Unique': unique_counts_dict[column_name]
        }
        
        if column_name in numeric_statistics:
            edd_result.update(numeric_statistics[column_name])
        
        edd_results.append(edd_result)
    
    df.unpersist()
    
    # Save
    try:
        os.makedirs(output_path, exist_ok=True)
    except:
        output_path = "/tmp/edd"
        os.makedirs(output_path, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{table_name.replace('.', '_')}_edd_ultrafast_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    pd.DataFrame(edd_results).to_csv(filepath, index=False)
    
    runtime_minutes = (time.time() - start_time) / 60
    print(f"Ultra-fast EDD completed in {runtime_minutes:.1f} minutes")
    print(f"EDD saved: {filepath}")
    
    return filepath

def batch_edd_fast(table_list, output_path="/tmp/edd", sample_rate=None, ultra_fast=False):
    """
    Batch EDD with fast processing
    """
    
    processing_function = generate_edd_ultra_fast if ultra_fast else generate_edd_fast
    mode_description = "ultra-fast" if ultra_fast else "fast"
    
    print(f"Batch EDD ({mode_description} mode) for {len(table_list)} tables")
    batch_results = {}
    
    for table_index, table_name in enumerate(table_list, 1):
        print(f"\n[{table_index}/{len(table_list)}] {table_name}")
        try:
            result_filepath = processing_function(table_name, output_path, sample_rate=sample_rate)
            batch_results[table_name] = result_filepath
        except Exception as error:
            print(f"ERROR: {error}")
            batch_results[table_name] = f"ERROR: {error}"
    
    successful_count = len([result for result in batch_results.values() if result.endswith('.csv')])
    print(f"\nBatch complete: {successful_count}/{len(table_list)} successful")
    
    return batch_results

# Usage examples
if __name__ == "__main__":
    print("Complete EDD System Ready")
    print("Functions available:")
    print("- generate_edd_fast(table_name) - Full EDD with all statistics")
    print("- generate_edd_ultra_fast(table_name) - Minimal statistics, fastest")
    print("- batch_edd_fast(table_list) - Process multiple tables")
    print()
    print("Example usage:")
    print('generate_edd_fast("your_table_name")')
    print('generate_edd_fast("large_table", filter_condition="year >= 2024")')
    print('generate_edd_fast("huge_table", sample_rate=0.01)')
