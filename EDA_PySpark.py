from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, desc, approx_count_distinct, isnan, isnull
import pandas as pd
import os
import time
from typing import List, Optional, Dict, Any

def generate_edd(table_name: str, output_path: str = "/tmp/edd", 
                filter_condition: Optional[str] = None, 
                sample_threshold: int = 50_000_000) -> str:
    """
    Generate EDD for a single table with intelligent sampling
    
    Args:
        table_name: Spark table name
        output_path: Output directory 
        filter_condition: Optional WHERE clause
        sample_threshold: Row count threshold for sampling
    
    Returns:
        Path to generated CSV file
    """
    
    print(f"Processing: {table_name}")
    start_time = time.time()
    
    # Load and prepare data
    df = spark.table(table_name)
    if filter_condition:
        df = df.filter(filter_condition)
    
    # Get table size and determine strategy
    total_rows = df.count()
    sample_rate = min(1.0, sample_threshold / max(total_rows, 1))
    
    if sample_rate < 1.0:
        df = df.sample(sample_rate, seed=42)
        print(f"Sampling {sample_rate:.3f} of {total_rows:,} rows")
        total_rows = df.count()
    
    df.cache()
    
    columns = df.columns
    print(f"Analyzing {total_rows:,} rows, {len(columns)} columns")
    
    # Batch compute basic statistics
    null_exprs = [count(when(col(c).isNull() | isnan(col(c)), c)).alias(f"null_{c}") for c in columns]
    unique_exprs = [approx_count_distinct(col(c)).alias(f"unique_{c}") for c in columns]
    
    null_counts = df.select(null_exprs).collect()[0].asDict()
    unique_counts = df.select(unique_exprs).collect()[0].asDict()
    
    # Process each column for type detection and stats
    results = []
    for i, column_name in enumerate(columns, 1):
        null_count = null_counts[f"null_{column_name}"]
        unique_count = unique_counts[f"unique_{column_name}"]
        non_null_count = total_rows - null_count
        
        # Attempt numeric processing first
        numeric_stats = _get_numeric_stats(df, column_name)
        
        if numeric_stats:
            # Numeric column
            percentiles = _get_percentiles(df, column_name)
            result = {
                'Field_Num': i,
                'Field_Name': column_name,
                'Type': 'Numeric',
                'Num_Blanks': null_count,
                'Num_Entries': non_null_count,
                'Num_Unique': unique_count,
                'Stddev': numeric_stats.get('stddev'),
                'Mean_or_Top1': numeric_stats.get('mean'),
                'Min_or_Top2': numeric_stats.get('min'),
                'P1_or_Top3': percentiles[0] if percentiles else None,
                'P5_or_Top4': percentiles[1] if percentiles else None,
                'P25_or_Top5': percentiles[2] if percentiles else None,
                'Median_or_Bot5': percentiles[3] if percentiles else None,
                'P75_or_Bot4': percentiles[4] if percentiles else None,
                'P95_or_Bot3': percentiles[5] if percentiles else None,
                'P99_or_Bot2': percentiles[6] if percentiles else None,
                'Max_or_Bot1': numeric_stats.get('max')
            }
        else:
            # Categorical column
            cat_values = _get_categorical_stats(df, column_name, unique_count, non_null_count)
            result = {
                'Field_Num': i,
                'Field_Name': column_name,
                'Type': 'Categorical',
                'Num_Blanks': null_count,
                'Num_Entries': non_null_count,
                'Num_Unique': unique_count,
                'Stddev': None,
                **cat_values
            }
        
        results.append(result)
    
    df.unpersist()
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{table_name.replace('.', '_')}_edd_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    pd.DataFrame(results).to_csv(filepath, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"EDD completed in {elapsed:.1f} minutes: {filepath}")
    
    return filepath

def _get_numeric_stats(df, column_name: str) -> Optional[Dict[str, Any]]:
    """Attempt to get numeric statistics for a column"""
    try:
        # Try to cast to double and get basic stats
        numeric_df = df.select(col(column_name).cast("double").alias(column_name))
        stats = numeric_df.select(column_name).describe().toPandas().set_index('summary')
        
        # Check if we have valid numeric data
        if stats.loc['count', column_name] == '0':
            return None
            
        return {
            'mean': round(float(stats.loc['mean', column_name]), 6),
            'stddev': round(float(stats.loc['stddev', column_name]), 6),
            'min': float(stats.loc['min', column_name]),
            'max': float(stats.loc['max', column_name])
        }
    except:
        return None

def _get_percentiles(df, column_name: str) -> Optional[List[float]]:
    """Get percentiles for numeric column"""
    try:
        percentile_values = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        return df.select(column_name).na.drop().approxQuantile(column_name, percentile_values, 0.01)
    except:
        return None

def _get_categorical_stats(df, column_name: str, unique_count: int, non_null_count: int) -> Dict[str, str]:
    """Get categorical statistics following original EDD format"""
    
    # Handle special cases
    if unique_count == 0:
        default_val = 'No_Data'
    elif unique_count == 1:
        # Get the actual single value
        try:
            single_val = df.select(column_name).filter(col(column_name).isNotNull()).first()
            if single_val:
                default_val = f"{single_val[0]}:{non_null_count}"
            else:
                default_val = 'All_Same'
        except:
            default_val = 'All_Same'
    elif unique_count == non_null_count:
        default_val = 'All_Unique'
    else:
        # Get frequency distribution - need more than 10 to get top AND bottom values
        try:
            # For high cardinality, limit to reasonable number to get top/bottom split
            limit_size = min(1000, max(100, unique_count))
            
            all_values = (df.select(column_name)
                         .filter(col(column_name).isNotNull())
                         .groupBy(column_name)
                         .count()
                         .orderBy(desc("count"), col(column_name))  # Secondary sort for consistency
                         .limit(limit_size)
                         .collect())
            
            if all_values and len(all_values) >= 10:
                # Convert to list of formatted strings
                formatted_values = [f"{row[column_name]}:{row['count']}" for row in all_values]
                
                # Get top 5 and bottom 5 following original EDD logic
                top_5 = formatted_values[:5]
                bottom_5 = formatted_values[-5:] if len(formatted_values) > 5 else formatted_values
                
                # Pad if needed
                top_5.extend([''] * (5 - len(top_5)))
                bottom_5.extend([''] * (5 - len(bottom_5)))
                
                return {
                    'Mean_or_Top1': top_5[0],
                    'Min_or_Top2': top_5[1], 
                    'P1_or_Top3': top_5[2],
                    'P5_or_Top4': top_5[3],
                    'P25_or_Top5': top_5[4],
                    'Median_or_Bot5': bottom_5[-5] if len(bottom_5) >= 5 else bottom_5[0],
                    'P75_or_Bot4': bottom_5[-4] if len(bottom_5) >= 4 else bottom_5[0],
                    'P95_or_Bot3': bottom_5[-3] if len(bottom_5) >= 3 else bottom_5[0],
                    'P99_or_Bot2': bottom_5[-2] if len(bottom_5) >= 2 else bottom_5[0],  
                    'Max_or_Bot1': bottom_5[-1]
                }
            else:
                # Fallback for low data
                cat_values = [f"{row[column_name]}:{row['count']}" for row in all_values] if all_values else []
                cat_values.extend([''] * (10 - len(cat_values)))
                
                return {
                    'Mean_or_Top1': cat_values[0],
                    'Min_or_Top2': cat_values[1],
                    'P1_or_Top3': cat_values[2], 
                    'P5_or_Top4': cat_values[3],
                    'P25_or_Top5': cat_values[4],
                    'Median_or_Bot5': cat_values[5],
                    'P75_or_Bot4': cat_values[6],
                    'P95_or_Bot3': cat_values[7],
                    'P99_or_Bot2': cat_values[8],
                    'Max_or_Bot1': cat_values[9]
                }
        except:
            cat_values = ['High_Cardinality'] * 10
            return {
                'Mean_or_Top1': cat_values[0],
                'Min_or_Top2': cat_values[1], 
                'P1_or_Top3': cat_values[2],
                'P5_or_Top4': cat_values[3],
                'P25_or_Top5': cat_values[4],
                'Median_or_Bot5': cat_values[5],
                'P75_or_Bot4': cat_values[6],
                'P95_or_Bot3': cat_values[7],
                'P99_or_Bot2': cat_values[8],
                'Max_or_Bot1': cat_values[9]
            }
    
    # For special cases, use same value across all fields
    return {
        'Mean_or_Top1': default_val,
        'Min_or_Top2': default_val,
        'P1_or_Top3': default_val,
        'P5_or_Top4': default_val,
        'P25_or_Top5': default_val,
        'Median_or_Bot5': default_val,
        'P75_or_Bot4': default_val,
        'P95_or_Bot3': default_val,
        'P99_or_Bot2': default_val,
        'Max_or_Bot1': default_val
    }

def batch_edd(table_list: List[str], output_path: str = "/tmp/edd", 
              sample_threshold: int = 50_000_000) -> Dict[str, str]:
    """
    Process multiple tables and generate batch summary
    
    Args:
        table_list: List of table names
        output_path: Output directory
        sample_threshold: Row threshold for sampling
    
    Returns:
        Dictionary mapping table names to file paths or error messages
    """
    
    print(f"Starting batch EDD for {len(table_list)} tables")
    batch_start = time.time()
    
    results = {}
    summary_data = []
    
    for i, table_name in enumerate(table_list, 1):
        print(f"\n[{i}/{len(table_list)}] {table_name}")
        
        try:
            start_time = time.time()
            filepath = generate_edd(table_name, output_path, sample_threshold=sample_threshold)
            elapsed = time.time() - start_time
            
            results[table_name] = filepath
            
            # Collect summary info
            summary_data.append({
                'Table_Name': table_name,
                'Status': 'Success',
                'File_Path': filepath,
                'Processing_Time_Minutes': round(elapsed / 60, 2)
            })
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            results[table_name] = error_msg
            print(error_msg)
            
            summary_data.append({
                'Table_Name': table_name,
                'Status': 'Failed',
                'File_Path': '',
                'Processing_Time_Minutes': 0,
                'Error': str(e)
            })
    
    # Generate batch summary
    summary_file = os.path.join(output_path, f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    
    # Print final summary
    successful = sum(1 for r in results.values() if r.endswith('.csv'))
    total_time = (time.time() - batch_start) / 60
    
    print(f"\nBatch completed: {successful}/{len(table_list)} successful in {total_time:.1f} minutes")
    print(f"Summary saved: {summary_file}")
    
    return results

# Usage examples
if __name__ == "__main__":
    # Single table
    # generate_edd("my_database.large_table")
    
    # Batch processing
    # tables = ["db.table1", "db.table2", "db.table3"]
    # batch_edd(tables)
    
    print("EDD system ready. Key functions:")
    print("- generate_edd(table_name)")  
    print("- batch_edd(table_list)")
