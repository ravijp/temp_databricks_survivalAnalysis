from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, desc, lit, isnan, isnull, approx_count_distinct
from pyspark.sql.types import DoubleType, StringType
import pandas as pd
import os
import time
from typing import List, Optional, Dict, Any

def generate_edd(table_name: str, output_path: str = "/dbfs/tmp/edd", 
                filter_condition: Optional[str] = None, 
                sample_threshold: int = 50_000_000) -> str:
    """
    Fast EDD generation with corrected logic
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
    
    # Batch operations for basic statistics (FAST)
    print("Computing basic statistics...")
    
    # Batch null counts
    null_exprs = [count(when(col(c).isNull() | (col(c) == ""), c)).alias(f"null_{c}") for c in columns]
    null_counts = df.select(null_exprs).collect()[0].asDict()
    
    # Batch unique counts  
    unique_exprs = [approx_count_distinct(col(c)).alias(f"unique_{c}") for c in columns]
    unique_counts = df.select(unique_exprs).collect()[0].asDict()
    
    # Quick type detection - test numeric conversion without collecting data
    print("Detecting column types...")
    numeric_columns, categorical_columns = _detect_column_types(df, columns)
    
    print(f"Numeric: {len(numeric_columns)}, Categorical: {len(categorical_columns)}")
    
    # Batch numeric statistics (FAST)
    numeric_stats = {}
    if numeric_columns:
        print("Computing numeric statistics...")
        numeric_stats = _get_batch_numeric_stats(df, numeric_columns)
    
    # Process results
    print("Building results...")
    results = []
    
    for i, column_name in enumerate(columns, 1):
        null_count = null_counts[f"null_{column_name}"]
        unique_count = unique_counts[f"unique_{column_name}"]
        non_null_count = total_rows - null_count
        
        if column_name in numeric_columns:
            # Numeric column
            result = _build_numeric_result(i, column_name, null_count, non_null_count, 
                                         unique_count, numeric_stats.get(column_name, {}))
        else:
            # Categorical column - only collect data for categorical processing
            result = _build_categorical_result(df, i, column_name, null_count, 
                                             non_null_count, unique_count)
        
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

def _detect_column_types(df, columns: List[str]) -> tuple[List[str], List[str]]:
    """Fast column type detection using Spark operations"""
    
    numeric_columns = []
    categorical_columns = []
    
    for column_name in columns:
        try:
            # Quick test: try to cast and count nulls after conversion
            original_nulls = df.filter(col(column_name).isNull() | (col(column_name) == "")).count()
            
            # Cast to double and count new nulls
            casted_nulls = df.select(col(column_name).cast(DoubleType()).alias("test")).filter(col("test").isNull()).count()
            
            # If new nulls appeared after casting, it means non-numeric data exists
            if casted_nulls > original_nulls:
                categorical_columns.append(column_name)
            else:
                numeric_columns.append(column_name)
                
        except Exception:
            # Any error means categorical
            categorical_columns.append(column_name)
    
    return numeric_columns, categorical_columns

def _get_batch_numeric_stats(df, numeric_columns: List[str]) -> Dict:
    """Get numeric statistics for all numeric columns in batch"""
    
    if not numeric_columns:
        return {}
    
    # Use Spark's built-in describe for basic stats
    numeric_df = df.select([col(c).cast(DoubleType()).alias(c) for c in numeric_columns])
    stats_df = numeric_df.describe()
    stats_pandas = stats_df.toPandas().set_index('summary')
    
    # Get percentiles for all numeric columns in batch
    percentile_stats = {}
    percentile_values = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    
    for column_name in numeric_columns:
        try:
            percentiles = numeric_df.select(column_name).na.drop().approxQuantile(column_name, percentile_values, 0.01)
            percentile_stats[column_name] = percentiles
        except:
            percentile_stats[column_name] = [None] * 7
    
    # Combine stats
    combined_stats = {}
    for column_name in numeric_columns:
        try:
            combined_stats[column_name] = {
                'mean': _safe_float(stats_pandas.loc['mean', column_name]),
                'stddev': _safe_float(stats_pandas.loc['stddev', column_name]),
                'min': _safe_float(stats_pandas.loc['min', column_name]),
                'max': _safe_float(stats_pandas.loc['max', column_name]),
                'percentiles': percentile_stats.get(column_name, [None] * 7)
            }
        except Exception as e:
            combined_stats[column_name] = {'error': str(e)}
    
    return combined_stats

def _safe_float(value) -> Optional[float]:
    """Safely convert to float, handling inf and nan"""
    try:
        f_val = float(value)
        if f_val == float('inf') or f_val == float('-inf') or f_val != f_val:  # NaN check
            return None
        return round(f_val, 6)
    except:
        return None

def _build_numeric_result(field_num: int, column_name: str, null_count: int, 
                         non_null_count: int, unique_count: int, stats: Dict) -> Dict:
    """Build numeric column result"""
    
    if 'error' in stats or not stats:
        return {
            'Field_Num': field_num,
            'Field_Name': column_name,
            'Type': 'Numeric',
            'Num_Blanks': null_count,
            'Num_Entries': non_null_count,
            'Num_Unique': unique_count,
            'Stddev': None,
            'Mean_or_Top1': None,
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
    
    percentiles = stats.get('percentiles', [None] * 7)
    
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Numeric',
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Stddev': stats.get('stddev'),
        'Mean_or_Top1': stats.get('mean'),
        'Min_or_Top2': stats.get('min'),
        'P1_or_Top3': percentiles[0],
        'P5_or_Top4': percentiles[1],
        'P25_or_Top5': percentiles[2],
        'Median_or_Bot5': percentiles[3],
        'P75_or_Bot4': percentiles[4],
        'P95_or_Bot3': percentiles[5],
        'P99_or_Bot2': percentiles[6],
        'Max_or_Bot1': stats.get('max')
    }

def _build_categorical_result(df, field_num: int, column_name: str, null_count: int, 
                            non_null_count: int, unique_count: int) -> Dict:
    """Build categorical column result with correct top/bottom logic"""
    
    # Handle special cases first
    if unique_count == 0:
        default_val = 'No_Data'
    elif unique_count == 1:
        # Get the single value
        try:
            single_row = df.select(column_name).filter((col(column_name).isNotNull()) & (col(column_name) != "")).first()
            if single_row:
                default_val = f"{single_row[0]}:{non_null_count}"
            else:
                default_val = 'All_Same'
        except:
            default_val = 'All_Same'
    elif unique_count == non_null_count:
        default_val = 'All_Unique'
    else:
        # Get value frequencies - collect more data to properly identify top AND bottom
        try:
            # Limit to reasonable size for performance
            limit_size = min(500, unique_count) if unique_count > 100 else unique_count
            
            value_counts = (df.select(column_name)
                          .filter((col(column_name).isNotNull()) & (col(column_name) != ""))
                          .groupBy(column_name)
                          .count()
                          .orderBy(desc("count"), col(column_name))
                          .limit(limit_size)
                          .collect())
            
            if value_counts:
                # Format as value:count
                formatted_values = [f"{row[column_name]}:{row['count']}" for row in value_counts]
                
                # Get top 5 and bottom 5 (following original EDD logic)
                if len(formatted_values) >= 10:
                    return {
                        'Field_Num': field_num,
                        'Field_Name': column_name,
                        'Type': 'Categorical',
                        'Num_Blanks': null_count,
                        'Num_Entries': non_null_count,
                        'Num_Unique': unique_count,
                        'Stddev': None,
                        'Mean_or_Top1': formatted_values[0],  # Top 1
                        'Min_or_Top2': formatted_values[1],   # Top 2
                        'P1_or_Top3': formatted_values[2],    # Top 3
                        'P5_or_Top4': formatted_values[3],    # Top 4
                        'P25_or_Top5': formatted_values[4],   # Top 5
                        'Median_or_Bot5': formatted_values[-5],  # Bottom 5
                        'P75_or_Bot4': formatted_values[-4],     # Bottom 4
                        'P95_or_Bot3': formatted_values[-3],     # Bottom 3
                        'P99_or_Bot2': formatted_values[-2],     # Bottom 2
                        'Max_or_Bot1': formatted_values[-1]      # Bottom 1
                    }
                else:
                    # Less than 10 unique values - just use sequentially
                    padded_values = formatted_values + [''] * (10 - len(formatted_values))
                    return {
                        'Field_Num': field_num,
                        'Field_Name': column_name,
                        'Type': 'Categorical',
                        'Num_Blanks': null_count,
                        'Num_Entries': non_null_count,
                        'Num_Unique': unique_count,
                        'Stddev': None,
                        'Mean_or_Top1': padded_values[0],
                        'Min_or_Top2': padded_values[1],
                        'P1_or_Top3': padded_values[2],
                        'P5_or_Top4': padded_values[3],
                        'P25_or_Top5': padded_values[4],
                        'Median_or_Bot5': padded_values[5],
                        'P75_or_Bot4': padded_values[6],
                        'P95_or_Bot3': padded_values[7],
                        'P99_or_Bot2': padded_values[8],
                        'Max_or_Bot1': padded_values[9]
                    }
        except Exception as e:
            default_val = 'High_Cardinality'
    
    # For special cases, return same value across all fields
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Categorical',
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Stddev': None,
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

def batch_edd(table_list: List[str], output_path: str = "/dbfs/tmp/edd", 
              sample_threshold: int = 50_000_000) -> Dict[str, str]:
    """Process multiple tables with performance tracking"""
    
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
    
    successful = sum(1 for r in results.values() if r.endswith('.csv'))
    total_time = (time.time() - batch_start) / 60
    
    print(f"\nBatch completed: {successful}/{len(table_list)} successful in {total_time:.1f} minutes")
    print(f"Summary saved: {summary_file}")
    
    return results

# Helper functions
def list_edd_files(output_path: str = "/dbfs/tmp/edd") -> None:
    """List all EDD files"""
    try:
        if os.path.exists(output_path):
            files = [f for f in os.listdir(output_path) if f.endswith('.csv')]
            if files:
                print(f"EDD files in {output_path}:")
                for i, filename in enumerate(files, 1):
                    filepath = os.path.join(output_path, filename)
                    size_mb = os.path.getsize(filepath) / (1024*1024)
                    print(f"{i:2d}. {filename} ({size_mb:.2f} MB)")
                print(f"\nDownload via: Data → DBFS → tmp → edd")
            else:
                print("No EDD files found")
        else:
            print(f"Directory {output_path} does not exist")
    except Exception as e:
        print(f"Error: {e}")

def display_edd_file(filepath: str, num_rows: int = 20) -> None:
    """Display EDD file in notebook"""
    try:
        df = pd.read_csv(filepath)
        print(f"EDD File: {os.path.basename(filepath)}")
        print(f"Columns: {len(df)} | Size: {os.path.getsize(filepath) / 1024:.1f} KB")
        print("-" * 60)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        display(df.head(num_rows))
        
        if len(df) > num_rows:
            print(f"... showing first {num_rows} of {len(df)} total columns")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Fast EDD system ready:")
    print("- generate_edd(table_name)")  
    print("- batch_edd(table_list)")
    print("- list_edd_files()")
    print("- display_edd_file(filepath)")
