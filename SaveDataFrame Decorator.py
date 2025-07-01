import functools
import os
from datetime import datetime
from pyspark.sql import DataFrame
from typing import Optional, Union, Callable

def save_dataframe(
    save_path: str = "/tmp/dataframe_snapshots",
    save_format: str = "csv",
    when: str = "after",  # "before", "after", or "both"
    include_timestamp: bool = True,
    use_pandas_fallback: bool = True,
    coalesce_partitions: int = 1,
    **save_options
):
    """
    Flexible decorator that supports both calling patterns:
    1. .transform(function_name)      # Direct reference
    2. .transform(function_name())    # Called with parentheses
    3. .transform(function_name(param=value))  # With parameters
    
    Usage Examples:
        # All of these work:
        df.transform(lower_case_names)
        df.transform(lower_case_names())
        df.transform(filter_by_sources(sources=["A", "B"]))
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Case 1: Called with DataFrame as first argument (direct transform usage)
            if args and isinstance(args[0], DataFrame):
                return _execute_with_save(func, args[0], *args[1:], **kwargs)
            
            # Case 2: Called with no DataFrame (function() pattern)
            # Return a function that can be used with .transform()
            def transform_function(df: DataFrame) -> DataFrame:
                return _execute_with_save(func, df, *args, **kwargs)
            
            return transform_function
        
        def _execute_with_save(func, df: DataFrame, *args, **kwargs) -> DataFrame:
            """Execute the function with save logic"""
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
            func_name = func.__name__
            
            def _save_dataframe(dataframe: DataFrame, suffix: str = ""):
                """Internal function to handle the actual saving"""
                # Build filename
                if include_timestamp:
                    filename = f"{func_name}_{timestamp}{suffix}"
                else:
                    filename = f"{func_name}{suffix}"
                
                # Create full path
                func_dir = os.path.join(save_path, func_name)
                file_path = os.path.join(func_dir, filename)
                
                # Ensure directory exists
                try:
                    os.makedirs(func_dir, exist_ok=True)
                except Exception as e:
                    print(f" Could not create directory {func_dir}: {e}")
                    return
                
                # Try Spark native write first
                try:
                    _spark_write(dataframe, file_path, save_format, coalesce_partitions, **save_options)
                    print(f" Saved (Spark): {file_path}")
                    
                except Exception as spark_error:
                    print(f" Spark write failed: {spark_error}")
                    
                    if use_pandas_fallback:
                        try:
                            _pandas_write(dataframe, file_path, save_format)
                            print(f" Saved (Pandas fallback): {file_path}")
                        except Exception as pandas_error:
                            print(f" Pandas fallback failed: {pandas_error}")
                            print(f" Could not save dataframe for {func_name}")
                    else:
                        print(f" Save failed for {func_name} (no fallback enabled)")
            
            # Save "before" snapshot if requested
            if when in ["before", "both"]:
                _save_dataframe(df, "_before")
            
            # Execute the actual transformation function
            result_df = func(df, *args, **kwargs)
            
            # Save "after" snapshot if requested
            if when in ["after", "both"]:
                suffix = "_after" if when == "both" else ""
                _save_dataframe(result_df, suffix)
            
            return result_df
        
        return wrapper
    return decorator


def _spark_write(dataframe: DataFrame, file_path: str, save_format: str, 
                coalesce_partitions: int, **save_options):
    """Handle Spark native writing with proper error handling"""
    
    # Coalesce to reduce number of output files
    if coalesce_partitions and coalesce_partitions > 0:
        df_to_save = dataframe.coalesce(coalesce_partitions)
    else:
        df_to_save = dataframe
    
    # Write based on format
    if save_format.lower() == "csv":
        df_to_save.write \
            .mode("overwrite") \
            .option("header", "true") \
            .option("timestampFormat", "yyyy-MM-dd HH:mm:ss") \
            .option("dateFormat", "yyyy-MM-dd") \
            .csv(file_path, **save_options)
            
    elif save_format.lower() == "parquet":
        df_to_save.write \
            .mode("overwrite") \
            .parquet(file_path, **save_options)
            
    elif save_format.lower() == "json":
        df_to_save.write \
            .mode("overwrite") \
            .json(file_path, **save_options)
    else:
        # Generic format
        df_to_save.write \
            .mode("overwrite") \
            .format(save_format) \
            .save(file_path, **save_options)


def _pandas_write(dataframe: DataFrame, file_path: str, save_format: str):
    """Handle pandas fallback writing"""
    pandas_df = dataframe.toPandas()
    
    if save_format.lower() == "csv":
        csv_path = f"{file_path}.csv"
        pandas_df.to_csv(csv_path, index=False)
    elif save_format.lower() == "parquet":
        parquet_path = f"{file_path}.parquet"
        pandas_df.to_parquet(parquet_path, index=False)
    elif save_format.lower() == "json":
        json_path = f"{file_path}.json"
        pandas_df.to_json(json_path, orient="records", indent=2)
    else:
        csv_path = f"{file_path}.csv"
        pandas_df.to_csv(csv_path, index=False)


# EXAMPLE USAGE - ALL PATTERNS WORK:

@save_dataframe(save_path="/tmp/transforms", when="after")
@spark_transform
def lower_case_names(df: DataFrame) -> DataFrame:
    """Convert all column names to lowercase"""
    from pyspark.sql.functions import col
    new_cols = [col(c).alias(c.lower()) for c in df.columns]
    return df.select(*new_cols)

@save_dataframe(save_path="/tmp/transforms", when="after")  
@spark_transform
def filter_by_sources(df: DataFrame, sources: list) -> DataFrame:
    """Filter dataframe by source list"""
    from pyspark.sql.functions import col
    return df.filter(col("source").isin(sources))

@save_dataframe(save_path="/tmp/transforms", when="after")
@spark_transform
def filter_active_records(df: DataFrame, status: str = "active") -> DataFrame:
    """Filter for records with specific status"""
    from pyspark.sql.functions import col
    return df.filter(col("status") == status)

# NOW ALL THESE PATTERNS WORK:

# Pattern 1: Direct function reference (original PySpark way)
result1 = df.transform(lower_case_names)

# Pattern 2: Function call with no parameters (your preferred way)  
result2 = df.transform(lower_case_names())

# Pattern 3: Function call with parameters (no lambda needed!)
result3 = df.transform(filter_by_sources(sources=["A", "B", "C"]))

# Pattern 4: Function call with keyword parameters
result4 = df.transform(filter_active_records(status="live"))
