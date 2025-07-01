import functools
import os
from datetime import datetime
from pyspark.sql import DataFrame

def save_dataframe(
    save_path="/tmp/dataframe_snapshots",
    save_format="csv",
    when="after",
    include_timestamp=True,
    use_pandas_fallback=True,
    coalesce_partitions=1,
    max_records_to_save=10000,
    **save_options
):
    """
    Decorator to save PySpark DataFrames with record count limit.
    Works with @spark_transform decorator and supports both calling patterns.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If called with DataFrame as first argument, execute with save logic
            if args and isinstance(args[0], DataFrame):
                df = args[0]
                other_args = args[1:]
                return execute_with_save(df, other_args, kwargs)
            
            # If called without DataFrame (e.g., function()), return a transform function
            def transform_function(df):
                return execute_with_save(df, args, kwargs)
            return transform_function
        
        def execute_with_save(df, args, kwargs):
            """Execute the function with save logic"""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
            func_name = func.__name__
            
            def save_dataframe_to_disk(dataframe, suffix=""):
                """Save dataframe to disk with all the logic"""
                # Validate that we actually have a DataFrame
                if not isinstance(dataframe, DataFrame):
                    print(f"Warning: Expected DataFrame for {func_name}{suffix}, got {type(dataframe)}")
                    return
                
                # Check record count
                try:
                    record_count = dataframe.count()
                    if record_count > max_records_to_save:
                        print(f"Skipping save for {func_name}{suffix} - {record_count} records exceeds limit of {max_records_to_save}")
                        return
                except Exception as e:
                    print(f"Warning: Could not count records for {func_name}: {e}")
                    return  # Skip save if we can't count
                
                # Build file path
                filename = f"{func_name}_{timestamp}{suffix}" if include_timestamp else f"{func_name}{suffix}"
                func_dir = os.path.join(save_path, func_name)
                file_path = os.path.join(func_dir, filename)
                
                # Create directory
                try:
                    os.makedirs(func_dir, exist_ok=True)
                except Exception as e:
                    print(f"Could not create directory {func_dir}: {e}")
                    return
                
                # Try to save
                try:
                    # Coalesce if specified
                    df_to_save = dataframe.coalesce(coalesce_partitions) if coalesce_partitions else dataframe
                    
                    # Save based on format
                    if save_format == "csv":
                        df_to_save.write.mode("overwrite").option("header", "true").csv(file_path, **save_options)
                    elif save_format == "parquet":
                        df_to_save.write.mode("overwrite").parquet(file_path, **save_options)
                    elif save_format == "json":
                        df_to_save.write.mode("overwrite").json(file_path, **save_options)
                    else:
                        df_to_save.write.mode("overwrite").format(save_format).save(file_path, **save_options)
                    
                    print(f"Saved {func_name}{suffix} to {file_path}")
                    
                except Exception as spark_error:
                    if use_pandas_fallback:
                        try:
                            pandas_path = f"{file_path}.csv"
                            dataframe.toPandas().to_csv(pandas_path, index=False)
                            print(f"Saved {func_name}{suffix} to {pandas_path} (pandas fallback)")
                        except Exception as pandas_error:
                            print(f"Failed to save {func_name}{suffix}: {spark_error}")
                    else:
                        print(f"Failed to save {func_name}{suffix}: {spark_error}")
            
            # Save before transformation if requested
            if when in ["before", "both"]:
                save_dataframe_to_disk(df, "_before")
            
            # Execute the actual transformation
            try:
                result_df = func(df, *args, **kwargs)
                
                # Validate result is a DataFrame
                if not isinstance(result_df, DataFrame):
                    print(f"Warning: Function {func_name} returned {type(result_df)}, expected DataFrame")
                    return result_df
                
                # Save after transformation if requested
                if when in ["after", "both"]:
                    suffix = "_after" if when == "both" else ""
                    save_dataframe_to_disk(result_df, suffix)
                
                return result_df
                
            except Exception as e:
                print(f"Error executing {func_name}: {e}")
                raise
        
        return wrapper
    return decorator
