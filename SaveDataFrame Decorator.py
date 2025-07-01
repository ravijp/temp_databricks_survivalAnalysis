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
    Supports both .transform(function) and .transform(function()) patterns.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Direct DataFrame call
            if args and isinstance(args[0], DataFrame):
                return _execute_with_save(func, args[0], *args[1:], **kwargs)
            
            # Return transform function for .transform() usage
            def transform_function(df):
                return _execute_with_save(func, df, *args, **kwargs)
            return transform_function
        
        def _execute_with_save(func, df, *args, **kwargs):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
            func_name = func.__name__
            
            def _save_df(dataframe, suffix=""):
                # Check record count first
                try:
                    record_count = dataframe.count()
                    if record_count > max_records_to_save:
                        print(f"Skipping save for {func_name}{suffix} - {record_count} records exceeds limit of {max_records_to_save}")
                        return
                except Exception as e:
                    print(f"Warning: Could not count records for {func_name}: {e}")
                    print("Proceeding with save")
                
                # Build file path
                filename = f"{func_name}_{timestamp}{suffix}" if include_timestamp else f"{func_name}{suffix}"
                func_dir = os.path.join(save_path, func_name)
                file_path = os.path.join(func_dir, filename)
                
                # Create directory
                os.makedirs(func_dir, exist_ok=True)
                
                # Try Spark write first
                try:
                    df_to_save = dataframe.coalesce(coalesce_partitions) if coalesce_partitions else dataframe
                    
                    if save_format == "csv":
                        df_to_save.write.mode("overwrite").option("header", "true").csv(file_path, **save_options)
                    elif save_format == "parquet":
                        df_to_save.write.mode("overwrite").parquet(file_path, **save_options)
                    elif save_format == "json":
                        df_to_save.write.mode("overwrite").json(file_path, **save_options)
                    else:
                        df_to_save.write.mode("overwrite").format(save_format).save(file_path, **save_options)
                    
                    print(f"Saved {func_name}{suffix} to {file_path}")
                    
                except Exception as e:
                    if use_pandas_fallback:
                        try:
                            pandas_path = f"{file_path}.csv"
                            dataframe.toPandas().to_csv(pandas_path, index=False)
                            print(f"Saved {func_name}{suffix} to {pandas_path} (pandas fallback)")
                        except Exception as pandas_error:
                            print(f"Failed to save {func_name}{suffix}: Spark error: {e}, Pandas error: {pandas_error}")
                    else:
                        print(f"Failed to save {func_name}{suffix}: {e}")
            
            # Save before if requested
            if when in ["before", "both"]:
                _save_df(df, "_before")
            
            # Execute transformation
            result_df = func(df, *args, **kwargs)
            
            # Save after if requested
            if when in ["after", "both"]:
                suffix = "_after" if when == "both" else ""
                _save_df(result_df, suffix)
            
            return result_df
        
        return wrapper
    return decorator


# Usage examples
@save_dataframe(save_path="/tmp/transforms")
def lower_case_names(df):
    from pyspark.sql.functions import col
    return df.select([col(c).alias(c.lower()) for c in df.columns])

@save_dataframe(save_path="/tmp/transforms", max_records_to_save=50000)
def filter_by_sources(df, sources):
    from pyspark.sql.functions import col
    return df.filter(col("source").isin(sources))

@save_dataframe(save_path="/tmp/transforms", when="both")
def filter_active_records(df, status="active"):
    from pyspark.sql.functions import col
    return df.filter(col("status") == status)

# All calling patterns work
# df.transform(lower_case_names)
# df.transform(lower_case_names())
# df.transform(filter_by_sources(sources=["A", "B"]))

# Configuration class
class SaveConfig:
    BASE_PATH = "/data/snapshots"
    FORMAT = "parquet"
    MAX_RECORDS = 25000
    
    @classmethod
    def get_decorator(cls, when="after", max_records=None):
        return save_dataframe(
            save_path=cls.BASE_PATH,
            save_format=cls.FORMAT,
            when=when,
            max_records_to_save=max_records or cls.MAX_RECORDS
        )

@SaveConfig.get_decorator()
def standard_transform(df):
    return df.dropDuplicates()

@SaveConfig.get_decorator(max_records=100000)
def large_transform(df):
    return df.groupBy("category").count()

