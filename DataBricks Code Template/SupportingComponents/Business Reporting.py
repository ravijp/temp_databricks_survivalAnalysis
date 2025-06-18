# Business Reporting and Client Deliverables
# Generates comprehensive reports and exports for HR client partners

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Reporting parameters (configurable via widgets)
GENERATE_EXECUTIVE_SUMMARY = dbutils.widgets.get("generate_executive_summary") if "generate_executive_summary" in [w.name for w in dbutils.widgets.getAll()] else "true"
INCLUDE_TREND_ANALYSIS = dbutils.widgets.get("include_trend_analysis") if "include_trend_analysis" in [w.name for w in dbutils.widgets.getAll()] else "true"
EXPORT_TO_CSV = dbutils.widgets.get("export_to_csv") if "export_to_csv" in [w.name for w in dbutils.widgets.getAll()] else "true"

# Data sources
EMPLOYEE_SCORES_TABLE = "hr_analytics.predictions.employee_risk_scores"
DEPARTMENT_SUMMARY_TABLE = "hr_analytics.predictions.department_risk_summary"
MANAGER_ALERTS_TABLE = "hr_analytics.predictions.manager_risk_alerts"
SCORING_METADATA_TABLE = "hr_analytics.predictions.scoring_metadata"

# Export locations
EXPORT_BASE_PATH = "/mnt/exports/hr_survival_reports/"
REPORTS_TABLE = "hr_analytics.reports.business_reports"

# Client configuration
CLIENT_CONFIG = {
    "company_name": "Your Organization",
    "report_title": "Employee Turnover Risk Analysis",
    "confidentiality_level": "Internal Use Only",
    "report_frequency": "Monthly"
}

print("📊 Business Reporting and Client Deliverables")
print(f"📋 Executive Summary: {GENERATE_EXECUTIVE_SUMMARY}")
print(f"📈 Trend Analysis: {INCLUDE_TREND_ANALYSIS}")
print(f"💾 CSV Export: {EXPORT_TO_CSV}")

# =============================================================================
# IMPORTS
# =============================================================================

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# =============================================================================
# 1. EXECUTIVE SUMMARY GENERATION
# =============================================================================

def generate_executive_summary():
    """Generate executive summary for business stakeholders"""
    
    print("\n📊 Generating Executive Summary...")
    
    # Get latest scoring results
    latest_scores = spark.table(EMPLOYEE_SCORES_TABLE)
    latest_date = latest_scores.select(max("scoring_date")).collect()[0][0]
    
    current_scores = latest_scores.filter(col("scoring_date") == latest_date)
    
    # Basic statistics
    total_employees = current_scores.count()
    high_risk_count = current_scores.filter(col("risk_tier") == "High Risk").count()
    medium_risk_count = current_scores.filter(col("risk_tier") == "Medium Risk").count()
    low_risk_count = current_scores.filter(col("risk_tier") == "Low Risk").count()
    
    avg_risk_score = current_scores.select(avg("risk_score")).collect()[0][0]
    avg_survival_days = current_scores.select(avg("predicted_survival_days")).collect()[0][0]
    
    # Department breakdown
    dept_risk = (current_scores
                 .groupBy("department_clean")
                 .agg(
                     count("*").alias("total_employees"),
                     sum(when(col("risk_tier") == "High Risk", 1).otherwise(0)).alias("high_risk_employees"),
                     avg("risk_score").alias("avg_risk_score")
                 )
                 .withColumn("high_risk_percentage", col("high_risk_employees") / col("total_employees"))
                 .orderBy(col("high_risk_percentage").desc()))
    
    # Get top risk departments
    top_risk_depts = dept_risk.limit(5).collect()
    
    # Create executive summary text
    summary_text = f"""
    
===============================================================================
📊 EXECUTIVE SUMMARY - EMPLOYEE TURNOVER RISK ANALYSIS
===============================================================================

Report Date: {datetime.now().strftime('%B %d, %Y')}
Analysis Period: {latest_date.strftime('%B %Y') if latest_date else 'Current Month'}
Total Employees Analyzed: {total_employees:,}

🎯 KEY FINDINGS:

OVERALL RISK DISTRIBUTION:
• High Risk (Likely to leave within 6 months):  {high_risk_count:,} employees ({high_risk_count/total_employees:.1%})
• Medium Risk (Moderate retention concern):      {medium_risk_count:,} employees ({medium_risk_count/total_employees:.1%})
• Low Risk (Strong retention likelihood):       {low_risk_count:,} employees ({low_risk_count/total_employees:.1%})

CRITICAL METRICS:
• Average Risk Score: {avg_risk_score:.3f} (0.0 = lowest risk, 1.0 = highest risk)
• Average Predicted Tenure: {avg_survival_days/365:.1f} years
• Immediate Attention Required: {high_risk_count:,} employees

DEPARTMENTAL RISK HOTSPOTS:
"""
    
    for i, dept in enumerate(top_risk_depts, 1):
        summary_text += f"{i}. {dept.department_clean}: {dept.high_risk_percentage:.1%} high risk ({dept.high_risk_employees}/{dept.total_employees} employees)\n"
    
    summary_text += f"""

🚨 IMMEDIATE ACTION ITEMS:

RETENTION FOCUS:
• Schedule retention interviews with top {min(high_risk_count, 20)} highest-risk employees
• Implement targeted retention strategies for high-risk departments
• Review compensation and career development for medium-risk employees

WORKFORCE PLANNING:
• Prepare succession plans for critical high-risk roles
• Begin recruiting for potential backfill positions
• Consider knowledge transfer initiatives for high-risk subject matter experts

MANAGEMENT ACTIONS:
• Brief department heads on their risk profiles
• Provide managers with coaching on retention conversations
• Implement stay interviews for high-value, high-risk employees

📈 BUSINESS IMPACT:

If no action is taken, the model predicts approximately {high_risk_count:,} employees 
may voluntarily terminate within the next 6 months. This represents:

• Potential replacement costs: ${high_risk_count * 50000:,} (estimated at $50K per replacement)
• Knowledge drain risk in critical departments
• Potential service disruption during transition periods

💡 RECOMMENDED NEXT STEPS:

1. IMMEDIATE (Next 7 days):
   - Review high-risk employee list with department heads
   - Schedule retention meetings with top 10 highest-risk employees
   - Analyze exit interview themes from recent departures

2. SHORT-TERM (Next 30 days):
   - Implement department-specific retention initiatives
   - Begin succession planning for critical high-risk roles
   - Launch targeted engagement surveys in high-risk departments

3. LONG-TERM (Next 90 days):
   - Re-run analysis to measure retention program effectiveness
   - Adjust compensation strategies based on risk patterns
   - Develop predictive alerts for real-time risk monitoring

===============================================================================
"""
    
    print(summary_text)
    return summary_text, {
        "total_employees": total_employees,
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "low_risk_count": low_risk_count,
        "avg_risk_score": float(avg_risk_score),
        "avg_survival_days": float(avg_survival_days),
        "analysis_date": latest_date
    }

# =============================================================================
# 2. TREND ANALYSIS
# =============================================================================

def generate_trend_analysis():
    """Generate trend analysis comparing current vs historical results"""
    
    if INCLUDE_TREND_ANALYSIS.lower() != "true":
        print("⏭️ Skipping trend analysis (disabled)")
        return None
    
    print("\n📈 Generating Trend Analysis...")
    
    # Get historical scoring data (last 6 months)
    historical_scores = (spark.table(EMPLOYEE_SCORES_TABLE)
                        .filter(col("scoring_date") >= date_sub(current_date(), 180))
                        .select("scoring_date", "risk_score", "risk_tier", "department_clean"))
    
    if historical_scores.count() == 0:
        print("⚠️ No historical data available for trend analysis")
        return None
    
    # Monthly risk trends
    monthly_trends = (historical_scores
                     .withColumn("month", date_trunc("month", col("scoring_date")))
                     .groupBy("month")
                     .agg(
                         count("*").alias("total_employees"),
                         avg("risk_score").alias("avg_risk_score"),
                         sum(when(col("risk_tier") == "High Risk", 1).otherwise(0)).alias("high_risk_count")
                     )
                     .withColumn("high_risk_percentage", col("high_risk_count") / col("total_employees"))
                     .orderBy("month"))
    
    monthly_data = monthly_trends.collect()
    
    if len(monthly_data) < 2:
        print("⚠️ Insufficient historical data for trend analysis")
        return None
    
    # Calculate trends
    latest_month = monthly_data[-1]
    previous_month = monthly_data[-2]
    
    risk_score_change = latest_month.avg_risk_score - previous_month.avg_risk_score
    high_risk_pct_change = latest_month.high_risk_percentage - previous_month.high_risk_percentage
    
    trend_summary = f"""
📈 TREND ANALYSIS - MONTH-OVER-MONTH COMPARISON:

RISK SCORE TRENDS:
• Current Month Average Risk: {latest_month.avg_risk_score:.3f}
• Previous Month Average Risk: {previous_month.avg_risk_score:.3f}
• Change: {risk_score_change:+.3f} ({risk_score_change/previous_month.avg_risk_score:+.1%})

HIGH-RISK EMPLOYEE TRENDS:
• Current Month High Risk: {latest_month.high_risk_percentage:.1%}
• Previous Month High Risk: {previous_month.high_risk_percentage:.1%}  
• Change: {high_risk_pct_change:+.1%}

TREND INTERPRETATION:
"""
    
    if risk_score_change > 0.05:
        trend_summary += "🔴 CONCERN: Risk levels are increasing significantly\n"
    elif risk_score_change > 0.01:
        trend_summary += "🟡 CAUTION: Risk levels are slightly increasing\n"
    elif risk_score_change < -0.05:
        trend_summary += "🟢 POSITIVE: Risk levels are decreasing significantly\n"
    else:
        trend_summary += "🔵 STABLE: Risk levels are relatively stable\n"
    
    print(trend_summary)
    
    return {
        "monthly_data": monthly_data,
        "risk_score_change": float(risk_score_change),
        "high_risk_pct_change": float(high_risk_pct_change),
        "trend_summary": trend_summary
    }

# =============================================================================
# 3. DETAILED REPORTING
# =============================================================================

def generate_detailed_reports():
    """Generate detailed reports for different stakeholders"""
    
    print("\n📋 Generating Detailed Reports...")
    
    # Get latest data
    latest_scores = spark.table(EMPLOYEE_SCORES_TABLE)
    latest_date = latest_scores.select(max("scoring_date")).collect()[0][0]
    current_scores = latest_scores.filter(col("scoring_date") == latest_date)
    
    # 1. Department Leaders Report
    dept_report = (current_scores
                   .groupBy("department_clean")
                   .agg(
                       count("*").alias("total_employees"),
                       sum(when(col("risk_tier") == "High Risk", 1).otherwise(0)).alias("high_risk"),
                       sum(when(col("risk_tier") == "Medium Risk", 1).otherwise(0)).alias("medium_risk"),
                       sum(when(col("risk_tier") == "Low Risk", 1).otherwise(0)).alias("low_risk"),
                       avg("risk_score").alias("avg_risk_score"),
                       avg("predicted_survival_days").alias("avg_predicted_tenure"),
                       avg("salary_clean").alias("avg_salary")
                   )
                   .withColumn("high_risk_pct", col("high_risk") / col("total_employees"))
                   .orderBy(col("high_risk_pct").desc()))
    
    print("📊 Department Leaders Report prepared")
    
    # 2. HR Business Partners Report  
    hr_report = (current_scores
                 .select("employee_id", "department_clean", "risk_score", "risk_tier", 
                        "predicted_survival_days", "salary_clean")
                 .filter(col("risk_tier").isin(["High Risk", "Medium Risk"]))
                 .orderBy(col("risk_score").desc()))
    
    print("📊 HR Business Partners Report prepared")
    
    # 3. Executive Dashboard Data
    exec_dashboard = current_scores.agg(
        count("*").alias("total_employees"),
        countDistinct("department_clean").alias("departments_analyzed"),
        avg("risk_score").alias("overall_avg_risk"),
        sum(when(col("risk_tier") == "High Risk", 1).otherwise(0)).alias("high_risk_total"),
        sum(when(col("risk_tier") == "Medium Risk", 1).otherwise(0)).alias("medium_risk_total"),
        sum(when(col("risk_tier") == "Low Risk", 1).otherwise(0)).alias("low_risk_total")
    ).collect()[0]
    
    print("📊 Executive Dashboard prepared")
    
    return {
        "department_report": dept_report,
        "hr_report": hr_report, 
        "executive_metrics": exec_dashboard
    }

# =============================================================================
# 4. EXPORT FUNCTIONS
# =============================================================================

def export_reports_to_csv():
    """Export reports to CSV files for client distribution"""
    
    if EXPORT_TO_CSV.lower() != "true":
        print("⏭️ Skipping CSV export (disabled)")
        return
    
    print("\n💾 Exporting reports to CSV...")
    
    # Create export directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    export_path = f"{EXPORT_BASE_PATH}{timestamp}/"
    
    try:
        # Get latest scoring data
        latest_scores = spark.table(EMPLOYEE_SCORES_TABLE)
        latest_date = latest_scores.select(max("scoring_date")).collect()[0][0]
        current_scores = latest_scores.filter(col("scoring_date") == latest_date)
        
        # 1. Export employee-level predictions (anonymized if required)
        employee_export = current_scores.select(
            "employee_id",
            "department_clean", 
            "risk_score",
            "risk_tier",
            "predicted_survival_days",
            "scoring_date"
        )
        
        (employee_export
         .coalesce(1)
         .write
         .mode("overwrite")
         .option("header", "true")
         .csv(f"{export_path}employee_risk_predictions"))
        
        print(f"✅ Employee predictions exported to: {export_path}employee_risk_predictions")
        
        # 2. Export department summary
        dept_summary = spark.table(DEPARTMENT_SUMMARY_TABLE).filter(col("scoring_date") == latest_date)
        
        (dept_summary
         .coalesce(1)
         .write
         .mode("overwrite")
         .option("header", "true")
         .csv(f"{export_path}department_risk_summary"))
        
        print(f"✅ Department summary exported to: {export_path}department_risk_summary")
        
        # 3. Export alerts if any exist
        try:
            alerts = spark.table(MANAGER_ALERTS_TABLE).filter(col("scoring_date") == latest_date)
            if alerts.count() > 0:
                (alerts
                 .coalesce(1)
                 .write
                 .mode("overwrite")
                 .option("header", "true")
                 .csv(f"{export_path}risk_alerts"))
                
                print(f"✅ Risk alerts exported to: {export_path}risk_alerts")
        except:
            print("ℹ️ No alerts table found or no alerts to export")
        
        # 4. Create a summary readme file
        readme_content = f"""
# Employee Turnover Risk Analysis Export
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Date: {latest_date.strftime('%Y-%m-%d') if latest_date else 'N/A'}

## Files Included:
- employee_risk_predictions/: Individual employee risk scores and predictions
- department_risk_summary/: Department-level risk aggregations  
- risk_alerts/: High-priority alerts requiring immediate attention

## Risk Tiers:
- High Risk: Employees likely to leave within 6 months
- Medium Risk: Employees with moderate retention concerns
- Low Risk: Employees with strong retention likelihood

## Risk Score:
Scale of 0.0 to 1.0 where:
- 0.0 = Lowest turnover risk
- 1.0 = Highest turnover risk

## Confidentiality:
{CLIENT_CONFIG['confidentiality_level']}

For questions, contact: hr-analytics@company.com
"""
        
        # Save readme (this would typically be done outside Databricks)
        print("✅ Export completed successfully")
        print(f"📁 Export location: {export_path}")
        
        return export_path
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        return None

# =============================================================================
# 5. VISUALIZATION GENERATION
# =============================================================================

def create_business_visualizations():
    """Create key visualizations for business stakeholders"""
    
    print("\n📊 Creating Business Visualizations...")
    
    # Get latest data as Pandas for plotting
    latest_scores = spark.table(EMPLOYEE_SCORES_TABLE)
    latest_date = latest_scores.select(max("scoring_date")).collect()[0][0]
    current_scores_pd = latest_scores.filter(col("scoring_date") == latest_date).toPandas()
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{CLIENT_CONFIG["report_title"]} - {latest_date.strftime("%B %Y") if latest_date else "Current"}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Risk Distribution Pie Chart
    risk_counts = current_scores_pd['risk_tier'].value_counts()
    colors = ['#ff4444', '#ffaa00', '#44aa44']  # Red, Orange, Green
    axes[0,0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                  colors=colors, startangle=90)
    axes[0,0].set_title('Overall Risk Distribution')
    
    # 2. Risk Score Distribution Histogram
    axes[0,1].hist(current_scores_pd['risk_score'], bins=30, alpha=0.7, color='steelblue')
    axes[0,1].axvline(current_scores_pd['risk_score'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {current_scores_pd["risk_score"].mean():.3f}')
    axes[0,1].set_xlabel('Risk Score')
    axes[0,1].set_ylabel('Number of Employees')
    axes[0,1].set_title('Risk Score Distribution')
    axes[0,1].legend()
    
    # 3. Risk by Department
    dept_risk = current_scores_pd.groupby('department_clean')['risk_tier'].apply(
        lambda x: (x == 'High Risk').sum() / len(x) * 100
    ).sort_values(ascending=True).tail(10)
    
    axes[1,0].barh(range(len(dept_risk)), dept_risk.values)
    axes[1,0].set_yticks(range(len(dept_risk)))
    axes[1,0].set_yticklabels(dept_risk.index)
    axes[1,0].set_xlabel('High Risk Percentage (%)')
    axes[1,0].set_title('High Risk % by Department (Top 10)')
    
    # 4. Predicted Tenure vs Risk Score Scatter
    scatter = axes[1,1].scatter(current_scores_pd['predicted_survival_days']/365, 
                               current_scores_pd['risk_score'],
                               c=current_scores_pd['salary_clean'], 
                               alpha=0.6, cmap='viridis')
    axes[1,1].set_xlabel('Predicted Tenure (Years)')
    axes[1,1].set_ylabel('Risk Score')
    axes[1,1].set_title('Risk Score vs Predicted Tenure\n(Color = Salary)')
    plt.colorbar(scatter, ax=axes[1,1], label='Salary ($)')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = f"{EXPORT_BASE_PATH}visualizations/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    try:
        plt.savefig(f'/tmp/business_dashboard_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Business visualizations created")
    except Exception as e:
        print(f"⚠️ Could not save visualization: {str(e)}")
    
    plt.close()

# =============================================================================
# 6. MAIN REPORTING PIPELINE
# =============================================================================

def run_business_reporting():
    """Main function to run complete business reporting pipeline"""
    
    print("🚀 Starting Business Reporting Pipeline...")
    start_time = datetime.now()
    
    try:
        # Generate executive summary
        if GENERATE_EXECUTIVE_SUMMARY.lower() == "true":
            exec_summary, exec_metrics = generate_executive_summary()
        else:
            print("⏭️ Skipping executive summary")
            exec_summary, exec_metrics = None, None
        
        # Generate trend analysis
        trend_analysis = generate_trend_analysis()
        
        # Generate detailed reports
        detailed_reports = generate_detailed_reports()
        
        # Create visualizations
        create_business_visualizations()
        
        # Export to CSV
        export_path = export_reports_to_csv()
        
        # Save reporting metadata
        report_metadata = {
            "report_timestamp": datetime.now(),
            "report_type": "monthly_business_report",
            "executive_summary_generated": GENERATE_EXECUTIVE_SUMMARY.lower() == "true",
            "trend_analysis_generated": trend_analysis is not None,
            "csv_exports_generated": export_path is not None,
            "export_path": export_path,
            "total_employees_analyzed": exec_metrics["total_employees"] if exec_metrics else 0,
            "high_risk_employees": exec_metrics["high_risk_count"] if exec_metrics else 0
        }
        
        # Save metadata
        metadata_df = spark.createDataFrame([report_metadata])
        (metadata_df
         .write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "true")
         .saveAsTable(REPORTS_TABLE))
        
        print(f"✅ Report metadata saved to: {REPORTS_TABLE}")
        
        # Pipeline completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("🎉 BUSINESS REPORTING PIPELINE COMPLETED!")
        print("="*80)
        print(f"⏱️ Total execution time: {duration}")
        print(f"📊 Reports generated: {sum([1 for x in [exec_summary, trend_analysis, detailed_reports] if x is not None])}")
        print(f"💾 Export path: {export_path or 'Not generated'}")
        
        if exec_metrics:
            print(f"👥 Employees analyzed: {exec_metrics['total_employees']:,}")
            print(f"🚨 High risk employees: {exec_metrics['high_risk_count']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Business reporting failed: {str(e)}")
        return False

# Run the reporting pipeline
if __name__ == "__main__":
    success = run_business_reporting()
    if not success:
        dbutils.notebook.exit("FAILED")
        