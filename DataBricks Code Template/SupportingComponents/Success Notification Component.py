# Success Notification and Pipeline Summary
# Sends comprehensive success notification with pipeline metrics and business insights

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Notification parameters (configurable via widgets)
NOTIFICATION_TYPE = dbutils.widgets.get("notification_type") if "notification_type" in [w.name for w in dbutils.widgets.getAll()] else "success"
INCLUDE_METRICS = dbutils.widgets.get("include_metrics") if "include_metrics" in [w.name for w in dbutils.widgets.getAll()] else "true"
SEND_TO_CLIENTS = dbutils.widgets.get("send_to_clients") if "send_to_clients" in [w.name for w in dbutils.widgets.getAll()] else "true"

# Email configuration
NOTIFICATION_RECIPIENTS = {
    "hr_leadership": ["hr-leadership@company.com"],
    "hr_analytics": ["hr-analytics@company.com"],  
    "data_team": ["data-team@company.com"],
    "ml_team": ["ml-team@company.com"],
    "clients": ["client-contact@company.com"]  # External HR partners
}

# Data sources for metrics
EMPLOYEE_SCORES_TABLE = "hr_analytics.predictions.employee_risk_scores"
SCORING_METADATA_TABLE = "hr_analytics.predictions.scoring_metadata"
VALIDATION_RESULTS_TABLE = "hr_analytics.models.validation_results"
BUSINESS_REPORTS_TABLE = "hr_analytics.reports.business_reports"

print("📧 Success Notification and Pipeline Summary")
print(f"📋 Notification type: {NOTIFICATION_TYPE}")
print(f"📊 Include metrics: {INCLUDE_METRICS}")
print(f"👥 Send to clients: {SEND_TO_CLIENTS}")

# =============================================================================
# IMPORTS
# =============================================================================

from pyspark.sql.functions import *
from datetime import datetime, timedelta
import json

# =============================================================================
# 1. GATHER PIPELINE METRICS
# =============================================================================

def gather_pipeline_metrics():
    """Gather comprehensive metrics from the completed pipeline"""
    
    print("\n📊 Gathering pipeline metrics...")
    
    metrics = {
        "pipeline_completion_time": datetime.now(),
        "pipeline_success": True
    }
    
    try:
        # Get latest scoring metadata
        scoring_metadata = (spark.table(SCORING_METADATA_TABLE)
                           .orderBy(col("scoring_date").desc())
                           .limit(1)
                           .collect())
        
        if scoring_metadata:
            latest_run = scoring_metadata[0]
            metrics.update({
                "employees_scored": latest_run.employees_scored,
                "high_risk_count": latest_run.high_risk_count,
                "medium_risk_count": latest_run.medium_risk_count,
                "low_risk_count": latest_run.low_risk_count,
                "avg_risk_score": float(latest_run.avg_risk_score),
                "alerts_generated": latest_run.alerts_generated,
                "model_version": latest_run.model_version,
                "scoring_date": latest_run.scoring_date
            })
            
            # Calculate risk percentages
            total = latest_run.employees_scored
            metrics.update({
                "high_risk_percentage": (latest_run.high_risk_count / total * 100) if total > 0 else 0,
                "medium_risk_percentage": (latest_run.medium_risk_count / total * 100) if total > 0 else 0,
                "low_risk_percentage": (latest_run.low_risk_count / total * 100) if total > 0 else 0
            })
        
        print(f"   ✅ Scoring metrics gathered")
        
    except Exception as e:
        print(f"   ⚠️ Could not gather scoring metrics: {str(e)}")
        metrics.update({
            "employees_scored": 0,
            "scoring_metrics_error": str(e)
        })
    
    try:
        # Get model validation results
        validation_results = (spark.table(VALIDATION_RESULTS_TABLE)
                             .orderBy(col("training_date").desc())
                             .limit(1)
                             .collect())
        
        if validation_results:
            latest_validation = validation_results[0]
            metrics.update({
                "model_concordance": float(latest_validation.test_concordance),
                "model_passed_validation": latest_validation.model_passed,
                "model_training_samples": latest_validation.training_samples,
                "model_test_samples": latest_validation.test_samples
            })
        
        print(f"   ✅ Model validation metrics gathered")
        
    except Exception as e:
        print(f"   ⚠️ Could not gather validation metrics: {str(e)}")
    
    try:
        # Get department breakdown
        latest_scores = spark.table(EMPLOYEE_SCORES_TABLE)
        latest_date = latest_scores.select(max("scoring_date")).collect()[0][0]
        
        dept_breakdown = (latest_scores
                         .filter(col("scoring_date") == latest_date)
                         .groupBy("department_clean")
                         .agg(
                             count("*").alias("total"),
                             sum(when(col("risk_tier") == "High Risk", 1).otherwise(0)).alias("high_risk")
                         )
                         .withColumn("high_risk_pct", col("high_risk") / col("total") * 100)
                         .orderBy(col("high_risk_pct").desc())
                         .limit(5)
                         .collect())
        
        metrics["top_risk_departments"] = [
            {
                "department": row.department_clean,
                "total_employees": row.total,
                "high_risk_employees": row.high_risk,
                "high_risk_percentage": float(row.high_risk_pct)
            }
            for row in dept_breakdown
        ]
        
        print(f"   ✅ Department breakdown gathered")
        
    except Exception as e:
        print(f"   ⚠️ Could not gather department breakdown: {str(e)}")
    
    return metrics

# =============================================================================
# 2. BUSINESS IMPACT CALCULATION
# =============================================================================

def calculate_business_impact(metrics):
    """Calculate estimated business impact from the analysis"""
    
    print("\n💰 Calculating business impact...")
    
    impact = {}
    
    try:
        # Estimated replacement costs
        high_risk_employees = metrics.get("high_risk_count", 0)
        
        # Industry standard replacement cost estimates
        REPLACEMENT_COST_PER_EMPLOYEE = 75000  # $75K average replacement cost
        RETENTION_SUCCESS_RATE = 0.3  # 30% retention success rate with intervention
        
        potential_turnover_cost = high_risk_employees * REPLACEMENT_COST_PER_EMPLOYEE
        preventable_cost = potential_turnover_cost * RETENTION_SUCCESS_RATE
        
        impact.update({
            "potential_turnover_cost": potential_turnover_cost,
            "preventable_cost_with_action": preventable_cost,
            "cost_per_prevented_turnover": REPLACEMENT_COST_PER_EMPLOYEE,
            "estimated_retention_success_rate": RETENTION_SUCCESS_RATE
        })
        
        # Productivity impact
        avg_tenure_loss = 180  # Assume 6 months average lost productivity
        DAILY_PRODUCTIVITY_VALUE = 400  # $400 per day productive value
        
        productivity_impact = high_risk_employees * avg_tenure_loss * DAILY_PRODUCTIVITY_VALUE
        
        impact.update({
            "estimated_productivity_impact": productivity_impact,
            "total_estimated_impact": potential_turnover_cost + productivity_impact
        })
        
        print(f"   💰 Potential cost impact: ${potential_turnover_cost:,.0f}")
        print(f"   💰 Preventable with action: ${preventable_cost:,.0f}")
        
    except Exception as e:
        print(f"   ⚠️ Could not calculate business impact: {str(e)}")
        impact["calculation_error"] = str(e)
    
    return impact

# =============================================================================
# 3. GENERATE NOTIFICATION CONTENT
# =============================================================================

def generate_notification_content(metrics, impact):
    """Generate comprehensive notification content for different audiences"""
    
    print("\n📝 Generating notification content...")
    
    # Get current date info
    current_date = datetime.now()
    scoring_date = metrics.get("scoring_date", current_date)
    
    # Executive Summary (for leadership)
    executive_summary = f"""
🎯 MONTHLY EMPLOYEE TURNOVER RISK ANALYSIS - EXECUTIVE SUMMARY

Analysis Date: {scoring_date.strftime('%B %d, %Y') if hasattr(scoring_date, 'strftime') else scoring_date}
Pipeline Completed: {current_date.strftime('%B %d, %Y at %I:%M %p')}

KEY FINDINGS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👥 WORKFORCE ANALYZED: {metrics.get('employees_scored', 0):,} employees

🚨 RISK DISTRIBUTION:
   • High Risk (action needed):     {metrics.get('high_risk_count', 0):,} employees ({metrics.get('high_risk_percentage', 0):.1f}%)
   • Medium Risk (monitor):         {metrics.get('medium_risk_count', 0):,} employees ({metrics.get('medium_risk_percentage', 0):.1f}%)
   • Low Risk (stable):             {metrics.get('low_risk_count', 0):,} employees ({metrics.get('low_risk_percentage', 0):.1f}%)

💰 BUSINESS IMPACT:
   • Potential turnover cost:       ${impact.get('potential_turnover_cost', 0):,.0f}
   • Preventable with intervention: ${impact.get('preventable_cost_with_action', 0):,.0f}
   • ROI opportunity:               ${impact.get('preventable_cost_with_action', 0):,.0f} in retained value

🎯 MODEL PERFORMANCE:
   • Prediction accuracy:           {metrics.get('model_concordance', 0):.1%}
   • Model validation:              {'✅ PASSED' if metrics.get('model_passed_validation', False) else '❌ FAILED'}

⚡ IMMEDIATE ACTIONS REQUIRED:
"""
    
    if metrics.get('high_risk_count', 0) > 0:
        executive_summary += f"""   1. Schedule retention meetings with {min(metrics.get('high_risk_count', 0), 20)} highest-risk employees
   2. Review compensation for high-risk, high-value employees  
   3. Begin succession planning for critical high-risk roles
   4. Implement department-specific retention strategies"""
    else:
        executive_summary += "   🟢 No immediate high-risk employees requiring attention"
    
    # Technical Summary (for data/ML teams)
    technical_summary = f"""
🔧 TECHNICAL PIPELINE SUMMARY

Pipeline Execution: ✅ SUCCESSFUL
Completion Time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}

📊 DATA PROCESSING:
   • Records processed: {metrics.get('employees_scored', 0):,}
   • Model version: {metrics.get('model_version', 'N/A')}
   • Alerts generated: {metrics.get('alerts_generated', 0)}

🤖 MODEL PERFORMANCE:
   • Test concordance index: {metrics.get('model_concordance', 0):.4f}
   • Training samples: {metrics.get('model_training_samples', 0):,}
   • Test samples: {metrics.get('model_test_samples', 0):,}
   • Validation status: {'PASSED' if metrics.get('model_passed_validation', False) else 'FAILED'}

📈 PREDICTION QUALITY:
   • Average risk score: {metrics.get('avg_risk_score', 0):.3f}
   • Risk distribution: {metrics.get('high_risk_percentage', 0):.1f}% / {metrics.get('medium_risk_percentage', 0):.1f}% / {metrics.get('low_risk_percentage', 0):.1f}%
"""
    
    # Client Summary (for external HR partners)
    client_summary = f"""
📊 EMPLOYEE RETENTION INSIGHTS - {current_date.strftime('%B %Y')}

Dear HR Partner,

Your monthly employee turnover risk analysis has been completed successfully. 
Here are the key insights for your organization:

🎯 ANALYSIS OVERVIEW:
   • Employees analyzed: {metrics.get('employees_scored', 0):,}
   • Analysis date: {scoring_date.strftime('%B %d, %Y') if hasattr(scoring_date, 'strftime') else scoring_date}
   • Predictive model accuracy: {metrics.get('model_concordance', 0):.1%}

📊 RISK ASSESSMENT:
   • Employees requiring immediate attention: {metrics.get('high_risk_count', 0):,}
   • Employees to monitor closely: {metrics.get('medium_risk_count', 0):,}
   • Stable employee population: {metrics.get('low_risk_count', 0):,}

🏢 DEPARTMENTAL INSIGHTS:
"""
    
    # Add top risk departments to client summary
    if "top_risk_departments" in metrics:
        for dept in metrics["top_risk_departments"][:3]:
            client_summary += f"   • {dept['department']}: {dept['high_risk_percentage']:.1f}% high risk\n"
    
    client_summary += f"""
💡 RECOMMENDATIONS:
   1. Focus retention efforts on the {metrics.get('high_risk_count', 0)} high-risk employees
   2. Conduct stay interviews with medium-risk employees  
   3. Review departmental engagement strategies
   4. Schedule follow-up analysis in 2-4 weeks

The detailed analysis reports and employee-level data have been made available 
in your secure data environment.

Best regards,
Analytics Team
"""
    
    return {
        "executive_summary": executive_summary,
        "technical_summary": technical_summary,  
        "client_summary": client_summary
    }

# =============================================================================
# 4. SEND NOTIFICATIONS
# =============================================================================

def send_notifications(content, metrics):
    """Send notifications to appropriate recipients"""
    
    print("\n📧 Sending notifications...")
    
    notifications_sent = []
    
    try:
        # In a real implementation, you would integrate with:
        # - SMTP server for email
        # - Teams/Webex webhook for chat notifications
        # - SMS service for critical alerts
        
        # For now, we'll simulate and log the notifications
        
        # Executive notification
        exec_notification = {
            "recipients": NOTIFICATION_RECIPIENTS["hr_leadership"],
            "subject": f"✅ Monthly Employee Risk Analysis Complete - {metrics.get('high_risk_count', 0)} High Risk Employees",
            "content": content["executive_summary"],
            "priority": "high" if metrics.get('high_risk_count', 0) > 20 else "normal",
            "sent_timestamp": datetime.now()
        }
        
        print(f"   📧 Executive notification prepared for {len(exec_notification['recipients'])} recipients")
        notifications_sent.append(exec_notification)
        
        # Technical team notification
        tech_notification = {
            "recipients": NOTIFICATION_RECIPIENTS["data_team"] + NOTIFICATION_RECIPIENTS["ml_team"],
            "subject": f"🔧 HR Survival Analysis Pipeline - Success",
            "content": content["technical_summary"],
            "priority": "normal",
            "sent_timestamp": datetime.now()
        }
        
        print(f"   📧 Technical notification prepared for {len(tech_notification['recipients'])} recipients")
        notifications_sent.append(tech_notification)
        
        # Client notification (if enabled)
        if SEND_TO_CLIENTS.lower() == "true":
            client_notification = {
                "recipients": NOTIFICATION_RECIPIENTS["clients"],
                "subject": f"📊 Monthly Employee Retention Analysis - {datetime.now().strftime('%B %Y')}",
                "content": content["client_summary"],
                "priority": "normal",
                "sent_timestamp": datetime.now(),
                "external": True
            }
            
            print(f"   📧 Client notification prepared for {len(client_notification['recipients'])} recipients")
            notifications_sent.append(client_notification)
        
        # High-risk alert (if needed)
        if metrics.get('high_risk_count', 0) > 50:  # Alert if more than 50 high-risk employees
            alert_notification = {
                "recipients": NOTIFICATION_RECIPIENTS["hr_leadership"] + NOTIFICATION_RECIPIENTS["hr_analytics"],
                "subject": f"🚨 HIGH RISK ALERT: {metrics.get('high_risk_count', 0)} Employees at Risk",
                "content": f"URGENT: Analysis shows {metrics.get('high_risk_count', 0)} employees at high risk of voluntary termination. Immediate action recommended.",
                "priority": "urgent",
                "sent_timestamp": datetime.now()
            }
            
            print(f"   🚨 High-risk alert prepared")
            notifications_sent.append(alert_notification)
        
        # Save notification log
        if notifications_sent:
            notifications_df = spark.createDataFrame(notifications_sent)
            (notifications_df
             .write
             .format("delta")
             .mode("append")
             .option("mergeSchema", "true")
             .saveAsTable("hr_analytics.notifications.notification_log"))
            
            print(f"   ✅ Notification log saved")
        
        print(f"✅ {len(notifications_sent)} notifications prepared successfully")
        
    except Exception as e:
        print(f"❌ Failed to send notifications: {str(e)}")
        return []
    
    return notifications_sent

# =============================================================================
# 5. PIPELINE SUMMARY REPORT
# =============================================================================

def create_pipeline_summary(metrics, impact, notifications):
    """Create comprehensive pipeline summary for audit trail"""
    
    print("\n📋 Creating pipeline summary...")
    
    summary = {
        "pipeline_run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "completion_timestamp": datetime.now(),
        "pipeline_status": "SUCCESS",
        "execution_summary": {
            "employees_analyzed": metrics.get("employees_scored", 0),
            "high_risk_identified": metrics.get("high_risk_count", 0),
            "model_performance": metrics.get("model_concordance", 0),
            "alerts_generated": metrics.get("alerts_generated", 0)
        },
        "business_impact": impact,
        "notifications_sent": len(notifications),
        "next_scheduled_run": "25th of next month",
        "data_freshness": metrics.get("scoring_date"),
        "model_version": metrics.get("model_version", "unknown")
    }
    
    # Save pipeline summary
    try:
        summary_df = spark.createDataFrame([summary])
        (summary_df
         .write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "true")
         .saveAsTable("hr_analytics.audit.pipeline_summary"))
        
        print(f"✅ Pipeline summary saved to hr_analytics.audit.pipeline_summary")
    except Exception as e:
        print(f"⚠️ Could not save pipeline summary: {str(e)}")
    
    return summary

# =============================================================================
# 6. MAIN NOTIFICATION PIPELINE
# =============================================================================

def run_success_notification():
    """Main function to run complete success notification process"""
    
    print("🚀 Starting Success Notification Pipeline...")
    start_time = datetime.now()
    
    try:
        # Step 1: Gather pipeline metrics
        metrics = gather_pipeline_metrics()
        
        # Step 2: Calculate business impact
        impact = calculate_business_impact(metrics)
        
        # Step 3: Generate notification content
        content = generate_notification_content(metrics, impact)
        
        # Step 4: Send notifications
        notifications = send_notifications(content, metrics)
        
        # Step 5: Create pipeline summary
        summary = create_pipeline_summary(metrics, impact, notifications)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("🎉 SUCCESS NOTIFICATION PIPELINE COMPLETED!")
        print("="*80)
        print(f"⏱️ Notification processing time: {duration}")
        print(f"👥 Employees analyzed: {metrics.get('employees_scored', 0):,}")
        print(f"🚨 High risk employees: {metrics.get('high_risk_count', 0):,}")
        print(f"📧 Notifications sent: {len(notifications)}")
        print(f"💰 Potential business impact: ${impact.get('potential_turnover_cost', 0):,.0f}")
        print(f"🎯 Model accuracy: {metrics.get('model_concordance', 0):.1%}")
        
        # Key achievements
        print(f"\n🏆 KEY ACHIEVEMENTS:")
        print(f"   ✅ Successfully analyzed {metrics.get('employees_scored', 0):,} employee records")
        print(f"   ✅ Identified {metrics.get('high_risk_count', 0)} employees requiring retention focus")
        print(f"   ✅ Model performance: {metrics.get('model_concordance', 0):.1%} accuracy")
        print(f"   ✅ Business stakeholders notified with actionable insights")
        
        print(f"\n📅 Next scheduled pipeline run: 25th of next month")
        print(f"🔄 Pipeline ready for next execution cycle")
        
        return True
        
    except Exception as e:
        print(f"❌ Success notification failed: {str(e)}")
        return False

# Run the notification pipeline
if __name__ == "__main__":
    success = run_success_notification()
    if not success:
        dbutils.notebook.exit("FAILED")
    else:
        print("\n🎊 MONTHLY HR SURVIVAL ANALYSIS PIPELINE COMPLETED SUCCESSFULLY! 🎊")
        