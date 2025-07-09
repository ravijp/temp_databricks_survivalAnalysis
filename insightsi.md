# **Employee Retention Survival Analysis \- Full Story**

## **1\. Baseline: What Normal Retention Looks Like**

**\[Chart: Overall Survival Curve with Milestones\]**

We analyzed 13M employees and found that ADP's client base has a **74% one-year retention rate**. The baseline survival curve shows critical patterns:

* **90-day retention: 90.8%** \- Strong initial retention  
* **180-day retention: 83.9%** \- Steeper decline after 6 months  
* **365-day retention: 74.1%** \- Final annual performance  
* **Median survival: Infinite** \- Less than 50% of employees terminate within a year

**Key Insight:** The steepest decline occurs between days 90-180, indicating a critical intervention window at the 6-month mark.

---

## **2\. Demographics: The Age Crisis and Salary Thresholds**

**\[Chart: Age Segments Survival Curves\]**

The most shocking finding is the **age-based retention crisis**:

* **Under-25 employees: 51% retention** \- Half of young employees leave within a year  
* **25-35 employees: 69.6% retention** \- Still problematic but improving  
* **55-65 employees: 83.6% retention** \- Most stable group  
* **32% performance gap** between youngest and oldest employees

This represents **2M+ young employees at elevated risk** and threatens our future talent pipeline.

**\[Chart: Salary Bands Survival Curves\]**

Salary creates clear retention tiers:

* **Under $40K: 61% retention** \- High-risk compensation level  
* **$40-60K: 77.8% retention** \- Moderate improvement  
* **$120K+: 85.5% retention** \- Stable high-retention threshold  
* **24% improvement gap** from salary optimization

**\[Chart: Tenure Segments Survival Curves\]**

Tenure patterns reveal onboarding effectiveness:

* **Under 6 months: 72.3% retention** \- New hire risk period  
* **6mo-1yr: 80.1% retention** \- Stabilizes after initial period  
* **5+ years: 95% retention** \- Extremely stable long-term employees

**Business Impact:** These three demographic patterns (age, salary, tenure) represent the highest-impact intervention opportunities.

---

## **3\. Industries: Performance Gaps Create Business Opportunities**

**\[Chart: Industry Survival Curves \- Top 10 by Volume\]**

Industry analysis reveals massive performance variations:

**Top Performers:**

* **NAICS 52 (Finance): 82% retention** \- Benchmark performance  
* **NAICS 33 (Manufacturing): 79.6% retention** \- Solid industrial performance  
* **NAICS 32 (Manufacturing): 79.2% retention** \- Consistent manufacturing strength

**Attention Needed:**

* **NAICS 72 (Hospitality): 59.4% retention** \- Major intervention opportunity  
* **NAICS 45 (Retail): 62.3% retention** \- Challenging sector  
* **NAICS 56 (Administrative): 70.3% retention** \- Below-average performance

**Key Finding:** **22.6% performance gap** between best and worst industries represents immediate client success opportunity. Finance sector best practices can be transferred to Hospitality and Retail.

**Business Application:** Industry-specific retention strategies can differentiate ADP's offering and create measurable client value.

---

## **4\. Model Performance: Predictive Power Validation**

**\[Chart: Feature Importance \- XGBoost AFT Model\]**

Our predictive model achieved **70% C-index** (strong survival model performance) with clear feature rankings:

**Top Predictors:**

1. **Salary growth ratio** \- Dominant predictor (compensation trajectory)  
2. **Baseline salary** \- Strong predictor (current compensation level)  
3. **Age** \- Significant predictor (demographic risk factor)  
4. **Manager changes count** \- Moderate predictor (stability indicator)  
5. **Tenure** \- Moderate predictor (experience factor)

**Model Validation:**

* **Training size: 4.3M employees**  
* **Validation size: 1.8M employees**  
* **Cross-validation C-index: 0.705** \- Consistent performance  
* **Feature count: 12** \- Parsimonious model

**Risk Segmentation:** Model identifies **20% high-risk population** for targeted interventions with **80% medium-to-low risk** employees requiring standard retention approaches.

---

## **5\. Temporal Stability: Patterns Are Consistent**

**\[Chart: 2023 vs 2024 Cohort Comparison\]**

Pattern stability validation shows:

* **2023 cohort: 6.2M employees**  
* **2024 cohort: 6.8M employees**  
* **90-day retention change: \+0.0%** \- Identical short-term performance  
* **1-year retention change: \-0.2%** \- Minimal long-term variation

**Business Confidence:** Retention patterns are **stable and predictable**, not seasonal noise. This validates our model for production deployment.

---

## **6\. Next Steps: Voluntary vs Involuntary Analysis (Blair's Guidance)**

**Current Analysis Status:**

* **All-cause termination model**: 25.9% event rate (any termination \= event)  
* **Strong demographic patterns**: Age, salary, industry effects validated  
* **Production-ready baseline**: 70% predictive accuracy achieved

**Blair's Requested Analysis:** We need to implement **competing risk framework** to focus on **voluntary terminations only**:

* **Voluntary terminations**: Event indicator \= 1 (what we can influence)  
* **Involuntary terminations**: Right-censored at termination date (external factors)  
* **Unknown terminations**: Right-censored (conservative approach)

**Expected Outcome:** This will reduce our event rate to approximately **14% voluntary termination rate** but focus on **controllable retention factors**.

**Population Validation:** Generate **SOR-level survival curves** to confirm our populations are homogeneous across systems before proceeding with voluntary analysis.

**Timeline:** 30 days for voluntary vs involuntary framework implementation, 60 days for SOR validation and competing risk models.

---

## **Summary: Key Business Messages**

1. **Age Crisis**: 32% retention gap between young and experienced employees requires immediate intervention  
2. **Salary Thresholds**: Clear compensation tiers create 24% improvement opportunity  
3. **Industry Performance**: 22.6% gap between sectors enables targeted client success strategies  
4. **Predictive Capability**: 70% model accuracy ready for production deployment  
5. **Stable Patterns**: Consistent 2023-2024 performance validates long-term strategy  
6. **Next Phase**: Voluntary termination analysis will focus on controllable retention factors

**Bottom Line:** Survival analysis delivers actionable retention intelligence with immediate business impact potential across 13M employees.

