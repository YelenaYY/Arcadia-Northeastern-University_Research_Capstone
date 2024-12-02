# Deep Learning for Heart Failure Prediction
Arcadia-Northeastern University Research_Capstone
## Research focused on comparing various machine learning models for heart failure prediction using real-world healthcare data.


__Abstract__

Heart failure (HF) is a major public health concern with high mortality and healthcare costs, emphasizing the importance of accurate early prediction. This research explores the application of deep learning models, specifically Recurrent Neural Networks with Long Short-Term Memory (RNN-LSTM), Gated Recurrent Units (RNN-GRU), and Transformer architectures to predict heart failure risk using patient time-series data. The goal is to reduce morbidity and improve patient outcomes.

This research utilized real datasets from [Arcadia.io](arcadia.io). The key features are listed as below:
* Health measurements/Vital Signs: Body Mass Index (BMI), Systolic and Diastolic Blood Pressure, Temperature, Heart Rate, Oxygen Saturation (SpO2) and Mean Arterial Pressure(MAP).
* Demographic information: Age, gender, and socioeconomic factors, including the percentage of unemployed individuals (pct_unemployed), the percentage of people living in poverty (pct_poverty), the percentage of households without cars (pct_no_car), and the percentage of single-parent households (pct_single_parent) in geographical areas.
* Health incidents: Presence of heart failure in outpatient records and the total number of days spent in the hospital for inpatient records.


## Data Preparations:
All vital sign-related features were used to generate derived features, including means, standard deviations, and counts. These features were then processed into a time-series matrix spanning 12 intervals at 3-month intervals over the past 3 years. Demographic information, in contrast, was handled separately as non-sequential data.

The objective of this study was to predict the heart failure risk for the following year. The output was structured as accumulated time periods, representing heart failure probabilities within 0–1 month, 0–3 months, 0–6 months, and 0–12 months, aligning with healthcare industry standards.

__Data Types__
```
target_cols = ['heart_failure_1mo', 'heart_failure_3mo', 'heart_failure_6mo', 'heart_failure_12mo']
binary_cols = ['male', 'has_heart_failure_outpatient']
```
The outcomes in the target columns and features in the binary columns are of binary data type (0/1). All other features are numeric and were normalized using z-score normalization before being fed into the model.


## Evaluation Metrix
Area Under the Curve (AUC) is used to evaluate each model's predictive performance. By analyzing the strengths and limitations of RNN-LSTM, RNN-GRU, and Transformer models, this research aims to identify the most effective deep learning architecture for early heart failure (HF) diagnosis.

Binary Cross-Entropy (BCE) is used as the loss function to optimize the model during training by measuring the difference between predicted probabilities and actual binary outcomes. This ensures the models are fine-tuned to minimize prediction errors and enhance their ability to differentiate between positive and negative cases.


## Data Resource

All Data used in this research is from Arcadia.io, including patient healthcare records and demographic information.

__Data Privacy and Access__

Due to privacy considerations, all research data related to this heart failure study, sourced from Arcadia.io, will remain confidential and cannot be disclosed.

__Contact__
* **Yelena Y. Yu <yu.yue16@northeastern.edu>** - Contact for inquires regarding the model and paper.
* **Samir Farooq<samir.farooq@arcadia.io>** - Contact for inquiries regarding data resources and company-related information.
