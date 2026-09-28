# Modeling-and-Understanding-Nurse-Stress-Using-Wearable-Physiological-Sensor-Data
This project aims to analyze and model physiological stress responses in nurses using wearable sensor data, including electrodermal activity (EDA), heart rate, skin temperature (TEMP), and motion signals. Each data point is labeled with a stress level: 0(low/no stress), 1(moderate stress), and 2(high stress). 
These labeled observations support supervised learning for stress prediction and allow for detailed analysis of temporal stress patterns and individual differences.

DATA: Kaggle -> merged_data.csv(861.07 MB)
https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors

TOOLS: Python (pandas,numpy, matplotlib, seaborn, sklearn (model selection, ensemble, metrics))

MODELS: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier

GOALS:
1) Predictive modeling: develop and evaluate machine learning models that classify stress levels of nurses based on physiological signals from wearable sensors.
2) Stress pattern analysis: Explore temporal patterns, individual differences, and the relative
importance of physiological signals in relation to stress levels.

CHALLANGES:
- Wearable data contained irregular sampling patterns across nurses: some nurses recorded data continuously, whereas others had sparse or intermittent recordings. The dataset was resampled to 30 second intervals, grouped by nurse ID, to standardize temporal spacing across all participants. However, due to large gaps between recording days for some nurses, resampling introduced extensive missing values corresponding to time ranges where no measurements were taken.

RESULS:

Random Forest performed strongly even without oversampling, this shows that tree based ensemble methods naturally handle noise and feature interactions in physiological data. However, after applying SMOTE, XGBoost and LightGBM performed as good as Random Forest, indicating that gradient boosting methods benefit more from balanced training data.
