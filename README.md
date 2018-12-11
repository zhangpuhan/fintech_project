# fintech_project
This is a flask micro website for real time LendingClub loan requests risk prediction with a xgboost model trained on LendingClub 2015 data.

The risk predicted is ranging from 0 to 1. The larger the risk is, the high probability the loan will be charged off.

This model only gives prediction for 36 months terms loans.

For study purpose only.

# Prediction performance
Below is a ROC figure. The ROC is obtained from the training with "1" label for "Charged Off" and "0" for "Fully Paid". The test AUC of ROC reaches 0.71
![alt text](https://github.com/zhangpuhan/fintech_project/blob/master/ROC.png)



The figure below shows the relative importance of the most important 20 features. The importance score is the xgboost model training fscore divided by total fscore for all features.
![alt text](https://github.com/zhangpuhan/fintech_project/blob/master/feature_importance_xgb1210.png)

# Requirements
python 3.5 and above

xgboost

plotly


pandas

numpy

flask

# Run
Add your own api key to /review/views.py line 13 "api_key="

Nevigate to the folder and type **python runsever.py** in terminal
