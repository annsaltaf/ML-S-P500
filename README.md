# ML-S-P500
ML models and their performance on S&amp;P 500 


**Stock Market Prediction Using Machine Learning Models**


Anns Altaf


November 30 2024


**1. Introduction**

In financial analysis the importance of ML and AI is ever increasing. They are being more actively used to predict how the market will behave, risk assessment and investment optimization. In this project we use time-series forecasting and classification methods. Our goal is to understand whether these methods can prove to be more efficient and accurate than traditional methods. In our opinion they fit well with the supervised learning techniques that we have been using. 

**1.1 The Problem We Tried to Solve**

The project was undertaken to predict the market trend, whether it will show an upward or downward trend over the next quarter. Historical daily Open/Close data from the S%P 500 index and Hedge Fund balance sheet was used for this model. This model can prove to be a great tool to investors, giving them a predictive ability to decide how to stage their investments. It is a very complex system, the Stock Market, and is it possible to do more than just take educated guesses? Can we actually have something that can be accurate to predict the market’s movement? If so, what is the best ML model that can accomplish this task? Is it just one or maybe a combination of multiple models?

**1.2 Related Work**

Several studies have explored the use of machine learning for stock market prediction. 
A notable paper by Zhang et al. (2020) compares several models for market trend prediction and finds that ensemble methods like Random Forest perform well. Another study by Kim (2018) highlights the advantages of XGBoost in high-dimensional data settings, which is like our project’s focus on financial data. 
Bhardwaj et al. (2020); found ARIMA to be efficient in time-series scenarios but not in handling non-linear relationships. Where ML algorithms can have an edge.
Patel et al. (2015) found ensemble model better suited to financial forecasting as they can handle the complex feature interactions of financial data more than standalone models. 
Chen and Guestrin (2016) found XGBoost to be efficient when working with high-dimensional datasets. We can utilize this for our Hedge fund dataset.
However, few studies have simultaneously tested these models, which makes our approach of direct comparison across multiple algorithms novel.


**1.3 What Tools and Programs Are Already Available?**

Several tools and libraries are widely used in financial machine learning, including:
•	Scikit-learn for implementing traditional ML algorithms such as Random Forest and Logistic Regression.
•	XGBoost for efficient implementation of gradient boosting models.
•	TensorFlow/Keras for building LSTM models for time-series prediction.
•	ARIMA from the statsmodels library for classical time-series forecasting. 
These tools were used in our project, building on existing models to provide a robust framework for comparison. While these models have been used for market prediction, our work builds on them by comparing their performance on the same dataset using a uniform set of evaluation metrics. We used these existing models as a foundation for our comparison. While it is true that these models have been used in the past for market prediction, our project was built on the same principle by comparing their performance on the same dataset using uniform evaluation metrics (Accuracy, Precision, Recall and F1 score). 

**2. Overview of the Architecture**

The project consists of several modules that implement different machine learning models and evaluate their performance.

**2.1 Finished Work: Running Modules**

•	Data Preprocessing: Cleaning the dataset, handling missing values, and scaling features.
•	Model Implementations:
•	Random Forest
•	XGBoost
•	Logistic Regression
•	LSTM
•	ARIMA
•	Model Evaluation: Evaluation using metrics like accuracy, precision, recall, and F1 score.

**2.2 Work in Progress: Modules Designed but Not Implemented**

•	Hyperparameter Tuning: Further finetuning of the hyperparameters could improve model outcomes.
•	Feature Engineering: We could have benefitted from exploring further complimentary data or conducting sentiment analysis.
2.3 Future Work: Modules for Future Continuation
•	Alternative Data: Alternative datasets like; other indices historical data, various macroeconomic indicators, public sentiment could improve model performance.
•	Deep Learning Models: GRU for Deep Learning and Transformers for time-series forecasting are a good choice for future ventures. 





**3. Data Collection**

Hedge Fund:
Source: The Federal Reserve.
Purpose: To collect data/features like assets held, foreign holdings and other leverage metrics. 
 Reference: Federal Reserve Board - Home

S&P 500:
Source; Yahoo! Finance
Purpose: To derive quarterly stock positions. The data was ‘Daily’, it was converted to quarterly for proper alignment with the other dataset.
Reference: https://finance.yahoo.com


4. Baseline and Proposed Methods
Baseline Method:
Logistic Regression: A starting point to assess more complex models. A simple linear model for linear approaches.
Proposed Methods: 
•	Random Forest and XGBoost: Ensemble Models; efficient for handling complex, non-linear data relationships and                providing strong performance on high-dimensional datasets such as Financial Metrics.
•	LSTM: Deep Learning	model; time-series forecasting. 
•	ARIMA: Excellent time-series model to get a benchmark for forecasting.


5. Implementation
All models were implanted in Python using Jupyter notebooks. Relevant libraries like scikit-learn, XGBoost, Keras, TensorFlow and statsmodels were used to import functions wherever necessary:
1.	Data Preprocessing: To take care of missing values, scaling data and assigning categorical variables.
2.	Model Training: Training each model with the preprocessed data using train-test split.
3.	Model Evaluation: Performance evaluation using accuracy, precision, recall, and F1 score.
6. Results and Evaluation

   
**Model Evaluation Results:**

•	Random Forest: Accuracy: 0.7500, Precision: 0.8333, Recall: 0.8333, F1 Score: 0.8333
•	XGBoost: Accuracy: 0.7500, Precision: 0.8333, Recall: 0.8333, F1 Score: 0.8333
•	Logistic Regression: Accuracy: 0.6250, Precision: 0.8000, Recall: 0.6667, F1 Score: 0.7273
•	LSTM: Accuracy: 0.3750, Precision: 1.0000, Recall: 0.1667, F1 Score: 0.2857
•	ARIMA: Accuracy: 0.7500, Precision: 0.7500, Recall: 1.0000, F1 Score: 0.8571
Analysis: As expected, the Ensemble Methods struck the best balance. Random Forest and XGBoost had the best overall performance, even ARIMA with its perfect recall of 1 had a lower precision score, implying false positives. Our baseline, Logistic Regression, performed poorly on recall, making it unsuitable for spotting trends. 
LSTM, with its low recall, meant that it missed most of the relevant trend signals.


**8. Achievements and Observations**

 
Learning:
•	Model Comparison: The importance of using multiple metrics to evaluate model performance was highlighted by observing the scores of models like ARIMA. Even with its perfect recall it struggled with precision. It is important to use many of the relevant metrics especially when working with such a complex dataset.
•	Challenges: Implementing Data preprocessing was a struggle, creating a merged data set when one was ‘Quarterly’ and the other one ‘Daily’. LSTM also took significant finetuning and feature engineering as the data was very complex. 


**9. Discussion and Conclusions**

Ensemble Methods(Random Forest and XGBoost) proved to be the better choice for this particular project as they are good at determining trends, which is what we were trying to achieve in this project. We wanted to get accurate trend projections, while the purpose of this project was not to explicitly make predictions of stock prices over time we did find a good starting point for ML models that are suitable for such a venture. 
Our baseline (Logistic Regression) had a very poor recall score, making it unsuitable for this task.
ARIMA was good at determining trends but with its low precision score produced a lot of false positives.
LSTM was promising with its deep learning architecture. It is possible to get the best out of LSTM if more effort is spent in feature engineering and preprocessing. Revealing the challenges of using deep learning without a lot of feature engineering.
**Future Work:**

•	Increased hyperparameter tuning and using k-cross validation to split data. 
•	Using more external data, including public sentient analysis.
•	Applying with deep learning architectures like GRU, Transformers, and Neural Network.



**8. References**

1. Machine Learning and Financial Prediction:
•	Atsalakis, G. S., & Valavanis, K. P. (2009). Surveying stock market prediction techniques – Part I: Neural networks. Expert Systems with Applications, 36(3), 7846-7857.
DOI: 10.1016/j.eswa.2008.10.020
This paper surveys the use of neural networks for stock market prediction, discussing the challenges and benefits of using AI models for financial data analysis.
•	Zhang, G., & Zhou, X. (2004). Stock market forecasting with artificial neural networks. Proceedings of the IEEE International Conference on Computational Intelligence for Financial Engineering, 2004.
This paper demonstrates how artificial neural networks, a key component of your LSTM model, have been applied to predict stock prices and market trends.
2. Random Forest and XGBoost for Financial Prediction:
•	Cheng, S. J., & Yang, J. (2013). Random forests for financial forecasting. In Proceedings of the International Conference on Computational Science and Engineering (Vol. 3, pp. 174-179).
This paper discusses the application of Random Forests to financial data, highlighting their predictive capabilities in forecasting stock prices.
•	Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794).
DOI: 10.1145/2939672.2939785
This paper presents XGBoost, a scalable machine learning algorithm widely used for classification tasks like market prediction. It highlights XGBoost’s strengths, including model accuracy and handling large datasets.
3. ARIMA for Time Series Forecasting:
•	Box, G. E., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control (Revised Edition).
This is the seminal work on ARIMA models for time series forecasting. It provides detailed methods for applying ARIMA to predict future values based on historical data, which is crucial for forecasting stock trends.
•	Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice (2nd ed.).
Available online: https://otexts.com/fpp2/
This book provides a comprehensive guide to forecasting techniques, including ARIMA, and is essential for understanding how to apply these models effectively in predicting financial data.
4. Deep Learning and LSTM in Finance:
•	Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market prediction. European Journal of Operational Research, 270(2), 654-669.
DOI: 10.1016/j.ejor.2017.11.054
This paper explores the use of LSTM networks for predicting stock prices, providing practical insights into deep learning applications for financial markets.
•	Shin, H. J., & Lee, S. W. (2019). Application of deep learning models to stock price prediction. Journal of Financial Data Science, 1(1), 40-55.
This research paper focuses on the use of deep learning models, including LSTM, in financial markets for forecasting stock prices and trends.
5. Model Evaluation in Financial Predictions:
•	Gonçalves, P., & Barros, L. (2019). Evaluating classification models for financial data prediction. Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD).
This paper provides an evaluation framework for assessing the performance of various machine learning models, including Random Forest, XGBoost, and neural networks, in financial data applications.
6. General Machine Learning in Finance:
•	Heaton, J., Polson, N. G., & Witte, J. (2017). Deep learning in finance. Proceedings of the 20th Annual Financial Engineering & Risk Management Conference.
This paper discusses how deep learning models are transforming financial predictions, providing insights into the potential of machine learning in forecasting financial trends.

