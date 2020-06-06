# Analyzing-Black-Friday-Sales

We analyzed the Black Friday Sales data for our IST 707: Data Analytics course. The Black Friday data was obtained from Kaggle. The data consisted of **538k rows and 12 attributes**. A retail store can use our analysis and know which products to stock before the holiday season so that the store doesn’t run out of products. We analyzed customer purchase behavior against different products.

**Data PreProcessing:** We analyzed data extensively and checked all the missing values and replaced some of them with means of the entire column. We also replaced data having values like 0. Performed feature engineering on data and removed some columns based on domain knowledge. We also analyzed outliers in some columns and removed them.

**Descriptive Analysis:** To analyze multiple things and compare relations between columns, we performed a descriptive analysis of the data. We answered various business questions using visualizations and statistical analysis. 

**Linear Regression:** For the regression model, we started by creating dummy variables for some of the categorical variables. We build multiple regression models by choosing variables based on p-values. We analyzed spending patterns using our model. For example, Age doesn’t affect the spending pattern; Every year leads to a decrease in Spendings.

**Random Forest Regressor:** We build a Random Forest Regressor to predict the sales using different factors. For improving the analysis, we tuned our model using hyperparameters like the Number of trees, Maximum data points, and Pruning. We divided the data into training and testing data and tested our model on the testing data. 80% of the data as training and the remaining 20% was testing.

**Random Forest Classifier:** This model works the same way as Random Forest Regressor. We obtained an accuracy of 60% for this model.

**Naive Bayes:** This model uses Bayes theorem of probability to predict the class of an unknown data set. We implemented this model on our Sales data. We obtained an accuracy of 58% for this model.

**Support Vector Machine:** We implement the SVM algorithm too on the dataset. We used 5 bins for this model and got an accuracy of 54%.

**Conclusion:** We compared each model based on the accuracy and time taken. The *Random Forest Classifier* was our best model, it learned the patterns quickly in the data set and gave us the highest accuracy.

**Programming:** Python
**Libraries:** Numpy, Pandas, Sklearn, Matplotlib
