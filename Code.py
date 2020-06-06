import numpy as np
import pandas as pd
#Cause 'seaborn' is based on matplotlib:
import matplotlib.pyplot as mplt
#For statistical data visualization:
#import seaborn as sns
#All the following functionalities have been imported for machine learning, data splitting, accuracy testing:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
#For scaling the unscaled data: (to avoid biasing?):
from sklearn.preprocessing import StandardScaler
#I can probably manually encode, I am just bored...:
from sklearn.preprocessing import LabelEncoder
#for Random Forest in a regression approach:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
#For random forest classification:
from sklearn.ensemble import RandomForestClassifier
#Calculating the accuracy of Classifier:
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def outlier_test(inp):
    #I am using the condiiton of absolute value of z-score>3 as the parameter of being an outlier:
    threshold=3
    avg_inp=np.mean(inp)
    stdev=np.std(inp)
    zscores=[]
    for i in inp:
        z=(i-avg_inp)/stdev
        zscores.append(z)
    return np.where(np.abs(zscores) > threshold)

#Reading the data:
df = pd.read_csv("BlackFriday.csv")
print(df.head())

#info will print the number of entries (non-null) and their data type. Describe gives statistical description:
print(df.info())
print(df.describe())

#fillna isn't wokring. Either that or taking just too long:
#Any other methods/suggestions to fill the 'na's?

#df=df.fillna(df.mean()['Product_Category_1':'Product_Category_2'])
#print(df.head())

#To see columnwise total number of missing data points:
print(df.isnull().sum())

#Imputing the missing values with mean to keep column mean unchanged:
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
#imputer = imputer.fit(df.iloc[:, 9:11].values)
#df.iloc[:,9:11] = imputer.transform(df.iloc[:, 9:11].values)

df=df.replace(np.nan,0)
print(df.info())

print(df.isnull().sum())

#Taking column names ina  list:
col_names=df.columns.values.tolist()

clean_df=pd.DataFrame()

#Columns that won't contribute to machine learning:
to_drop=["User_ID","Product_ID"]
for col in col_names:
    if col in to_drop:
        continue
    else:
        clean_df[col]=df[col]

to_convert=["Product_Category_2","Product_Category_3"]
print("Trying to coerce PC2 and PC3 into int...")
for col in col_names:
    if col in to_convert:
        print("Data type of",col,"is",df[col].dtype)
        clean_df[col]=clean_df[col].astype(int)
        print("Now it is",df[col].dtype)

print("Printing head...")
print(clean_df.head())

#0.67% of total data is classified as outliers (For Product Category 1).
#3.5% of total data is classified as outliers (For Product Category 3).
#The percentage is low, so don't think the outliers are wrongful observations, or that they'd drive the mean to a crzy value.
out_idx={}
for col in col_names:
    print("Running for column",col,"Data type:",df[col].dtype)
    if(df[col].dtype=="int64" or df[col].dtype=="float64"):
        out=list(outlier_test(df[col]))
        if len(out[0])>0:
            print("Outliers found in column",col,"Number of outliers:",len(out[0]))
            out_idx[col]=out
    else:
        print("No outliers in column",col)

print("\nNow for clean data frame...\n")

to_clean=["Stay_In_Current_City_Years"]
col_names1=clean_df.columns.values.tolist() #Not making this was screwing with the cleaning of '+' cause I was using the column names list from unclean dataframe.
#Cleaning the data for '+' signs. I need to put this out in the comments: Categories: '4' for Years in city and '55' for age are same as 4+ and 55+. May replace those in future

col="Stay_In_Current_City_Years"
clean_df[col]=clean_df[col].apply(lambda x: x.strip("+")).astype(int)

col="Age"
clean_df[col]=clean_df[col].apply(lambda x: x.strip("+"))

out_idx={}
for col in col_names1:
    print("Running for column",col,"Data type:",df[col].dtype)
    if(clean_df[col].dtype=="int64" or clean_df[col].dtype=="float64"):
        out=list(outlier_test(clean_df[col]))
        if len(out[0])>0:
            print("Outliers found in column",col,"Number of outliers:",len(out[0]))
            out_idx[col]=out
    else:
        print("No outliers in column",col)
#Now visualizations remain
print(clean_df.describe())

#for random forest:
for_for=clean_df

#The following plot is a visual representation of correlation matrix. Making this to know what affects target (Purchase), both positively and negatively:
#sns.set(style="white")
#corr_mat=clean_df.corr()
#corr_plot=sns.heatmap(corr_mat,vmax=1,center=0,square=True,linewidths=0.5,annot=True)
#mplt.show(corr_plot)

#sct_plot1=sns.relplot(x="Product_Category_1",y="Purchase",hue="Gender",col="Age",data=clean_df)
#mplt.show(sct_plot1)

clean_df["PC1"]=0
clean_df["PC5"]=0
clean_df["PC8"]=0

clean_df.loc[(clean_df.Product_Category_1==1)|(clean_df.Product_Category_2==1)|(clean_df.Product_Category_3==1),"PC1"]=1
clean_df.loc[(clean_df.Product_Category_1==5)|(clean_df.Product_Category_2==5)|(clean_df.Product_Category_3==5),"PC5"]=1
clean_df.loc[(clean_df.Product_Category_1==8)|(clean_df.Product_Category_2==8)|(clean_df.Product_Category_3==8),"PC8"]=1

print(clean_df.head(n=20))
#Testing the max function...:
#print(max(clean_df.Purchase))

#Dummies needed only for regression.
for_reg=pd.DataFrame()

for_reg["Purchase"]=clean_df["Purchase"]
#For gender, "F" is the baseline i.e. dummy 0
for_reg["Gender"]=0
for_reg.loc[(clean_df.Gender=="M"),"Gender"]=1

#6 for age groups.
#For Age, 0-17 is baseline (All age dummies 0): 0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55 (signifies 55+)
for_reg["dum_18-25"]=0
for_reg["dum_26-35"]=0
for_reg["dum_36-45"]=0
for_reg["dum_46-50"]=0
for_reg["dum_51-55"]=0
for_reg["dum_55+"]=0

for_reg.loc[(clean_df.Age=="18-25"),"dum_18-25"]=1
for_reg.loc[(clean_df.Age=="26-35"),"dum_26-35"]=1
for_reg.loc[(clean_df.Age=="36-45"),"dum_36-45"]=1
for_reg.loc[(clean_df.Age=="46-50"),"dum_46-50"]=1
for_reg.loc[(clean_df.Age=="51-55"),"dum_51-55"]=1
for_reg.loc[(clean_df.Age=="55"),"dum_55+"]=1

#So, I am thinking to make 6 dummies for 0,1,4,7,17,20 occupation bins.
for_reg["dum_0"]=0
for_reg["dum_1"]=0
for_reg["dum_4"]=0
for_reg["dum_7"]=0
for_reg["dum_17"]=0
for_reg["dum_20"]=0

for_reg.loc[(clean_df.Occupation==0),"dum_0"]=1
for_reg.loc[(clean_df.Occupation==1),"dum_1"]=1
for_reg.loc[(clean_df.Occupation==4),"dum_4"]=1
for_reg.loc[(clean_df.Occupation==7),"dum_7"]=1
for_reg.loc[(clean_df.Occupation==17),"dum_17"]=1
for_reg.loc[(clean_df.Occupation==20),"dum_20"]=1

#Gonna take city category C as baseline:
for_reg["City_A"]=0
for_reg["City_B"]=0

for_reg["In_City"]=clean_df["Stay_In_Current_City_Years"]

for_reg.loc[(clean_df.City_Category=="A"),"City_A"]=1
for_reg.loc[(clean_df.City_Category=="B"),"City_B"]=1

for_reg["Marital_Status"]=clean_df["Marital_Status"]

for_reg["P1_dum"]=clean_df["PC1"]
for_reg["P5_dum"]=clean_df["PC5"]
for_reg["P8_dum"]=clean_df["PC8"]

print(for_reg.head(n=20))

x=for_reg.iloc[:,1:20].values
y=for_reg.iloc[:,0].values

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

lm=LinearRegression()
result=lm.fit(x_train,y_train)

print("Coefficients: ",lm.coef_)

x2=sm.add_constant(x_train)
ols=sm.OLS(y_train,x2)
ols2=ols.fit()
print(ols2.summary())

pred=lm.predict(x_test)

acc=pd.DataFrame(y_test,pred)
print(acc.head(n=20))

#print(np.sqrt(mean_squared_error(y_test,pred)))
#1) Falk and Miller (1992) recommended that R2 values should be equal to or greater than 0.10 in order for the variance explained of a particular
#endogenous construct to be deemed adequate.
#2) Cohen (1988) suggested R2 values for endogenous latent variables are assessed as follows: 0.26 (substantial), 0.13 (moderate), 0.02 (weak).
#3) Chin (1998) recommended R2 values for endogenous latent variables based on: 0.67 (substantial), 0.33 (moderate), 0.19 (weak).
#4) Hair et al. (2011) & Hair et al. (2013) suggested in scholarly research that focuses on marketing issues, R2 values of 0.75, 0.50, or 0.25
#for endogenous latent variables can, as a rough rule of thumb, be respectively described as substantial, moderate or weak.

#Our Adjusted R-Squared value is 0.300, which qualifes linear regression as a substantial model for predicting sales values.

#Now Random Forest, because it is fairly stron and can be used for classification and regression as well:

x_for=for_for.iloc[:,0:9].values
y_for=for_for.iloc[:,9].values

x_for_train,x_for_test,y_for_train,y_for_test=train_test_split(x_for,y_for,test_size=0.2,random_state=0)

#Let's start with, encoding:

lab_encoder=LabelEncoder()
#encode=["Gender","City_Category"]
#for col in encode:
#For Gender:
x_for_train[:,0]=lab_encoder.fit_transform(x_for_train[:,0])
x_for_test[:,0]=lab_encoder.fit_transform(x_for_test[:,0])

#For Age Groups:
x_for_train[:,1]=lab_encoder.fit_transform(x_for_train[:,1])
x_for_test[:,1]=lab_encoder.fit_transform(x_for_test[:,1])

#For City Category:
x_for_train[:,3]=lab_encoder.fit_transform(x_for_train[:,3])
x_for_test[:,3]=lab_encoder.fit_transform(x_for_test[:,3])

#Now we gonna do scaling:
scalar=StandardScaler()
x_for_train=scalar.fit_transform(x_for_train)

test_scalar=StandardScaler()
x_for_test=test_scalar.fit_transform(x_for_test)

def mean_error(max_leaves,X_train,X_test,Y_train,Y_test):
    model=RandomForestRegressor(max_leaf_nodes=max_leaves,random_state=0)
    model.fit(X_train,Y_train)
    pred=model.predict(X_test)
    mae=mean_absolute_error(Y_test,pred)
    print("For",max_leaves,"nodes, the mean absolute error:",mae)
    mse=mean_squared_error(Y_test,pred)
    rmse=np.sqrt(mse)
    print("Root Mean Squared value for",max_leaves,"nodes is:",rmse)

for i in [700,800,1200,2000,3000,4000]:
    mean_error(i,x_for_train,x_for_test,y_for_train,y_for_test)

#We gotta make bins for purchase amount:
bin_width=(((max(clean_df.Purchase))-(min(clean_df.Purchase)))/5)
print("For 5 bins, bin width=",bin_width)

bins=[]
lab=[]
i=0
while i<6:
    if i==0:
        bins.append(min(clean_df.Purchase))
        lab.append(i)
    else:
        bins.append(bins[i-1]+bin_width)
        lab.append(i)
    i=i+1

#print(bins)
lab=lab[:-1]
#print(lab)

y_train_bin=pd.cut(y_for_train,bins=bins,labels=lab)
print(y_train_bin[:20])

y_test_bin=pd.cut(y_for_test,bins=bins,labels=lab)

y_train_bin=y_train_bin.fillna(4)
y_test_bin=y_test_bin.fillna(4)
#print(y_train_bin.describe())
#print(y_test_bin.describe())

def forest_classifier(max_leaves,X_train,X_test,Y_train,Y_test):
    classifier=RandomForestClassifier(max_leaf_nodes=max_leaves,bootstrap=True,criterion='gini',n_estimators=150)
    classifier.fit(X_train,Y_train)
    pred=classifier.predict(X_test)
    print(confusion_matrix(Y_test,pred))
    print(classification_report(Y_test,pred))
    print(accuracy_score(Y_test,pred))

for i in [500,750,100,1250,1500]:
    forest_classifier(i,x_for_train,x_for_test,y_train_bin,y_test_bin)

#34.49% of accuracy (highest) for a forest with 150 trees and 1500 maximum leaves (Gini method) (20 bins)
#62.18% of accuracy (highest) for a forest with 150 trees, 1500 leaves (entropy) (5 bins)
#61.98% accuracy for a forest with 150 trees and 1250 leaves (gini) (5 bins)

#The code was giving this error earlier:
'''
Traceback (most recent call last):
  File "project.py", line 310, in <module>
    forest_classifier(i,x_for_train,x_for_test,y_train_bin,y_test_bin)
  File "project.py", line 303, in forest_classifier
    classifier.fit(X_train,Y_train)
  File "/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py", line 251, in fit
    y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
  File "/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py", line 573, in check_array
    allow_nan=force_all_finite == 'allow-nan')
  File "/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py", line 56, in _assert_all_finite
    raise ValueError(msg_err.format(type_err, X.dtype))
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
'''
#Evaluating the structure of the y_test_bin showed me that 2 values are binned as 'NaN', for test data and 5 for train data (damn it took so long....)
#That is because of the sheer blunder in binning. Cause for max, the bining boundary is actually 0.0000.....0001 less than actual max value. Thus the 2 data points with Purchase value
#equal to max value are binned as NaN.

#I tried using 'replace', but I had an AttributeError: AttributeError: 'Categorical' object has no attribute 'replace'
