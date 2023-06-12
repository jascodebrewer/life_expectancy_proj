from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# load the data and make its copy
aggregated_df_main = pd.read_csv('life_expectancy_proj/aggregated_data.csv')
aggregated_df = aggregated_df_main.copy()

# Split the data into test and train
X = aggregated_df.drop(columns=['Life expectancy'])
y = aggregated_df['Life expectancy']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Data Processing Steps"""
# Remove missing values from the training data
# Identify columns with missing values
missing_cols = X.columns[X_train.isnull().any()].tolist()
# missing_cols_test = X_train.columns[X_test.isnull().any()].tolist()

# Create an imputer and fit on training data
imputer = SimpleImputer(strategy='mean')
X_train[missing_cols] = imputer.fit_transform(X_train[missing_cols])
X_test[missing_cols] = imputer.transform(X_test[missing_cols])
missing_cols_train = X_train.columns[X_train.isnull().any()].tolist()
mean_value = X_test['Alcohol'].mean()
X_test['Alcohol'].fillna(mean_value, inplace=True)

# ONE-HOT Encoding
# Identify categorical columns
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
OHE = ce.OneHotEncoder(cols=cat_cols,use_cat_names=True)
# encode the categorical variables
X_train_encoded = OHE.fit_transform(X_train)
X_test_encoded = OHE.transform(X_test)

# Feature Scaling
num_cols = X_train_encoded.select_dtypes(include='number').columns.tolist()
# apply standardization on numerical features
for i in num_cols:
    
    # fit on training data column
    scale = StandardScaler().fit(X_train_encoded[[i]])
    
    # transform the training data column
    X_train_encoded[i] = scale.transform(X_train_encoded[[i]])
    
    # transform the testing data column
    X_test_encoded[i] = scale.transform(X_test_encoded[[i]])


"""Model Building"""
# Multiple Linear Regression
import statsmodels.api as sm
# Select numerical columns
X_train_num = X_train.select_dtypes(include='number')
# to ensure that the indices of X_train and y_train are aligned 
# correctly before fitting the model
X_train_num.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
model = sm.OLS(y_train, sm.add_constant(X_train_num))
results = model.fit()
summary = results.summary()
print(summary)
# Extract the table data and convert it to a dataframe
table_data = summary.tables[1]
df_summary = pd.DataFrame(table_data.data[1:], columns=table_data.data[0])
# Save the dataframe as a CSV file
df_summary.to_csv('life_expectancy_proj/stats_summary.csv', index=False)
'''
   Observations: Adult Mortality, BMI, Schooling, Infant Deaths, under-five deaths,
   and Income composition of resources
   have p-value<0.05
   R-squared and Adj. R-squared values are closed to 0.9 (0.937 and 0.928 respectively)
'''
# Random Forest Regressor
forest_clf = RandomForestRegressor(n_estimators=100, random_state=42)
forest_clf.fit(X_train_encoded, y_train)
y_pred = forest_clf.predict(X_test_encoded)
forest_scores = cross_val_score(forest_clf, X_train_encoded, y_train, cv=5)
print('*'*20)
print('Random Forest\'s cross_val_score: ', forest_scores.mean())
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error for Random Forest:", mae)
print("Mean Squared Error for Random Forest:", mse)
print("Root Mean Squared Error for Random Forest:", rmse)
print("R-squared for Random Forest:", r2)
print('*'*20)

# Linear Regression
lm = LinearRegression()
lm.fit(X_train_encoded, y_train)
y_pred = lm.predict(X_test_encoded)
linear_scores = cross_val_score(lm,X_train_encoded,y_train, cv= 5)
print('Linear regression\'s cross_val_score: ', linear_scores.mean())
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error (Linear Regression):", mae)
print("Mean Squared Error (Linear Regression):", mse)
print("Root Mean Squared Error (Linear Regression):", rmse)
print("R-squared (Linear Regression):", r2)
print('*'*20)

# Lasso regression
lm_L = Lasso()
lm_L.fit(X_train_encoded, y_train)
y_pred = lm_L.predict(X_test_encoded)
lasso_scores = cross_val_score(lm_L,X_train_encoded,y_train, cv= 5)
print('Lasso\'s cross_val_score: ',lasso_scores.mean())
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error (Lasso regression):", mae)
print("Mean Squared Error (Lasso regression):", mse)
print("Root Mean Squared Error (Lasso regression):", rmse)
print("R-squared (Lasso regression):", r2)
print('*'*20)

# Decision Trees
# Create a decision tree regressor
tree_reg = DecisionTreeRegressor(random_state=42)
# Fit the model to the training data
tree_reg.fit(X_train_encoded, y_train)
y_pred = tree_reg.predict(X_test_encoded)
tree_scores = cross_val_score(tree_reg,X_train_encoded,y_train, cv= 5)
print('Decicion Tree\'s cross_val_score: ', tree_scores.mean())
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error (Decision Tree):", mae)
print("Mean Squared Error (Decision Tree):", mse)
print("Root Mean Squared Error(Decision Tree):", rmse)
print("R-squared (Decision Tree):", r2)
print('*'*20)