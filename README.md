## Ex.No : 08 MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### DATE :

### AIM :
To implement Moving Average Model and Exponential smoothing Using Python.

### ALGORITHM :
1. Import necessary libraries
2. Read the CSV file.
3. Display the shape and the first 10 rows of the dataset
4. Perform rolling average transformation with a window size of 5 and 10
5. Display first 10 and 20 values repecively and plot them both
6. Perform exponential smoothing and plot the fitted graph and orginal graph

### PROGRAM :
```
Developed By : NITHYA D
Reg.No : 212223240110
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
```
#### Read the Gold Price dataset
```
data = pd.read_csv('Gold Price Prediction.csv')
```
#### Parse and set the Date column as datetime index
```
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.sort_values('Date')  # Ensure chronological order
data.set_index('Date', inplace=True)
```
#### Focus on 'Price Today' as the target variable
```
gold_price = data[['Price Today']]
```
#### Display basic info
```
print("Shape of the dataset:", gold_price.shape)
print("First 10 rows of the dataset:")
print(gold_price.head(10))
```
#### Plot original gold price data
```
plt.figure(figsize=(12, 6))
plt.plot(gold_price, label='Gold Price Today')
plt.title('Gold Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
```
#### Moving averages
```
rolling_mean_5 = gold_price['Price Today'].rolling(window=5).mean()
rolling_mean_10 = gold_price['Price Today'].rolling(window=10).mean()

print("Rolling Mean (window=5):")
print(rolling_mean_5.head(10))

print("Rolling Mean (window=10):")
print(rolling_mean_10.head(20))
```
#### Plot moving averages
```
plt.figure(figsize=(12, 6))
plt.plot(gold_price['Price Today'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)', linestyle='--')
plt.plot(rolling_mean_10, label='Moving Average (window=10)', linestyle='--')
plt.title('Moving Averages of Gold Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
```
#### Resample monthly (optional, depending on granularity—comment out if already daily)
```
data_monthly = gold_price.resample('MS').mean()
```
#### Scale data
```
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
```
#### Multiplicative model requires strictly positive data
```
scaled_data += 1
```
#### Train/test split
```
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]
```
#### Fit model with additive trend and multiplicative seasonality
```
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
```
#### Visual comparison
```
ax = train_data.plot(figsize=(12, 6))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["Train Data", "Test Predictions", "Test Data"])
ax.set_title('Exponential Smoothing Forecast - Gold Price')
plt.grid()
plt.show()
```
#### RMSE evaluation
```
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f'RMSE: {rmse:.4f}')
```
#### Summary stats
```
print("Standard Deviation:", np.sqrt(scaled_data.var()))
print("Mean:", scaled_data.mean())
```
#### Predict for one-fourth of the dataset (future forecasting)
```
model_full = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model_full.forecast(steps=int(len(data_monthly) / 4))
```
#### Plot final prediction
```
ax = data_monthly.plot(figsize=(12, 6))
predictions.plot(ax=ax)
ax.legend(["Gold Price", "Forecast"])
ax.set_xlabel('Date')
ax.set_ylabel('Gold Price')
ax.set_title('Gold Price Forecast (Next Quarter Horizon)')
plt.grid()
plt.show()

```

### OUTPUT :
#### Original data :
![image](https://github.com/user-attachments/assets/d0b81f70-adba-425b-b1ce-668f697d85ce)

![image](https://github.com/user-attachments/assets/8230523b-1b98-4b57-85ff-b2457f35aa3c)

#### Moving Average:- (Rolling) 
###### window(5) :
![image](https://github.com/user-attachments/assets/ef2b3cbe-b664-45fb-a4d7-daeb4484ac26)
###### window(10) :
![image](https://github.com/user-attachments/assets/31d5ac24-20c1-44a8-bede-c99b3b772b75)

#### Plot:
![image](https://github.com/user-attachments/assets/083398d9-0911-4a46-9053-637f179e73a6)

#### Exponential Smoothing:-
###### Test:
![image](https://github.com/user-attachments/assets/9ac94ca8-b1b4-4577-b964-e10b00d600d7)

##### Performance: (MSE)

![image](https://github.com/user-attachments/assets/3028f063-9235-442c-a328-b14cb182d646)

#### Prediction:
![image](https://github.com/user-attachments/assets/e8e512e6-c17f-466e-b78a-98f793b5897c)

### RESULT :
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
