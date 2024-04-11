from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.DataFrame, title: str = ''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

def dates_from_str(dates: List[str]):
    """
    Function to convert dates from strings to datetime objects in the format "%B %Y",
    where '%B' is the full month name and '%Y' is the year with century.
    """
    return [datetime.strptime(date, "%B %Y") for date in dates]

def convert_dates_with_mappings(index: List[str]):
    mapping_cro_eng = {
        'Siječanj': 'January',
        'Veljača': 'February',
        'Ožujak': 'March',
        'Travanj': 'April',
        'Svibanj': 'May',
        'Lipanj': 'June',
        'Srpanj': 'July',
        'Kolovoz': 'August',
        'Rujan': 'September',
        'Listopad': 'October',
        'Studeni': 'November',
        'Prosinac': 'December'
    }
    mapping_years = {
        "'20": '2020',
        "'21": '2021',
        "'22": '2022',
        "'23": '2023',
        "'24": '2024',
    }

    # Convert the index using the mappings
    converted_index = [
        f"{mapping_cro_eng[x.split(' ')[0]]} {mapping_years[x.split(' ')[1]]}"
        for x in index
    ]

    # Use the dates_from_str function to convert to datetime objects
    return dates_from_str(converted_index)


def load_and_prepare_data(csv_path: str, keep_columns: list[str]):
    # load time series data
    mdata = pd.read_csv(csv_path, index_col=0)
    parties = ["HDZ", "SDP", "MOST", "DP", "MOZEMO"]

    mdata.index = convert_dates_with_mappings(mdata.index)

    keep_columns = parties + keep_columns
    mdata = mdata[keep_columns]

    # each column value ends with % sign, we need to remove it
    mdata = mdata.replace({'%': ''}, regex=True)
    mdata = mdata.replace({'€' : ''}, regex=True)
    mdata = mdata.replace({',' : ''}, regex=True)
    mdata = mdata.astype(float)
    return mdata


def perform_adf_test(mdata: pd.DataFrame):
    for column in mdata.columns:
        adf_test(mdata[column], title=column)
        print('\n')
    
    mdata_diff = mdata.diff().dropna()

    for column in mdata_diff.columns:
        adf_test(mdata_diff[column], title=f'{column} (diff)')
        print('\n')


def find_optimal_model_params(mdata: pd.DataFrame, max_order: int = 4):
    model = VAR(mdata)
    
    for ic in ['aic', 'bic', 'hqic', 'fpe']:
        for trend in ['c', 'ct', 'ctt']:
            result = model.fit(max_order, ic=ic, trend=trend, verbose=True)
            print(f'IC: {ic}, Trend: {trend}, Order: {result.k_ar}, AIC: {result.aic}, BIC: {result.bic}, HQIC: {result.hqic}, FPE: {result.fpe}, LLF: {result.llf}')
    


def fit_var_model(train_endog: pd.DataFrame, train_exog: pd.DataFrame, max_lags: int, ic: str = 'aic'):
    model = VAR(endog=train_endog, exog=train_exog)
    result = model.fit(max_lags, ic=ic, verbose=True)
    return result


def forecast_var_model(result, train_endog, train_exog, num_predictions: int, lag_order: int):
    forecast = result.forecast(train_endog.values[-lag_order:], num_predictions, exog_future=train_exog.values[-num_predictions:])
    return forecast


def correct_forecast(forecast: pd.DataFrame, test: pd.DataFrame, train_original: pd.DataFrame):
    idx = test.index
    # get number of predictions
    num_predictions = forecast.shape[0]
    df_forecast = pd.DataFrame(forecast, index=idx, columns=test.columns + '2d')

    for column in train_original.columns:
        df_forecast[f'{column}1d'] = train_original[column].iloc[-1] - train_original[column].iloc[-2] + df_forecast[f'{column}2d'].cumsum()
        df_forecast[f'{column}forecast'] = train_original[column].iloc[-1] + df_forecast[f'{column}1d'].cumsum()

    return df_forecast


def analyze_predicted(test: pd.DataFrame, df_forecast: pd.DataFrame):
    # compare predicted vs actual values for each column
    parties = ["HDZ", "SDP", "MOST", "DP", "MOZEMO"]
    actual = test[parties]
    predicted = df_forecast[[f'{column}forecast' for column in parties]]

    mse = np.mean((actual.values - predicted.values)**2)

    return mse, actual, predicted


def perform_var_analysis(endog_mdata: pd.DataFrame, exog_mdata, max_lags: int, test_size: int):
    endog_mdata_diff = endog_mdata.diff().dropna()
    exog_mdata_diff = exog_mdata.diff().dropna()

    train_endog = endog_mdata_diff.iloc[:-test_size]
    test_endog = endog_mdata.iloc[-test_size:]
    train_endog_original_values = endog_mdata.iloc[:-test_size]

    train_exog = exog_mdata_diff.iloc[:-test_size]
    test_exog = exog_mdata.iloc[-test_size:]
    train_exog_original_values = exog_mdata.iloc[:-test_size]

    result = fit_var_model(train_endog, train_exog, max_lags)
    forecast = forecast_var_model(result, train_endog, train_exog, test_size, max_lags)
    df_forecast = correct_forecast(forecast, test_endog, train_endog_original_values)
    mse, actual, predicted = analyze_predicted(test_endog, df_forecast)

    return mse, df_forecast, actual, predicted


def perform_var_using_test_set(endog_mdata: pd.DataFrame, exog_mdata: pd.DataFrame, max_lags: int, test_set_size: int):
    mse_list = []
    for i in range(test_set_size):
        if i > 0:
            endog_mdata_current = endog_mdata.iloc[:-i]
            exog_mdata_current = exog_mdata.iloc[:-i]
        else:
            endog_mdata_current = endog_mdata
            exog_mdata_current = exog_mdata
        mse, _, actual, predicted = perform_var_analysis(
            endog_mdata=endog_mdata_current,
            exog_mdata=exog_mdata_current,
            max_lags=max_lags,
            test_size=1
        )
        print(f'Discarding {i} observations')
        print(f'MSE: {mse}')
        print(f'Actual values: {actual.values}')
        print(f'Predicted values: {predicted.values}')
        print()
        mse_list.append(mse)
    
    return np.mean(mse_list)
