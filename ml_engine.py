'''
    Author: Rui Sun
    Date: 2022-09-25

    Predict timeseries data

    Official guide book of Prophet: https://facebook.github.io/prophet/docs/quick_start.html#python-api
'''

from prophet import Prophet


class MLPredictor(object):
    '''
    Example usage method:

        from ml_engine import MLPredictor

        predictor = MLPredictor(pm25_df)
        predictor.train()
        forecast = predictor.predict()


        fig = predictor.plot_result(forecast)
        fig.savefig(os.path.join("Your target dir path", "Your target file name))

    '''

    def __init__(self, data_df):
        '''
        :param data_df: Dataframe type dataset
        '''
        self.__train_data = self.__convert_col_name(data_df)
        self.__trainer = Prophet(changepoint_prior_scale=12)

    def train(self):
        self.__trainer.fit(self.__train_data)

    def __convert_col_name(self, data_df):
        data_df.rename(columns={"Timestamp": "ds", "Value": "y"}, inplace=True)
        print(f"After rename columns \n{data_df.columns}")
        return data_df

    def __make_future(self, periods=15):
        future = self.__trainer.make_future_dataframe(periods=periods)
        return future

    def predict(self):
        future = self.__make_future()
        forecast = self.__trainer.predict(future)
        return forecast

    def plot_result(self, forecast):
        fig = self.__trainer.plot(forecast, figsize=(15, 6))
        return fig
