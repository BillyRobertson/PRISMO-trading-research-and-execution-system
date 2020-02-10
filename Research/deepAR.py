from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas

#Read Data
data = pickle.load(open( "D:/PRISMO/historicalData/Data/asx200nobiasFINAL.pickle", "rb" ) )
data = pickle.load(open('D:/PRISMO/historicalData/data/asx200nobiasFINAL.pickle','rb'))
data_ = data.xs('OPEN',axis=1,level=1).dropna()
stocks = data_.columns
#Concatenate open and close prices
data_open = data.xs('OPEN',axis=1,level=1)
data_open.index = data_open.index+pd.Timedelta(10,'H')
data_close = data.xs('CLOSE',axis=1,level=1)
data_close.index = data_open.index+pd.Timedelta(16,'H')
openAndClose = pd.concat([data_open,data_close]).sort_index()
stocks = openAndClose.columns

pred_len = 2

temp = data_close[['TLS.AX',]].dropna()
train_ratio = 0.88

train_data = temp.iloc[:int(train_ratio*len(temp))]
test_data = temp.iloc[: int((train_ratio)*len(temp))+pred_len]
target_asset = temp.columns[0]

#Create univariate dataset
training_data = ListDataset(
    [{"start": train_data[target_asset].index[0],
      "target": train_data[target_asset]}],
    freq = "1d"
)

# Create testing datatset
testing_data = ListDataset(
    [{"start": test_data[target_asset].index[0],
      "target": train_data[target_asset]}],
    freq = "1d"
)

# Create testing datatset
testing_data_plots = ListDataset(
    [{"start": test_data[target_asset].index[0],
      "target": test_data[target_asset]}],
    freq = "1d"
)
#Create the estimator and train
estimator = DeepAREstimator(freq="1d", prediction_length=pred_len, trainer=Trainer(epochs=100))
predictor = estimator.train(training_data=training_data)

### OPTIONAL PLOT PREDICTION RESULTS
#Forecast
for test_entry, forecast in zip(testing_data_plots, predictor.predict(testing_data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
    
### GENERATE FORECASTS  
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=testing_data,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)


###### VERY SIMPLE TRADING STRATEGY
#      Signal: If Forecast > Current Price, Buy
                     "     <      "       , Short


test_data = temp.iloc[:int(train_ratio*len(temp))+pred_len] 
start_date = train_data.index[-1]

df = pd.DataFrame()
date= start_date

#Iterate over remaining data 
for i in range(len(temp.iloc[int(train_ratio*len(temp))+pred_len:])):

    #Compare yesterdays prediction with today's close price
    data_temp = temp.iloc[i:int(train_ratio*len(temp))+pred_len+i]
    currentPrice = data_temp[target_asset].iloc[-1]
    
    #If yesterdays predicion in the df, compare the prediction with the current close price
    if date-pd.Timedelta(1,'D') in df.index:
        yesterdayDate = date-pd.Timedelta(1,'D') 
        if currentPrice <= df.loc[(yesterdayDate, 'lowerQ')]:
            df.loc[(date, 'signal') ] =  'BUY'
            print('BUY')
        elif currentPrice >= df.loc[(yesterdayDate, 'upperQ')]:
            df.loc[(date, 'signal') ] =  'SHORT'
            print('SHORT')
        else:
            df.loc[(date, 'signal') ] =  np.nan
    
    
    
    #Compute tomorrows prediction

    # Create testing datatset
    testing_data = ListDataset(
        [{"start": data_temp[target_asset].index[0],
          "target": data_temp[target_asset]}],
        freq = "1d"
    )

    
    prediction = []
    for test_entry, forecast in zip(testing_data, predictor.predict(testing_data)):
#         to_pandas(test_entry)[-20:].plot()
#         forecast.plot()
#         plt.show()
        prediction.append(forecast.quantile(0.1)[0])
        prediction.append(forecast.mean[0])
        prediction.append(forecast.quantile(0.9)[0])
        
    df.loc[(date, 'price') ] =  currentPrice 
    df.loc[(date, 'lowerQ')] = prediction[0]
    df.loc[(date, 'mean')] = prediction[1]
    df.loc[(date, 'upperQ')] = prediction[2]
      
    date += pd.Timedelta(1,'D')
    
df['returns']=df['price'].pct_change()
df['returnsShift1'] = df['returns'].shift(-1)
df['long'] = 0
df['short'] = 0
df['long'][df['signal']=='BUY']=1
df['short'][df['signal']=='SHORT']=1
df['numUnits'] = df['long']-df['short']
df['marketValue'] = df['numUnits']*df['returnsShift1']
plt.plot(np.cumsum(df['marketValue']))




  
