
from multiprocessing import Process
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import random
from mathCode.johansenMain import coint_johansen
from tqdm import tqdm_notebook
from IPython.display import clear_output
import itertools
import datetime
import os
from base_model_backtest import backtest
import random
from pandas_datareader import data as pdr
import logging
from pandas.plotting import register_matplotlib_converters
from functionSource import *

register_matplotlib_converters()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
plt.style.use(['ggplot'])


from models.Cointegrated_Pairs_1 import cointegrated_pairs_1


        
global toTrade
global price
global df
global results
global runDate

#Ranges for parameters
ve_values =  [round(0.1**(ve_), 11) for ve_ in range(1,10)]
delta_values = [round(0.1**(delta_), 11) for delta_ in range(1,10)]

# ve_values = [1e-10]
# delta_values = [0.005]

transactionFeesPercent = 0.0008
minimumComission = 0
capitalPerTrade = 1

# toTrade = ['EX20.AX','ISO.AX']
# toTrade = ['BBOZ.AX','ETPMAG.AX']
# toTrade = ['VAS.AX','ETPMPM.AX']


# toTrade = ['OZR.AX','GLIN.AX']
# ve_values = [2e-10]
# delta_values = [2e-05]


toTrade = ['GEAR.AX','BBOZ.AX']
# ve_values = [2e-05]
# delta_values = [4e-07]

# [4e-07,
#   2e-05,
#   6.074345979618439,
#   0.0023309078970139827,
#   5.778966018256601,
#   0.0022175617875121265,
#   61.403807164207265,
#   58.417896541320324,
#   'TLSAX_SKCAX_.pickle']


# toTrade = ['STW.AX','IJR.AX']
# toTrade = ['IFRA.AX','GLIN.AX']
# toTrade = ['IVE.AX','VHY.AX']

riskFreeTs = '^AXJO'


# #BUTCOIN
# df = pickle.load(open('D:/Crypto_data/ohlc_data/poloniex/1d/BTC-USDT_data.pickle','rb'))
# df2 = pickle.load(open('D:/Crypto_data/ohlc_data/poloniex/1d/ETH-USDT_data.pickle','rb'))
# df = pd.concat([df.Close,df2.Close], axis=1)
# df.columns = ['BTC','ETH']
# close = df.dropna()

# #ERNIE CHAN DATA
# df = pd.read_csv('D:/Machine_learning_finance/raw_data/ewaewcige.csv')
# df.set_index('Date',inplace=True)
# toTrade = ['ewa','ewc']
# close = df[toTrade]
# ve_values = [0.001]
# delta_values = [0.0001]


#RUNESCAPE
# toTrade = ['Nature rune','Law rune']
# asset_a = toTrade[0]
# asset_b = toTrade[1]
# data = pd.read_csv('D:/Runescape_Market_Data.csv')
# data = pd.DataFrame(data.set_index(['ItemName','PriceDate'])['Price'])
# a = data.loc[([asset_a, asset_b],)].loc[asset_a]
# a.columns =[asset_a]
# b = data.loc[([asset_a, asset_b],)].loc[asset_b]
# b.columns = [asset_b]
# close = a.join(b, on='PriceDate').sort_index()
# print(toTrade, close_.head())

# ASX DATA
data = pickle.load(open( "D:/PRISMO/historicalData/ETFSnobiasFINAL.pickle", "rb" ) )
# data = pickle.load(open( "D:/PRISMO/historicalData/data/asx200nobiasFINAL.pickle",'rb'))
close = data.xs('CLOSE', axis = 1, level = 1)
close = close[toTrade].dropna()

# data = pickle.load(open('D:/PRISMO/historicalData/data/asx200nobiasFINAL.pickle','rb'))
data_open = data.loc[(slice(None), toTrade)].xs('OPEN',axis=1,level=1).dropna()
data_open.index = data_open.index+pd.Timedelta(10,'H')
data_close = data.loc[(slice(None), toTrade)].xs('CLOSE',axis=1,level=1).dropna()
data_close.index = data_open.index+pd.Timedelta(16,'H')
openAndClose = pd.concat([data_open,data_close]).sort_index()
openAndClose['OPEN'] =False
openAndClose['CLOSE'] =False
openAndClose.loc[openAndClose.index.hour==16,'CLOSE']=True
openAndClose.loc[openAndClose.index.hour==10,'OPEN']=True
close = openAndClose

#Create rundate and directory of logs and results
runDate = str(datetime.datetime.today()).replace('-','_').replace(' ','_').replace(':','_')[:-10]
logDirectory = 'D:/PRISMO/logs/MULTI_pairTradeKalman_' + flatten(toTrade) +'_'+ runDate 
try:
    os.makedirs(logDirectory)
except Exception as e:
    pass

results = setup_logger('results', logDirectory+'/results.log')
results.info('Delta, Ve, Final Returns Value, Number of Trades')



def runThatShitBack(delta, cointegrated_pairs_1):

    for ve in ve_values:

    
        #Update to include specific deltas and ve
        logDirectoryBatch = logDirectory +  '/delta_'+str(delta) + '_ve_'+str(ve)
        try:
            os.makedirs(logDirectoryBatch)
        except Exception as e:
            pass

        strategy = cointegrated_pairs_1(close[toTrade],
                            toTrade, 
                            3, 
                            False, 
                            plotBool = False, 
                            decimalPlaces = None,
                            lookback = None, 
                            delta = delta,
                            ve = ve,
                            riskFreeTs = riskFreeTs,
                            logDir = logDirectoryBatch,
                            transactionFeesPercent = transactionFeesPercent,
                            minimumComission = minimumComission,
                            InitialCapital = 0,
                            capitalPerTrade = capitalPerTrade,
                            simOpen = 1)
       
        #Run the
        strategy.run()
        plt.plot(strategy.hedges)
        plt.show()

        #Print simmary of results
        results.info([delta, ve, np.cumsum(strategy.returns.values)[-1], strategy.numberOfTrades])

       #Re-index the returns, placing zero on days where no trades were made
        dates = strategy.dfHistory.index
        returns =  strategy.returns.T.reindex(dates, fill_value=0)

        #Calculate Drawdowns
        drawdownArray, drawDownDf = drawdowns(returns.returns)
            
        #Calculate Alpha, beta, sharpe values
        alpha, beta = alpha_beta(returns, riskFreeTs)
        sharpeRatio = sharpe(returns, riskFreeTs)
        APRvalue = APR(returns)
        #Print simmary of results
        results.info([delta, ve, np.cumsum(strategy.returns.values)[-1], strategy.numberOfTrades])


        plotResults(dates,
                    returns,
                    np.cumsum(returns),
                    riskFreeTs,
                    directory = logDirectoryBatch,
                    drawdownArray = drawdownArray,
                    maxDrawDown = round(min(drawdownArray),4),
                    maxDrawDownDuration = drawDownLength(drawDownDf),
                    alpha = round(alpha,4),
                    beta = round(beta,4),
                    sharpe = round(sharpeRatio,4),
                    APR = round(APRvalue,4),
                    additionalData = [['ve',str(ve)],
                                        ['delta',str(delta)],
                                        ['comission',str(transactionFeesPercent)],
                                        ['minComish',str(minimumComission)],
                                        ['capPerTr',str(capitalPerTrade)],
                                        ['numTrades',str(strategy.numberOfTrades)]])




        # except Exception as e:
        #     print(e)



if __name__ == '__main__':

    processes = []
    for delta in delta_values:
        process = Process(target=runThatShitBack, args=(delta, cointegrated_pairs_1))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


    
