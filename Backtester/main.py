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

def pickler(directory, data):
    pickle_out = open(directory,'wb')
    pickle.dump(data,pickle_out)
    pickle_out.close()
    
def dePickler(directory):
    pickle_in = open(directory,'rb')
    return pickle.load(pickle_in)


from models.gapping3longshort import gapping_3


        
global toTrade
global price
global df
global results
global runDate




#Ranges for parameters
# stdLookbacks = [90]
# ZscoreEntry = [1]

stdLookbacks = [90]
ZscoreEntry = [2]
quantile = [10]


data = pickle.load(open( "D:/PRISMO/historicalData/data/asx200nobiasInIndexAdded.pickle", "rb" ) )
inIndex = data.xs('inIndex', axis = 1, level = 1).loc['2010-01-01 00:00:00':]
df_close = data.xs('CLOSE', axis = 1, level = 1).loc['2010-01-01 00:00:00':]
df_open = data.xs('OPEN', axis = 1, level = 1).loc['2010-01-01 00:00:00':]
df_low = data.xs('LOW', axis = 1, level = 1).loc['2010-01-01 00:00:00':]


toTrade = list(df_close.columns)

# print(df.head())

#Create rundate and directory of logs and results
runDate = str(datetime.datetime.today()).replace('-','_').replace(' ','_').replace(':','_')[:-10]

logDirectory = 'D:/PRISMO/logs/ASXgapping2SHORTED'+ runDate 

try:
    os.makedirs(logDirectory)
except Exception as e:
    pass

results = setup_logger('results', logDirectory+'/results.log')
results.info('stdLookback,Zscore entry, Final Returns Value, Number of Trades')

riskFreeTs = '^AXJO'


def runThatShitBack(capital, gapping_1):
        quant_ = 10
        for stdLookback in stdLookbacks:
            for Z in ZscoreEntry:
            #Update to include specific deltas and ve
                    logDirectoryBatch = logDirectory +  '/stdLookback_'+str(stdLookback)+'_Zscore_'+str(Z)+'_quantile_'+str(10) + '_capital_'+str(capital)
                    try:
                        os.makedirs(logDirectoryBatch)
                    except Exception as e:
                        pass

                    strategy = gapping_3(df_close,
                                      df_low,
                                      df_open,
                                      inIndex,
                                      list(df_close.columns),
                                      None, 
                                      True,
                                      True,
                                      logDir = logDirectoryBatch,
                                      transactionFeesPercent = 0.0008,
                                      minimumComission = 0,
                                      InitialCapital = 0,
                                      capitalPerTrade = capital,
                                      stdLookback = stdLookback,
                                      ZscoreShort = Z,
                                      ZscoreLong = 1,
                                      quantile = quant_)


                    #Run that shit family
                    strategy.run()

                    #Re-index the returns, placing zero on days where no trades were made
                    dates = strategy.dfHistory.index
                    # returns =  strategy.returns.T.reindex(dates, fill_value=0)

                    #Calculate Drawdowns
                    drawdownArray, drawDownDf = drawdowns(returns.returns)
                        
                    #Calculate Alpha, beta, sharpe values
                    alpha, beta = alpha_beta(returns, riskFreeTs)
                    sharpeRatio = sharpe(returns, riskFreeTs)
                    APRvalue = APR(returns)
                    #Print simmary of results
                    results.info([stdLookback,Z,quant_, capital, np.cumsum(strategy.returns.values)[-1], strategy.numberOfTrades])

                    plotResults(dates,
                                returns,
                                strategy.cumulativeCapital,
                                riskFreeTs,
                                directory = logDirectoryBatch,
                                drawdownArray = drawdownArray,
                                maxDrawDown = round(min(drawdownArray),4),
                                maxDrawDownDuration = drawDownLength(drawDownDf),
                                alpha = round(alpha,4),
                                beta = round(beta,4),
                                sharpe = round(sharpeRatio,4),
                                APR = round(APRvalue,4),
                                additionalData = [['Z ENT',str(Z)],['NUMSHRT',str(quant_)],['STDlookb',str(stdLookback)],['CAP',capital]])


    

if __name__ == '__main__':

    processes = []
    for capital in [6275]:
        process = Process(target=runThatShitBack, args=(capital, gapping_3))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


    
