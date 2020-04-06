import sys
sys.path.append('C:/Users/Billy/Documents/PRISMO/Backtester')
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

pairs = [['TLS.AX','SKC.AX','Stock']]
def APR(returns):
    ave = []
    for year in range(returns.index[0].year, returns.index[-1].year+1):
        annualReturns = returns.loc[returns.index<str(year+1)+'-01-01']
        annualReturns = annualReturns.loc[annualReturns.index>=str(year)+'-01-01']

        aveReturns = np.mean(annualReturns) 
        ave.append(aveReturns*252*100)
        

    return np.mean(ave)


def flatten(list_):
    string = ''
    for element in list_:
        string = string + element.replace('.','')+'_'

    return string

def monotonicityMeasure(returns):
     # Start by counting how many points in the future returns are greater than the current
     # If its less than current, we want to emphasise that it's non monatonic in the long run and is a bad strategy
    # Check x chunks for monotonicity
    chunks = 50
    binWidth = int(len(returns)/chunks)
    
    totalCounter = 0
    monotonicCounter = 0
    for i in range(0,len(returns),binWidth):
        for j in range(i+chunks,len(returns),binWidth):
            
            if returns[j]>returns[i]:
                monotonicCounter +=1
            
            totalCounter+=1
            
    return monotonicCounter/totalCounter
    



def optimize(toTrade):

    asset = toTrade[-1]
    toTrade = toTrade[:-1]

    if asset == 'ETF':
        data = pickle.load(open( "D:/PRISMO/historicalData/ETFSnobiasFINAL.pickle", "rb" ) )
        close = data.xs('CLOSE', axis = 1, level = 1)
        close_= close[toTrade].dropna()

        if len(close_)<242:
            return 0
    if asset == 'Stock':
        data = pickle.load(open( "D:/PRISMO/historicalData/Data/asx200nobiasFINAL.pickle", "rb" ) )
        data = pickle.load(open('D:/PRISMO/historicalData/data/asx200nobiasFINAL.pickle','rb'))
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
        close_= close[toTrade].dropna()

        if len(close_)<242:
            return 0


    results=[]
    results.append(['delta','ve','cumret','sharpe','weightedCum','weightedSharpe'])
    ve_values = []
    delta_values = []
    for ve_ in range(1,10):
        order = round(0.1**(ve_+1), 12)
        for i in range(1,10):
            ve_values.append(round(i*order,12))
            delta_values.append(round(i*order,12))
        
    for delta in delta_values:
        for Ve in ve_values:
            close = close_
            #INIT
            yhatList = []
            QList= []
            eList=[]
            betaList = [np.matrix([[0],[0]])]
            R = np.zeros([len(toTrade),len(toTrade)])
            P = np.zeros([len(toTrade),len(toTrade)])
            y = []
            hedges = []
            Vw = delta/(1-delta)*np.diag(np.ones(len(toTrade)))

            for row in close.iterrows():
                #Extract x and y from the row, put them into numpy form. We include a constant for x so we can fit with a constant.
                x = np.matrix([[row[1][toTrade[0]]],[1]])
                y = np.matrix(row[1][toTrade[1]])

                ## 1 STEP AHEAD PREDICTION ##
                beta = betaList[-1]                                            # beta(t|t-1) = beta(t-1|t-1)
                R = P +Vw                                       #    R(t|t-1) = R(t-1|t-1) + Vw
                yhat = np.dot(x.T, beta)                                        #        yhat = x.beta
                e_t = y - yhat                                                  #        e(t) = y(t) - yhat(t)
                Q_t = np.dot( np.dot(x.T, R) , x ) +Ve               #        Q(t) = var(e(t)) = var(y(t) - yhat(t)) 
                #                                                                             = var(y(t)) + var(yhat(t)) + cov[y(t), yhat(t)]
                #                                                                             = x . R(t|t-1) + Ve


                ## UPDATE PARAMETERS ##
                K = np.dot(R, x) / Q_t                                                 # K is the kalman gain
                beta = beta + K*e_t                                                         # beta(t|t) = beta(t|t-1)+K(t)e(t)
                P = R - np.dot( np.dot(K, np.transpose(x)), R)               # We denote R(t|t) by P, and R(t|t-1) as R. R(t | t − 1) = cov(β(t) − βhat(t|t-1))

                #Add beta and predicted y values to arrays for storage
                betaList.append(beta)
                yhatList.append(yhat)

                #Form the hedge ratio as a float
                hedgeRatio = beta[0].tolist()[0][0]
                hedges.append(hedgeRatio)
                eList.append(e_t.tolist()[0][0])
                QList.append(Q_t.tolist()[0][0])

            close.loc[slice(None),'e'] = eList
            close.loc[slice(None),'Q'] = QList

            if np.mean(close['e'])>np.sqrt(np.mean(close['Q'])):
#                 print('means',np.mean(close['e']), np.sqrt(np.mean(close['Q'])))
    #             if the average error is greater than the average deviation the parameters are fucked. No point going further
                break


            close.loc[slice(None),'LongEntry'] = close['e']<-np.sqrt(close['Q'])
            close.loc[slice(None),'LongExit'] = close['e']>-np.sqrt(close['Q'])
            close.loc[slice(None),'ShortEntry'] = close['e']>np.sqrt(close['Q'])
            close.loc[slice(None),'ShortExit'] = close['e']<np.sqrt(close['Q'])
            ratios = hedges
            numUnitsLong = np.array([[np.nan]]*len(close))
            numUnitsShort = np.array([[np.nan]]*len(close))

            numUnitsLong[0] = 1
            numUnitsLong[np.array(close['LongEntry'])] = 1
            numUnitsLong[np.array(close['LongExit'])] = 0

            numUnitsShort[0] = 1
            numUnitsShort[np.array(close['ShortEntry'])] = -1
            numUnitsShort[np.array(close['ShortExit'])] = 0

            close.loc[slice(None),'short'] = numUnitsShort
            close.loc[slice(None),'short'].fillna('ffill')
            close.loc[slice(None),'long'] =  numUnitsLong
            close.loc[slice(None),'long'].fillna('ffill')

            close.loc[slice(None),'numUnits'] = close['long'] + close['short']

            close.loc[slice(None),'numUnits'+toTrade[0]] = -close['numUnits']*ratios
            close.loc[slice(None),'numUnits'+toTrade[1]] = close['numUnits']

            close.loc[slice(None),'positions'+toTrade[0]] = close['numUnits'+toTrade[0]]*close[toTrade[0]]
            close.loc[slice(None),'positions'+toTrade[1]] = close['numUnits'+toTrade[1]]*close[toTrade[1]]

            close.loc[slice(None),'pnl'] = close[toTrade[0]].pct_change()*(close['positions'+toTrade[0]].shift(1)) + close[toTrade[1]].pct_change()*(close['positions'+toTrade[1]].shift(1))

            close.loc[slice(None),'returns'] = (close['pnl'])/(np.abs(close['positions'+toTrade[0]]) +np.abs(close['positions'+toTrade[0]])).shift(1)
           
            close.loc[slice(None),'returns']= close['returns'][close['returns']<0.1]

            sharpe = (np.mean(close['returns']))
            apr = APR(close['returns'])

            try:
                numTrades = len(close[close['numUnits']!=0])/2
                if numTrades > 50:
                    monotonicity = monotonicityMeasure(np.cumsum(close['returns'].fillna(0)))
                    weightedSharpe = monotonicity*sharpe
                    weightedRet = monotonicity*np.cumsum(close['returns'].fillna(0)).iloc[-1]
                    results.append([delta, Ve,np.cumsum(close['returns'].fillna(0)).iloc[-1], sharpe, weightedRet, weightedSharpe, apr, apr*monotonicity])
            
            except Exception as e:
                print(e)
    pickle.dump(results, open('C:/Users/Billy/Documents/PRISMO/Backtester/optimizers/cointegration/results/'+flatten(toTrade)+'_openClose.pickle','wb'))


if __name__ == '__main__':

    
    for i in range(0,len(pairs),8):
        processes = []
        for index,pair in enumerate(pairs[i:i+8]):
            print(i+index)
            process = Process(target=optimize, args=([pair]))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()



