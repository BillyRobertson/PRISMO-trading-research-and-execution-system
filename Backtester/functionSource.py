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

from models.Cointegrated_Pairs_1 import cointegrated_pairs_1

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def flatten(list_):
    string = ''
    for element in list_:
        string = string + element.replace('.','')+'_'

    return string

def drawdowns(returns):
    maxVal = None
    cumulativeReturns = []
    drawdownArray = []

    for retVal in returns:
        #Update cumulativeReturns
        if len(cumulativeReturns)>0:
            cumulativeReturns.append(retVal+cumulativeReturns[-1])
        else:
            cumulativeReturns.append(retVal)

        if maxVal is None:
            maxVal = retVal
            drawdownArray.append(0)
        elif cumulativeReturns[-1] > maxVal:
            maxVal=cumulativeReturns[-1]
            drawdownArray.append(0)

        elif cumulativeReturns[-1]<=maxVal:
            drawdownArray.append(drawdownArray[-1]+retVal)


    drawDownDf = pd.DataFrame(drawdownArray)    

    drawDownDf['Date'] = returns.index        
    drawDownDf.set_index(['Date'],inplace=True)

    return drawdownArray, drawDownDf


def plotResults(Dates,
                Returns, 
                cumulativeCapital, 
                riskFreeTs, 
                directory,
                drawdownArray, 
                maxDrawDown = 0,
                maxDrawDownDuration = 0,
                alpha = 0,
                beta = 0,
                sharpe = 0,
                APR = 0,
                additionalData = []):

    plt.style.use('ggplot')


    ReturnsHist = Returns[Returns.returns!=0]
    CumulativeReturns = np.cumsum(Returns)

    
    retHist = [x for x in Returns if x != 0]
    cumulativeCapital = cumulativeCapital.T.reindex(Dates, fill_value=0)
    cumulativeCapital = np.cumsum(cumulativeCapital.values)

    riskFree = pdr.get_data_yahoo(riskFreeTs , interval = 'd')
    riskFree = riskFree['Adj Close'].pct_change()
    riskFreeReturns = riskFree[Dates].dropna()
    riskFreeReturns= riskFreeReturns.reindex(Dates, fill_value=0)
    cumulativeRiskFreeReturns = np.cumsum(riskFreeReturns)


    fig = plt.figure(figsize=(10,10))

    gs = fig.add_gridspec(3,4)

    #Plots for the returns
    returnsCumPlot = fig.add_subplot(gs[0, :-1])
#     returnsCumPlot.set_title('Cumulative Returns')
    returnsCumPlot.set_xlabel('Time')
    returnsCumPlot.set_ylabel('Cumulative Returns')
    returnsCumPlot.plot(Dates, CumulativeReturns)
    returnsCumPlot.plot(Dates, cumulativeRiskFreeReturns)
    returnsCumPlot.legend(['Strategy Returns', 'AXJO Returns'])

    #Plots for the returns
    returnsPlot = fig.add_subplot(gs[1, :-1])
#     returnsPlot.set_title('Returns (strategy only)')
    returnsPlot.set_xlabel('Time')
    returnsPlot.set_ylabel('Returns')
    returnsPlot.plot(Dates, Returns)
    returnsPlot.legend(['Strategy Returns'])


    #Plots for drawdown
    drawdownPlot = fig.add_subplot(gs[2, :-1])
#     drawdownPlot.set_title('Drawdowns')
    drawdownPlot.set_xlabel('Time')
    drawdownPlot.set_ylabel('Drawdowns')
    drawdownPlot.plot(Dates, drawdownArray ,color='orange')
    drawdownPlot.legend(['Strategy Drawdown'])
    
    resultsTable = fig.add_subplot(gs[0, -1:])
    tableContents = [['ALPHA',str(alpha)],
                     ['BETA',str(beta)],
                     ['APR%',str(APR)],
                     ['SHAR',str(sharpe)],
                     ['MAX.DDOWN',str(maxDrawDown)],
                     ['M.DDWN.',str(maxDrawDownDuration)+' days']]
    
    tableContents = tableContents + additionalData
    
    resultsTable.table(cellText=tableContents, loc='center',colWidths = [0.5,0.5]).scale(1,2)
    resultsTable.axis('off')
    
    fig.savefig(directory+'/results.jpg')   # save the figure to file
    
def drawdowns(returns):
    maxVal = None
    cumulativeReturns = []
    drawdownArray = []

    for retVal in returns:
        #Update cumulativeReturns
        if len(cumulativeReturns)>0:
            cumulativeReturns.append(retVal+cumulativeReturns[-1])
        else:
            cumulativeReturns.append(retVal)

        if maxVal is None:
            maxVal = retVal
            drawdownArray.append(0)
        elif cumulativeReturns[-1] > maxVal:
            maxVal=cumulativeReturns[-1]
            drawdownArray.append(0)

        elif cumulativeReturns[-1]<=maxVal:
            drawdownArray.append(drawdownArray[-1]+retVal)


    drawDownDf = pd.DataFrame()    
    drawDownDf['drawdown'] = drawdownArray
    drawDownDf['Date'] = returns.index        
    drawDownDf.set_index(['Date'],inplace=True)

    return drawdownArray, drawDownDf

def drawDownLength(drawDownDf):
    minDrawDownDate = min(drawDownDf[drawDownDf.drawdown == min(drawDownDf.drawdown)].index)
    drawDownStart = max(drawDownDf[drawDownDf.index < minDrawDownDate][drawDownDf[drawDownDf.index <= minDrawDownDate]==0].dropna().index)
    if len(drawDownDf[drawDownDf.index >= minDrawDownDate][drawDownDf[drawDownDf.index >= minDrawDownDate]==0].dropna().index)==0:
        drawDownEnd = drawDownDf.index[-1]
    else:
        drawDownEnd = min(drawDownDf[drawDownDf.index >= minDrawDownDate][drawDownDf[drawDownDf.index >= minDrawDownDate]==0].dropna().index)
    
    drawDownLen = str(drawDownEnd-drawDownStart)[:-14]
    return drawDownLen

def sharpe(returns, riskFreeTs):
    sharpes = []
    
    riskFree = pdr.get_data_yahoo(riskFreeTs , interval = 'd')
    riskFree = riskFree['Adj Close'].pct_change()
    
    for year in range(returns.index[0].year, returns.index[-1].year+1):
        annualReturns = returns.loc[returns.index<str(year+1)+'-01-01']
        annualReturns = annualReturns.loc[annualReturns.index>=str(year)+'-01-01']
        annualReturnsDropZero = annualReturns[annualReturns!=0].dropna()
        datesTraded = annualReturns.index

        riskFreeReturns = riskFree.reindex(datesTraded, fill_value=0).dropna()
        meanriskFreeReturns = np.mean(riskFreeReturns)
        
        daysTraded = len(annualReturnsDropZero)

        aveReturns = np.mean(annualReturnsDropZero) 
        stdReturns = np.std(annualReturnsDropZero)
        sharpes.append((aveReturns-meanriskFreeReturns)/stdReturns*np.sqrt(daysTraded))

    return np.mean(sharpes)

def APR(returns):
    ave = []
    for year in range(returns.index[0].year, returns.index[-1].year+1):
        annualReturns = returns.loc[returns.index<str(year+1)+'-01-01']
        annualReturns = annualReturns.loc[annualReturns.index>=str(year)+'-01-01']

        aveReturns = np.mean(annualReturns) 
        ave.append(aveReturns*252*100)
        

    return np.mean(ave)

def alpha_beta(returns, riskFreeTs):
    import statsmodels.api as sm

    riskFree = pdr.get_data_yahoo(riskFreeTs , interval = 'd')
    riskFree = riskFree['Adj Close'].pct_change()

    riskFreeReturns = riskFree.reindex(returns.index, fill_value=0).dropna()

    regression = sm.OLS(returns.returns, sm.add_constant(riskFreeReturns))
    results = regression.fit()

    beta = results.params[1]
    alpha = results.params[0]
    
    return alpha, beta

