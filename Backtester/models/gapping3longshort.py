import pandas as pd
from pandas_datareader import data as pdr
import sys
sys.path.append('C:/Users/Billy/Documents/PRISMO/Backtester')
from base_model_backtest import backtest
import datetime
import os
from tqdm import tqdm_notebook
import time
import numpy as np
import pickle
from itertools import compress
import matplotlib.pyplot as plt


class gapping_3(backtest):
    def __init__(self, 
                timeSeriesClose, 
                timeSeriesLow,
                timeSeriesOpen,
                inIndex,
                toTrade, 
                shape, 
                printBool, 
                plotBool,
                transactionFeesPercent,
                minimumComission ,
                InitialCapital,
                capitalPerTrade,
                decimalPlaces = 10,
                formationPeriod = None, 
                logDir = None,
                stdLookback = 90,
                ZscoreShort = 1,
                ZscoreLong = 1,
                quantile = 10
                ):
        # Feeds the selected variables into the __init__ function in the base bactesting model
        # Order of variables is the order they'll be fed into backtest
        super().__init__(timeSeriesClose, 
                toTrade, 
                shape, 
                printBool, 
                plotBool,
                decimalPlaces,
                transactionFeesPercent =transactionFeesPercent,
                minimumComission = minimumComission,
                InitialCapital = InitialCapital,
                logDir=logDir)
        
        
        self.directory = logDir            
        self.features = toTrade    
        #Decimal Places should be assigned globally for transparency
        self.decimalPlaces = decimalPlaces
        
        
        
        
        self.timeSeriesClose = timeSeriesClose
        self.timeSeriesOpen = timeSeriesOpen
        self.timeSeriesLow = timeSeriesLow
        



        self.indexes1 = pd.MultiIndex.from_product([['returns', 'returnsAve20Days'],list(self.features)])
        self.indexes2 = pd.MultiIndex.from_product([['returns', 'returnsStd'+str(stdLookback)+'Days'],list(self.features)])

        
        #Store all returns values
        self.returnsDf = pd.DataFrame(index = self.indexes1)
        self.returnsLowOpenDf = pd.DataFrame(index = self.indexes2)
        self.capitalPerTrade = capitalPerTrade
        
        #Blank template for new row
        self.newRow = pd.DataFrame(index = self.indexes1)

        
        self.stdLookback = stdLookback 
        self.ZscoreShort = ZscoreShort
        self.ZscoreLong = ZscoreLong
        
        
        self.quantile = quantile


        self.inIndexDf = inIndex
        self.featuresInIndex = None

    def run(self):

        for index, row in enumerate(self.timeSeriesClose.itertuples()):
            #Close prices in form [date, asset1, asset2, ...]

            closePrices = row
            self.featuresInIndex = [x for x in compress(self.features, self.inIndexDf.iloc[index].tolist())]

            #Close prices in form [date, asset1, asset2, ...]
            openPrices = [pd.DataFrame(self.timeSeriesOpen.iloc[index]).columns[0]] + self.timeSeriesOpen.iloc[index].tolist()

            if index > 1:
                yesterdaysLowPrices = [pd.DataFrame(self.timeSeriesLow.iloc[index-1]).columns[0]] + self.timeSeriesLow.iloc[index-1].tolist()
            else:
                yesterdaysLowPrices = None

            #Create array of open orders to pass into strategy
            if self.orderbook is not None and len(self.orderbook)>0:
                openOrderBook = self.orderbook.loc[self.orderbook['openClosed']=='OPEN']  
            else:
                openOrderBook = None

            #Generate orders
            orders = self.generate_signals(closePrices, openPrices, yesterdaysLowPrices, openOrderBook)



           



    def generate_signals(self, closePriceRow, openPriceRow, yesterdaysLowPriceRow, openOrderBook, closeAll = False):
        
        
        
        ####START OF DAY####
        
        orders = [] 
        date = closePriceRow[0]
        yesterdaysDate = date - pd.DateOffset(1)
    
        #Update price history with the openPrices, as if its a new trading day and all we know is the open prices
        if closeAll != True:
            if self.dfOpenHistory is None:
                self.dfOpenHistory = pd.DataFrame([openPriceRow])
                self.dfOpenHistory.columns = ['Date'] + self.features
                self.dfOpenHistory.set_index(['Date'], inplace=True)
            else:
                temp = pd.DataFrame([openPriceRow])
                temp.columns = ['Date'] + self.features
                temp.set_index(['Date'], inplace=True)
                self.dfOpenHistory = pd.concat([self.dfOpenHistory, temp]) 
                
            if yesterdaysLowPriceRow is not None:
                if self.dfLowHistory is None:
                    self.dfLowHistory = pd.DataFrame([yesterdaysLowPriceRow])
                    self.dfLowHistory.columns = ['Date'] + self.features
                    self.dfLowHistory.set_index(['Date'], inplace=True)
                else:
                    temp = pd.DataFrame([yesterdaysLowPriceRow])
                    temp.columns = ['Date'] + self.features
                    temp.set_index(['Date'], inplace=True)
                    self.dfLowHistory = pd.concat([self.dfLowHistory, temp]) 
        
    
        #compute the returns from yesterdays low to today's open
        if yesterdaysLowPriceRow is not None:
            returnsLowOpen = [openPrice/yesterdaysLowPriceRow[indexLow+1]-1 for indexLow, openPrice in enumerate(openPriceRow[1:])]
            self.returnsLowOpenDf.loc[('returns',slice(None)), date] = returnsLowOpen
            
            
            
        #compute the returns for the close prices and the standard deviation over the past 90 days
        if self.dfHistory is not None and len(self.dfHistory)>2:
            
            self.returnsDf.loc[('returns',slice(None)), yesterdaysDate] =(self.dfHistory.iloc[-1]/self.dfHistory.iloc[-2]-1).values
        
            #Average of values of returns (close prices) for the last 20 days
            if len(self.returnsDf.columns)>20:
                self.returnsDf.loc[('returnsAve20Days',slice(None)), yesterdaysDate] = self.returnsDf.T.returns.rolling(20).mean().iloc[-1].values
            else:
                self.returnsDf.loc[('returnsAve20Days',slice(None)), yesterdaysDate] = self.returnsDf.T.returns.rolling(len(self.returnsDf.columns)).mean().iloc[-1].values
            
            
            
        #Standard Deviation of returns (Low-Open) for the last 90 days
        if self.returnsLowOpenDf is not None and len(self.returnsLowOpenDf.columns)>1:
            if len(self.returnsDf.columns)>self.stdLookback:
                self.returnsLowOpenDf.loc[('returnsStd'+str(self.stdLookback)+'Days',slice(None)), date] = self.returnsLowOpenDf.T.returns.rolling(self.stdLookback).std().iloc[-1].values
            else:
                self.returnsLowOpenDf.loc[('returnsStd'+str(self.stdLookback)+'Days',slice(None)), date] = self.returnsLowOpenDf.T.returns.rolling(len(self.returnsLowOpenDf.columns)).std().iloc[-1].values
        
        
        
        
        #SHORT ORDERS
        
        
        # select stocks whose returns (from yesterdays low to todays open) are greater than one standard deviation (calculated above)
        if self.returnsLowOpenDf is not None and len(self.returnsLowOpenDf.columns)>1:
            gappedStocks = self.returnsLowOpenDf.T.loc[date].returns > self.ZscoreShort*self.returnsLowOpenDf.T.loc[date]['returnsStd'+str(self.stdLookback)+'Days']
            gappedStocks = gappedStocks[gappedStocks==True].index
            
            #Filter down by stocks that are actually in the index
            gappedStocks = gappedStocks[[x in self.featuresInIndex for x in gappedStocks]]

        gappedAndOpenBELOWAve = None
        if self.dfHistory is not None and self.dfOpenHistory is not None and len(self.dfHistory)>20:
            # narrow this list by requiring their open prices to be LOWER than the 20 day moving average
            aveClose =  self.dfHistory.rolling(20).mean().T.loc[gappedStocks].T.iloc[-1]
            todayOpen = self.dfOpenHistory.T.loc[gappedStocks].T.loc[date]

            gappedAndOpenBELOWAve = todayOpen < aveClose
            gappedAndOpenBELOWAve = gappedAndOpenBELOWAve[gappedAndOpenBELOWAve==True].index
        

        shortIndex = None
        #Buy the 10 stocks that have the most negative returns from the previous days low to current open.
        if gappedAndOpenBELOWAve is not None:
            if len(gappedAndOpenBELOWAve) <=10:
                shortIndex = gappedAndOpenBELOWAve
            else:
                shortIndex = self.returnsLowOpenDf.dropna().loc[('returns',gappedAndOpenBELOWAve), date].sort_values().iloc[:self.quantile].index.codes[1]
                shortIndex = self.returnsLowOpenDf.index.levels[1][shortIndex]
        
              
        #SHORT STONKS IN THA LIST
        if shortIndex is not None:
            for shortTick in shortIndex:
                price = self.dfOpenHistory[shortTick].iloc[-1]
                quantity = -float(self.capitalPerTrade/price)
                if np.isnan(price) or np.isnan(quantity):
                        pass
                else:
                    orders.append([None, 'SELL',  shortTick,  quantity, price, date, 'OPEN'])
        
        
        
        
        
        #LONG ORDERS
        
        
        
         # select stocks whose returns (from yesterdays low to todays open) are less than one standard deviation (calculated above)
        if self.returnsLowOpenDf is not None and len(self.returnsLowOpenDf.columns)>1:
            gappedStocks = self.returnsLowOpenDf.T.loc[date].returns< -self.ZscoreLong*self.returnsLowOpenDf.T.loc[date]['returnsStd'+str(self.stdLookback)+'Days']
            gappedStocks = gappedStocks[gappedStocks==True].index
            
            #Filter down by stocks that are actually in the index
            gappedStocks = gappedStocks[[x in self.featuresInIndex for x in gappedStocks]]


        gappedAndOpenAboveAve = None
        if self.dfHistory is not None and self.dfOpenHistory is not None and len(self.dfHistory)>20:
            # narrow this list by requiring their open prices to be higher than the 20 day moving average
            aveClose =  self.dfHistory.rolling(20).mean().T.loc[gappedStocks].T.iloc[-1]
            todayOpen = self.dfOpenHistory.T.loc[gappedStocks].T.loc[date]

            gappedAndOpenAboveAve = todayOpen>aveClose
            gappedAndOpenAboveAve = gappedAndOpenAboveAve[gappedAndOpenAboveAve==True].index
        

        buyIndex = None
        #Buy the 10 stocks that have the most negative returns from the previous days low to current open.
        if gappedAndOpenAboveAve is not None:
            if len(gappedAndOpenAboveAve) <=10:
                buyIndex = gappedAndOpenAboveAve
            else:
                buyIndex = self.returnsLowOpenDf.dropna().loc[('returns',gappedAndOpenAboveAve), date].sort_values().iloc[-self.quantile:].index.codes[1]
                buyIndex = self.returnsLowOpenDf.index.levels[1][buyIndex]

        
        
        
        #BUY STONKS IN THA LIST
        if buyIndex is not None:
            for buyTick in buyIndex:
                price = self.dfOpenHistory[buyTick].iloc[-1]
                quantity = float(self.capitalPerTrade/price)
                if np.isnan(price) or np.isnan(quantity):
                        pass
                else:
                    orders.append([None, 'BUY',  buyTick,  quantity, price, date, 'OPEN'])

        #Tick orders here. This is necessary as we want to update the open order book so we can close orders at the market close price
        self.tick(closePriceRow, orders)
        
# #         PLOTTING FOR VERIFICATION
#         if len(orders)>0:
#                 for order in orders:
#                     assetSample = order[2]
#                     print(assetSample)
# #                     print(self.returnsLowOpenDf.T.loc[date]['returnsStd'+str(self.stdLookback)+'Days'][gappedStocks])
# #                     print(self.returnsLowOpenDf.T.loc[date].returns[gappedStocks])
# #                     print(self.returnsLowOpenDf.T.loc[date].returns[gappedStocks]<-self.returnsLowOpenDf.T.loc[date]['returnsStd'+str(self.stdLookback)+'Days'][gappedStocks])
#                     lowPrices = self.dfLowHistory[assetSample].iloc[-20:]
#                     openPrices = self.dfOpenHistory[assetSample].iloc[-20:]
#                     plt.plot(lowPrices)
#                     plt.plot(openPrices)
#                     plt.legend(['low','open'])
#                     plt.show()

        ##### TO DELETE #####
        #Help out the thicc memory issues 
        if len(self.returnsDf)>self.stdLookback:
            self.returnsDf = self.returnsDf[self.returnsDf.columns[-(self.stdLookback+1):]]
            self.dfOpenHistory = self.dfOpenHistory.iloc[-(self.stdLookback+1):]
            try:
                self.orderbook = self.orderbook.loc[self.orderbook.openClosed == 'OPEN']
            except Exception as e:
                pass
        
        
        
        
         ####END OF DAY####
            
    
        #Create array of open orders
        if self.orderbook is not None and len(self.orderbook)>0:
            openOrderBook = self.orderbook.loc[self.orderbook['openClosed']=='OPEN']  
        else:
            openOrderBook = None
        
    
    
        
        #Create the orders that we close.
        orders = []
        
        # Close All is true on the last iteration of the loop to close all open trades. Don't want to double-add prices to df histories
        if closeAll != True:
             #Update price history with the close prices
            if self.dfOpenHistory is None:
                self.dfHistory = pd.DataFrame([closePriceRow])
                self.dfHistory.columns = ['Date'] + self.features
                self.dfHistory.set_index(['Date'], inplace=True)
            else:
                temp = pd.DataFrame([closePriceRow])
                temp.columns = ['Date'] + self.features
                temp.set_index(['Date'], inplace=True)
                self.dfHistory = pd.concat([self.dfHistory, temp])  
                
        
        #Sell all stonks that are open
        #Close all open orders on each tick
        if openOrderBook is not None:
            for index, openOrder in openOrderBook.iterrows():
                
                signal = openOrder[1]
                asset = openOrder[2]
                oldDate = openOrder[5]
                quantityTrade = round(openOrder[3], self.decimalPlaces)
                price = round(openOrder[4], self.decimalPlaces)
                orderID = openOrder[0]
                newDate = closePriceRow[0]
                newOrder = None     


                if signal == 'SELL':
                    newSignal = 'BUY'
                    newQuantityTrade = -quantityTrade
                    newPrice =  self.dfHistory[asset].iloc[-1]

                    if np.isnan(newPrice) or np.isnan(newQuantityTrade):
                        pass
                    else:
                        #['ID','BUY/SELL','asset','quantity','price','date','OPEN/CLOSED']
                        newOrder = [orderID, newSignal, asset, newQuantityTrade, newPrice, newDate,'CLOSED']
                        orders.append(newOrder)

                if signal == 'BUY':
                    newSignal = 'SELL'
                    newQuantityTrade = -quantityTrade
                    newPrice =  self.dfHistory[asset].iloc[-1]

                    if np.isnan(newPrice) or np.isnan(newQuantityTrade):
                        pass
                    else:
                        #['ID','BUY/SELL','asset','quantity','price','date','OPEN/CLOSED']
                        newOrder = [orderID, newSignal, asset, newQuantityTrade, newPrice, newDate,'CLOSED']
                        orders.append(newOrder)
                        
        #Close orders
        self.tick(closePriceRow, orders)