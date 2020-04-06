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


class momentum_1(backtest):
    def __init__(self, 
                timeSeries, 
                inIndex,
                toTrade, 
                shape, 
                printBool, 
                plotBool,
                decimalPlaces = 10,
                formationPeriod = None, 
                logDir = None,
                capitalPerTrade = 100,
                decileQuantity = 10,
                holdingPeriod = 1,
                timeSeriesOpen = None,
                simeltaneouslyOpenTrades = 1):

        super().__init__(timeSeries, 
                toTrade, 
                shape, 
                printBool, 
                plotBool,
                decimalPlaces,
                lookback=formationPeriod,
                logDir=logDir)
        
        
        self.directory = logDir            
        self.features = toTrade    
        #Decimal Places should be assigned globally for transparency
        self.decimalPlaces = decimalPlaces
        self.timeSeries = timeSeries
        
        self.lookbackStd = 10
        self.lookbackMean = 10


        self.holdingPeriod = holdingPeriod
        self.T = formationPeriod #formation period
        self.decileQuantity = decileQuantity
        self.simeltaneouslyOpenTrades = simeltaneouslyOpenTrades  #number of simeltaneous trades, per asset
        self.indexes = pd.MultiIndex.from_product([['returns','returnsCum','returnsMean','returnsStd','returnsRiskAdj'],list(self.features)])
        
        
        #Store all returns values
        self.returnsDf = pd.DataFrame(index = self.indexes)
        
        
        self.capitalPerTrade = capitalPerTrade
        
        #Blank template for new row
        self.newRow = pd.DataFrame(index = self.indexes)

        self.openPrices = timeSeriesOpen

        self.inIndexDf = inIndex
        self.featuresInIndex = None
        

    def run(self):
        # pls dont bully me for using a counter as an index, the enumerate function outpits price as a row not a pandas series F
        for index, row in enumerate(self.timeSeries.itertuples()):

            self.featuresInIndex = [x for x in compress(self.features, self.inIndexDf.iloc[index].tolist())]

             #Stupid but this creates a list of [date, price1, price2, ...]
            
            closePrices = row
            openPrices = [pd.DataFrame(self.openPrices.iloc[index]).columns[0]] + self.openPrices.iloc[index].tolist()
            
            #Create array of open orders to pass into strategy
            if self.orderbook is not None and len(self.orderbook)>0:
                openOrderBook = self.orderbook.loc[self.orderbook['openClosed']=='OPEN']  
            else:
                openOrderBook = None
                
#             print(date)
            
            orders = self.generate_signals(closePrices, openPrices, openOrderBook)
#             print(orders)
            
            self.tick(closePrices, orders)




        #Close all remaining orders
        #Create array of open orders to pass into strategy
        if self.orderbook is not None and len(self.orderbook)>0:
            openOrderBook = self.orderbook.loc[self.orderbook['openClosed']=='OPEN']  
        else:
            openOrderBook = None

        print(len(openOrderBook))
        orders = self.generate_signals(closePrices, openPrices, openOrderBook, closeAll = True)
        self.tick(closePrices, orders, closeAll = True)
            
    def generate_signals(self, closePriceRow, openPriceRow, openOrderBook, closeAll = False):

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
        
        
        #Calculate returns values based on yesterdays prices and those preceeding that.
        if len(self.returnsDf.columns)==0 or len(self.dfHistory)<=2:
            self.returnsDf.loc[('returns',slice(None)), yesterdaysDate] = 0
            self.returnsDf.loc[('returnsCum',slice(None)), yesterdaysDate] = 0
            self.returnsDf.loc[('returnsMean',slice(None)), yesterdaysDate] = 0
            self.returnsDf.loc[('returnsStd',slice(None)), yesterdaysDate] = 0
            self.returnsDf.loc[('returnsRiskAdj',slice(None)), yesterdaysDate] = 0
            
        else:
            
            if len(self.dfHistory)>2:
                # Returns = P(final)/P(initial)-1
                self.returnsDf.loc[('returns',slice(None)), yesterdaysDate] =(self.dfHistory.iloc[-1]/self.dfHistory.iloc[-2]-1).values
                
                # CumulativeReturns = P(final)/P(final-T) - 1
                if len(self.returnsDf.columns)>self.T:
                    self.returnsDf.loc[('returnsCum',slice(None)), yesterdaysDate] = (self.dfHistory.iloc[-1]/self.dfHistory.iloc[-(self.T)]-1).values
                else:
                    self.returnsDf.loc[('returnsCum',slice(None)), yesterdaysDate] = 0
                    
                #Average of values of returns    
                if len(self.returnsDf.columns)>self.T:
                    self.returnsDf.loc[('returnsMean',slice(None)), yesterdaysDate] = self.returnsDf.T.returns.rolling(self.T).mean().iloc[-1].values
                else:
                    self.returnsDf.loc[('returnsMean',slice(None)), yesterdaysDate] = self.returnsDf.T.returns.rolling(len(self.returnsDf.columns)).mean().iloc[-1].values
                
                #Standard Deviation of returns
                if len(self.returnsDf.columns)>self.T:
                    self.returnsDf.loc[('returnsStd',slice(None)), yesterdaysDate] = self.returnsDf.T.returns.rolling(self.T).std().iloc[-1].values
                else:
                    self.returnsDf.loc[('returnsStd',slice(None)), yesterdaysDate] = self.returnsDf.T.returns.rolling(len(self.returnsDf.columns)).std().iloc[-1].values
                
                #Risk Adjusted Returns
                meanRet = self.returnsDf.T.returnsMean.iloc[-1]
                stdRet = self.returnsDf.T.returnsStd.iloc[-1]
                riskAdjRet = meanRet/stdRet
                self.returnsDf.loc[('returnsRiskAdj',slice(None)), yesterdaysDate] = riskAdjRet.values
            



        ##### TO DELETE #####
        #Help out the thicc memory issues 
        if len(self.returnsDf)>self.T:
            self.returnsDf = self.returnsDf[self.returnsDf.columns[-(self.T+1):]]
            self.dfOpenHistory = self.dfOpenHistory.iloc[-(self.T+1):]
            try:
                self.orderbook = self.orderbook.loc[self.orderbook.openClosed == 'OPEN']
            except Exception as e:
                pass

#         print(self.dfHistory, self.dfHistory)
            

        # print(date, yesterdaysDate, self.returnsDf.loc['returnsRiskAdj'])

        # Seperate into quantiles
    
        codesTop = (self.returnsDf.dropna().loc[('returnsRiskAdj',self.featuresInIndex), yesterdaysDate].sort_values()).iloc[-self.decileQuantity:].index.codes[1]
        codesBottom = (self.returnsDf.dropna().loc[('returnsRiskAdj',self.featuresInIndex), yesterdaysDate].sort_values()).iloc[:self.decileQuantity].index.codes[1]
        
#         print(self.returnsDf.dropna().loc[('returnsRiskAdj',slice(None)), date].sort_values())
#         print(self.returnsDf.index.levels[1][codesTop])
#         print(self.returnsDf.index.levels[1][codesBottom])
        




        #We should close all orders on the close price

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
                

                 #IF: date is >=  than buying date + holding period
                if (signal == 'SELL'and newDate >= oldDate + pd.DateOffset(self.holdingPeriod)) or closeAll == True:
                    newSignal = 'BUY'
                    newQuantityTrade = -quantityTrade
                    newPrice =  self.dfHistory[asset].iloc[-1]
                    
                    if np.isnan(newPrice):
                        pass
                    else:
                        #['ID','BUY/SELL','asset','quantity','price','date','OPEN/CLOSED']
                        newOrder = [orderID, newSignal, asset, newQuantityTrade, newPrice, newDate,'CLOSED']
                        orders.append(newOrder)

                if (signal == 'BUY' and newDate >= oldDate + pd.DateOffset(self.holdingPeriod)) or closeAll == True:
                    newSignal = 'SELL'
                    newQuantityTrade = -quantityTrade
                    newPrice =  self.dfHistory[asset].iloc[-1]

                    
                    if np.isnan(newPrice):
                        pass
                    else:
                        #['ID','BUY/SELL','asset','quantity','price','date','OPEN/CLOSED']
                        newOrder = [orderID, newSignal, asset, newQuantityTrade, newPrice, newDate,'CLOSED']
                        orders.append(newOrder)



        # #TEST: only want to buy stocks when there are no currently open orders.
        openOrders = False
        if (openOrderBook is not None )  and len(orders)+len(openOrderBook)>=self.decileQuantity*(self.simeltaneouslyOpenTrades+1):
            openOrders = True


        #It's not good enough to just filter 'if stock price is in index'. This results in trades not occurring at a all.
        if len(self.returnsDf.columns)>self.T and closeAll != True and openOrders == False:

            # Create buy orders for stocks in top quantile for risk adj. returns, and create short orders for the others
            for topQuantileTick in list(self.returnsDf.index.levels[1][codesTop]):
                price = self.dfOpenHistory[topQuantileTick].iloc[-1]


                #CTEST: heck if the asset is in the open order book. DOn't buy if so
                tickOpen = False
                # if openOrderBook is not None:
                #     if topQuantileTick in list(openOrderBook.asset):
                #         tickOpen = True



                #topQuantileTick not in list(openOrderBook.asset) implies there are no currently open orders for the said asset.
                if not np.isnan(price) and topQuantileTick and tickOpen == False:
                    quantity = float(self.capitalPerTrade/price)
                    orders.append([None, 'BUY',  topQuantileTick,  quantity, price, date, 'OPEN'])

            for bottomQuantileTick in list(self.returnsDf.index.levels[1][codesBottom]):

                price = self.dfOpenHistory[bottomQuantileTick].iloc[-1]

                 #CTEST: heck if the asset is in the open order book. DOn't buy if so
                tickOpen = False
                # if openOrderBook is not None:
                #     if topQuantileTick in list(openOrderBook.asset):
                #         tickOpen = True


                if not np.isnan(price) and tickOpen == False:
                    quantity = float(-self.capitalPerTrade/price)
                    orders.append([None, 'SELL',  bottomQuantileTick,  quantity, price, date, 'OPEN'])



          #Update price history with the close prices
        if closeAll != True:

            if self.dfHistory is None:
                self.dfHistory = pd.DataFrame([closePriceRow])
                self.dfHistory.columns = ['Date'] + self.features
                self.dfHistory.set_index(['Date'], inplace=True)
            else:
                temp = pd.DataFrame([closePriceRow])
                temp.columns = ['Date'] + self.features
                temp.set_index(['Date'], inplace=True)
                self.dfHistory = pd.concat([self.dfHistory, temp])  
        

        return orders