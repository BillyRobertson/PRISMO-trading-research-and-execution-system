
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


class cointegrated_pairs_2(backtest):
    def __init__(self, 
                timeSeries, 
                toTrade, 
                shape, 
                printBool, 
                plotBool,
                decimalPlaces = 8,
                lookback = None, 
                alpha_ = 0.45,
                std_ratio = 1,
                logDir = None,
                riskFreeTs = None,
                transactionFeesPercent = 0.0008,
                minimumComission = 0,
                InitialCapital = 0,
                capitalPerTrade = 1,
                simOpen = 1):
        
        super().__init__(timeSeries, 
                        toTrade, 
                        shape, 
                        printBool, 
                        plotBool,
                        decimalPlaces,
                        lookback,
                        transactionFeesPercent,
                        minimumComission,
                        InitialCapital,
                        logDir)
        

        self.directory = logDir             # Directory for the logs
        self.features = toTrade             # Features being traded
        self.plot = plotBool                # Plots returns and bollinger entries every tick
        
        self.S1 = []
        self.S2 = []
        self.predictions = []
        self.true = []

        self.error = []
        self.std_array = []

        self.tau = 1
        self.simOpen = simOpen

        self.capitalPerTrade = capitalPerTrade
        #Decimal Places should be assigned globally for transparency
        self.decimalPlaces = decimalPlaces
        
        #Store the hedge ratios
        self.hedges = []


        self.alpha = alpha_
        self.std_ratio = std_ratio

        #The underlying time series, e.g. we use the snp500 for the ASX if we're testing aussie assets.
        self.riskFreeTs = riskFreeTs
        
        self.timeSeries = timeSeries

    # Iterate through each row of the time series dataframe. For each row, we want to update the openOrderBook for that timesetp, then generate orders using the generate signals function.
    # We then pass that through to the tick function, which "executes" the orders fed in. For a backtest, execution just appends profits to records and keeps track of all the changes made.
    def run(self):

        self.fig, self.axes = plt.subplots(nrows = 2, ncols = 1,sharex=True)

        for row in tqdm_notebook(self.timeSeries.iterrows()):

            #Stupid but this creates a list of [date, price1, price2, ...]
            date = row[0]
            prices = [row[1][feature] for feature in self.features]
            row = [date]
            row = row + prices
            
            # See what orders are currently OPEN and filter to a seperate dataframe.
            if self.orderbook is not None and len(self.orderbook)>0:
                openOrderBook = self.orderbook.loc[self.orderbook['openClosed']=='OPEN']  
            else:
                openOrderBook = None
            
                
            #Generate Orders, unique to the strategy
            orders = self.generateSignals(row, openOrderBook, self.portfolio)
            # print(orders)
            # Adjust the portfolio according to the generated orders
    

    
            
        
            
            
   

    def generateSignals(self, row, openOrderBook, currentPortfolio):

     # This function generates a list of orders based on the current and previous prices
        # prices = [date, feature1.price, feature2.price, feature3.price, ...]
        # orders = [[order1],[order2],[...],... ]
        #        =  [[int(ORDER_ID), str(BUY/SELL), str(fFEATURE_NAME), floata(QUANTITY), float(PRICE), unix(DATE), str(OPEN/CLOSED)], .....]
        #     e.g.  [[420, 'BUY', GOLD.AX, 1, 100, 1583987952, 'OPEN'],...]

        # order types can be partitioned into 2 classes: OPEN and CLOSED trades. 
        #           A OPEN trade is a trade that does not exist, and we create from scratch. A OPEN order remains OPEN until we close it.
        #           A CLOSED order is created by looking at all of the currently OPEN trades, and deciding whether or not to close the trade depending on the price and market conditions.
        #           If the OPEN order is a BUY, then we SELL the asset to close it.
        #           IF the OPEN order is a SELL, then we BUY it back to CLOSE it. When we open an order by selling, we're usually gonna be entering a short contract where we enter borrow money against the value 
        #           of the asset, and buy it back against the value of the same asset in the future. Essentially we're selling the stock without the need to actually own it, but we're obliged to buy it back in the future.

        # openOrderBook is a sub-dataFrame of the main orderBook dataframe, filtered down to just the currently OPEN orders.
        # currentPortfolio is the current portfolio lmao.

        #Update price history
        if self.dfHistory is None:
            self.dfHistory = pd.DataFrame([row])
            self.dfHistory.columns = ['Date'] + self.features
            self.dfHistory.set_index(['Date'], inplace=True)
        else:
            temp = pd.DataFrame([row])
            temp.columns = ['Date'] + self.features
            temp.set_index(['Date'], inplace=True)
            self.dfHistory = pd.concat([self.dfHistory, temp])                
        

        
        # Therefore:
        featureX = self.features[0]
        featureY = self.features[1]

        #Extract x and y from the row, put them into numpy form. We include a constant for x so we can fit with a constant.
        x = row[1]
        y = row[2]
        beta = y/x
        predictedSpread = None
        if len(self.predictions)>0:
                predictedSpread = y-self.predictions[-1]*x
                self.error.append(predictedSpread)
                self.std_array.append(np.std(self.error))
                self.true.append(beta)
        else:
            self.error.append(0)
            self.std_array.append(0)




        if len(self.S1)==0:
            self.S1.append(beta)
            self.S2.append(beta)
        else:
            self.S1.append(self.alpha*beta + (1-self.alpha)*self.S1[-1])
            self.S2.append(self.alpha*self.S1[-1] + (1-self.alpha)*self.S2[-1])

        prediction = (2+ self.alpha*self.tau/(1-self.alpha))*self.S1[-1] - (1+self.alpha*self.tau/(1-self.alpha))*self.S2[-1]
        self.predictions.append(prediction)  

        #PREDICTED SPREAD SHOULD = 0, since it uses the beta predicted foor the current time step.
        #IF > 0, we expect the value to come back down.


        #####CREATE CLOSING ORDERS####
        #All orders that are created are appended to this array. This array is returned at the end of the funciton.
        orders = []

        # If the openOrderBook is not empty, go through all of its rows and see if we should close each trade.
        if openOrderBook is not None:                                                   
            for index, openOrder in openOrderBook.iterrows():

                #Signal = BUY/SELL, asset = assetName, date = unix(date), quantityTrade = quantity bought/sold when the trade was opened
                signal = openOrder[1]
                asset = openOrder[2]
                date = openOrder[5]
                quantityTrade = openOrder[3]
                orderID = openOrder[0]

                # TODO: when we don't round to ~10 decimal places, we get wack float values. e.g. if we're trynna have a price of 1, it reads as 0.999999999999999999999998
                # I remember some computer science nerd telling me why this shit happens but i cant remember how to fix, so i just round to 10 decimal places for now.
                price = round(openOrder[4], self.decimalPlaces)

                
                newHedgedOrder = None  #x
                newOrder = None        #y
                

                # We look to see if the open order should be closed. 
                # We create the new orders only if the current order belongs to the asset corresponds to the 'y' feature in the kalman model.
                # We do this because it allows us to find the corresponding 'x' asset and close that trade too. It is necessary to close both trades in pairs trading.

                #buy back shorts if e(t) < sqrt(Q(t))
                if signal == 'SELL' and predictedSpread is not None and predictedSpread < self.std_array[-1]*self.std_ratio:
                    # If the asset corresponds t the 'y' feature = features[1]
                    if asset == self.features[1]:      
                        #Reverse the trade signal, quantity, use the current price and current date.      
                        newSignal = 'BUY'                           
                        newQuantityTrade = -quantityTrade
                        newPrice = row[self.features.index(asset)+1]
                        newDate = row[0]

                        #['ID','BUY/SELL','asset','quantity','price','date','OPEN/CLOSED']
                        newOrder = [orderID, newSignal, asset, newQuantityTrade, newPrice, newDate,'CLOSED']
                        orders.append(newOrder)

                        # FIND THE CORRESPONDING LONG TRADE CREATED TO HEDGE THE ABOVE SHORT
                        hedgedFeature = self.features[0]

                        # looks for the trade created on the same date with the hedged feature. This will correspond to the hedged trade.
                        # hedgedOrder = ['ID','BUY/SELL','asset','quantity','price','date','OPEN/CLOSED']
                        hedgedOrder = openOrderBook[(openOrderBook.asset == hedgedFeature) & (openOrderBook.date == date)]
                        hedgedOrder = list(hedgedOrder.loc[0])

                        #Invert the signal
                        oldHedgedSignal = hedgedOrder[1]
                        if oldHedgedSignal == 'BUY': 
                            newHedgedSignal = 'SELL'
                        elif oldHedgedSignal == 'SELL':
                            newHedgedSignal = 'BUY'

                        #Get the current price for the hedged asset
                        newHedgedPrice = row[self.features.index(hedgedFeature)+1]
                        
                        #Create the order
                        newHedgedOrder = [hedgedOrder[0], newHedgedSignal, hedgedFeature, - hedgedOrder[3], newHedgedPrice, newDate, 'CLOSED']
                        orders.append(newHedgedOrder)


                #Same as above, but the y asset is being sold to close the order.
                if signal == 'BUY' and predictedSpread is not None and predictedSpread > -self.std_array[-1]*self.std_ratio:
                    
                    if asset == self.features[1]:
                        newSignal = 'SELL'
                        newQuantityTrade = -quantityTrade
                        newPrice = row[self.features.index(asset)+1]
                        newDate = row[0]

                        newOrder = [orderID, newSignal, asset, newQuantityTrade,newPrice,newDate,'CLOSED']
                        orders.append(newOrder)
                        
                        
                        hedgedFeature = self.features[0]
                        hedgedOrder = openOrderBook[(openOrderBook.asset == hedgedFeature) & (openOrderBook.date == date)]

                        hedgedOrder = list(hedgedOrder.loc[0])

                        oldHedgedSignal = hedgedOrder[1]
                        if oldHedgedSignal == 'BUY': 
                            newHedgedSignal = 'SELL'
                        elif oldHedgedSignal == 'SELL':
                            newHedgedSignal = 'BUY'
                        newHedgedPrice = row[self.features.index(hedgedFeature)+1]
                        
                        newHedgedOrder = [hedgedOrder[0], newHedgedSignal, hedgedFeature, - hedgedOrder[3], newHedgedPrice, newDate, 'CLOSED']
                        orders.append(newHedgedOrder)
                        


        
        
        self.tick(row, orders)



        if self.orderbook is not None and len(self.orderbook)>0:
                openOrderBook = self.orderbook.loc[self.orderbook['openClosed']=='OPEN']  
        else:
            openOrderBook = None
    
        orders = []

        #####OPENING NEW ORDERS####
        # We only want to open new trades if there are no currently open trades. 
        # since we're predicting y = b . x  + e, then y - b.x = e, for e distributed randomly about mean zero, so for every +1 unit of y, we want to own -b of x and visa versa.

        # quantity = min(self.capitalPerTrade/row[1],self.capitalPerTrade/row[2]) 

        #Quantity is the base quantity we will trade (if we trade 1*quantity of y, we hedge against hedgeRatio*quantity of x)

        # quantity = 1000

        #If we want the total value of our investment to = self.capitalPerTrade, then we can set the quantities traded        x = y
        #                                                                                                            p1*x+ p2*x = capitalPerTrade
        #                                                                                                                     x = apitalPerTrade/(f1+f2)

        # quantity = self.capitalPerTrade/(abs(row[1])+abs(row[2]))
        quantity = 1

        lengthOpen = 0
        if openOrderBook is not None: 
            lengthOpen=len(openOrderBook)
        #Long Signal, buy y long, short x
        # if e_t < -np.sqrt(Q_t) and lengthOpen < self.simOpen:
        if predictedSpread is not None and predictedSpread > self.std_array[-1]*self.std_ratio:
            
            #Base Ratios
            yRatio = -1        #belongs to row[2], or y
            xRatio = prediction           #belongs to row[1], or x
            
            #Multiply by the asset factor
            yQuantity= yRatio*quantity
            xQuantity= xRatio*quantity
            
            # Long y, short x
            orders.append([None, 'BUY', self.features[1], yQuantity, row[2], row[0], 'OPEN']) #y
            orders.append([None, 'SELL', self.features[0], xQuantity, row[1], row[0], 'OPEN']) #x
        
        #Short signal, sell y in quantity of beta[0] (dynamichedge ratio), long x tin proportion to 1
        # if e_t > np.sqrt(Q_t)and lengthOpen < self.simOpen:
        if predictedSpread is not None and predictedSpread < -self.std_array[-1]*self.std_ratio:
            
            #Base Ratios
            yRatio = 1        #belongs to row[2], or y
            xRatio = -prediction              #belongs to row[1], or x
            
            #Multiply by the asset factor
            yQuantity= yRatio*quantity
            xQuantity= xRatio*quantity
            
            # Short y, long x
            orders.append([None, 'SELL', self.features[1], yQuantity, row[2], row[0], 'OPEN']) #y
            orders.append([None, 'BUY', self.features[0], xQuantity, row[1], row[0], 'OPEN']) #x
          
        
        # if len(self.y)>100 and self.plot==True:
        #     window = 20
        #     self.axes[1].clear()
        #     self.axes[0].clear()

        #     self.axes[1].plot(range(window),self.y[-window:])
        #     self.axes[1].plot(range(window),self.yhatPlusQ[-window:])
        #     self.axes[1].plot(range(window),self.yhatMinusQ[-window:])

        #     self.axes[0].plot(range(window),self.dfHistory.iloc[-window:])
        #     if len(orders)>0:
        #         for order in orders:
        #             feature = order[2]
        #             if order[6]=='OPEN':
        #                 if order[1]=='BUY':
        #                     self.axes[0].scatter(window-1,self.dfHistory[feature].iloc[-1],marker='^',color='green')
        #                 elif order[1]=='SELL':
        #                     self.axes[0].scatter(window-1,self.dfHistory[feature].iloc[-1],marker='v',color='red')
        #     plt.pause(0.5)

        self.tick(row, orders)
        self.S1 = self.S1[-5:]
        self.S2 = self.S1[-5:]
        self.predictions = self.predictions[-5:]
        self.error = self.error[-5:]