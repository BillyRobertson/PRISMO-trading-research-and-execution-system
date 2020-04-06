
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



#####################################################
# Aim here is to figure out how much further prices need to move to "cancel" the transaction fees


class cointegrated_pairs_1(backtest):
    def __init__(self, 
                timeSeries, 
                toTrade, 
                shape, 
                printBool, 
                plotBool,
                decimalPlaces = 8,
                lookback = None, 
                delta = 0.0001,
                ve = 0.001,
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
        
        self.comission = transactionFeesPercent
        self.directory = logDir             # Directory for the logs
        self.features = toTrade             # Features being traded
        self.plot = plotBool                # Plots returns and bollinger entries every tick
        
        self.yhat = [np.matrix(0)]           #stores the measurement prediction of the linear model y (t) = x(t)*β(t) + e(t). Store for each time step.

        #The variance in the measurement prediction. Store for each time step.
        self.Q = [None]

        #The prediction error at each time step.
        self.e = [None]
        
        #Store the beta values
        self.beta = [np.matrix([[0],[0]])]

        #Covariance Arrays,  measuring the covariance of the error of the hidden variable, β,  estimates
        #For clarity, we denote R(t|t) by P, and R(t|t-1) as R. R(t | t − 1) is cov(β(t) − βhat(t|t-1))
        self.R = np.zeros([len(self.features),len(self.features)])
        self.P = np.zeros([len(self.features),len(self.features)])

        #delta is a pre-defined value, but can be trained from the dataset, delta of 0 means we're just doing linear regression
        self.delta = delta
        self.Vw = self.delta/(1-self.delta)*np.diag(np.ones(len(self.features)))
        self.Ve = ve

        self.simOpen = simOpen

        self.capitalPerTrade = capitalPerTrade
        #Decimal Places should be assigned globally for transparency
        self.decimalPlaces = decimalPlaces
        
        #Store the hedge ratios
        self.hedges = []

        # # For validation
        # # DELETE FOR SPEED
        # self.yhat_val = []
        self.y = []
        self.yhatPlusQ = []
        self.yhatMinusQ = []
        # self.Q = []
        # self.e = []


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
        

        ### KALMAN FILTER ###
        
        #Importantly declare x as the first element of the row, and y as the second:
        # x= row[1]
        # y= row[2] 
        # Therefore:
        featureX = self.features[0]
        featureY = self.features[1]

        #Extract x and y from the row, put them into numpy form. We include a constant for x so we can fit with a constant.
        x = np.matrix([[row[1]],[1]])
        y = np.matrix(row[2])
        


        ## 1 STEP AHEAD PREDICTION ##
        beta = self.beta[-1]                                            # beta(t|t-1) = beta(t-1|t-1)
        self.R = self.P + self.Vw                                       #    R(t|t-1) = R(t-1|t-1) + Vw
        yhat = np.dot(x.T, beta)                                        #        yhat = x.beta
        e_t = y - yhat                                                  #        e(t) = y(t) - yhat(t)
        Q_t = np.dot( np.dot(x.T, self.R) , x ) + self.Ve               #        Q(t) = var(e(t)) = var(y(t) - yhat(t)) 
        #                                                                             = var(y(t)) + var(yhat(t)) + cov[y(t), yhat(t)]
        #                                                                             = x . R(t|t-1) + Ve
        
        
        ## UPDATE PARAMETERS ##
        K = np.dot(self.R, x) / Q_t                                                 # K is the kalman gain
        beta = beta + K*e_t                                                         # beta(t|t) = beta(t|t-1)+K(t)e(t)
        self.P = self.R - np.dot( np.dot(K, np.transpose(x)), self.R)               # We denote R(t|t) by P, and R(t|t-1) as R. R(t | t − 1) = cov(β(t) − βhat(t|t-1))

        #Add beta and predicted y values to arrays for storage
        self.beta.append(beta)
        self.yhat.append(yhat)
        
        #Form the hedge ratio as a float
        hedgeRatio = beta[0].tolist()[0][0]
        self.hedges.append(hedgeRatio)
      
        # # TODO: THIS WILL BE DEPRICATED
        # # Store parameters for validation
        # self.yhat_val.append(yhat.tolist()[0][0]) 
        self.y.append(y.tolist()[0][0])
        self.yhatPlusQ.append((yhat+np.sqrt(Q_t)).tolist()[0][0])
        self.yhatMinusQ.append((yhat-np.sqrt(Q_t)).tolist()[0][0])
        # self.Q.append(np.sqrt(Q_t).tolist()[0][0])
        # self.e.append(e_t.tolist()[0][0])
        
    




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
                if signal == 'SELL' and e_t < np.sqrt(Q_t):
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

                        ####### AMENDUM
                        #If the drop in market value is n
                        quantity = 1
                        oldXPrice = hedgedOrder[4]
                        oldXRatio= hedgedOrder[3]/quantity
                        oldYRatio = quantityTrade/quantity
                        oldYhat = price - oldXRatio*oldXPrice
                        x_ = np.matrix([[oldXPrice],[1]])
                        y_ = np.matrix(price)
                        yhat_ = np.dot(x_.T, beta)
                        e_t_old =y_ - yhat
                        print(e_t_old, oldYhat)
                        newYhat = e_t
                        comission = self.comission*(abs(newQuantityTrade*newPrice) +  abs(hedgedOrder[3]*newHedgedPrice) + abs(oldYRatio*price )+ abs(oldXRatio*oldXPrice))
                        # print(comission,0.5*np.sqrt(Q_t), oldYhat,newYhat)

                        # if comission < 0.5*np.sqrt(Q_t) and oldYhat- newYhat > comission:
                        #     pass
                        # else:
                        #     orders = orders[:-2]





                #Same as above, but the y asset is being sold to close the order.
                if signal == 'BUY' and e_t > -np.sqrt(Q_t):
                    
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


                        ####### AMENDUM
                        #If the drop in market value is n
                        quantity = 1
                        oldXPrice = hedgedOrder[4]
                        oldXRatio= hedgedOrder[3]/quantity
                        oldYRatio = quantityTrade/quantity
                        oldYhat = oldYRatio*price - oldXRatio*oldXPrice
                        newYhat = e_t
                        comission = self.comission*(abs(newQuantityTrade*newPrice) +  abs(hedgedOrder[3]*newHedgedPrice) + abs(oldYRatio*price )+ abs(oldXRatio*oldXPrice))
                        oldYhat = price - oldXRatio*oldXPrice
                        print(oldXPrice,oldXRatio,oldYRatio, oldYhat, newYhat)
                        # print(comission,0.5*np.sqrt(Q_t), oldYhat, newYhat)
                        # if comission < 0.5*np.sqrt(Q_t) and oldYhat- newYhat > comission:
                        #     pass
                        # else:
                        #     orders = orders[:-2]
                        


        
        
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
        if e_t < -np.sqrt(Q_t) and lengthOpen < self.simOpen:
        # if e_t > np.sqrt(Q_t)and lengthOpen < self.simOpen:
            
            #Base Ratios
            yRatio = 1        #belongs to row[2], or y
            xRatio = -hedgeRatio              #belongs to row[1], or x
            
            #Multiply by the asset factor
            yQuantity= -yRatio*quantity
            xQuantity= -xRatio*quantity
            
            # Long y, short x
            orders.append([None, 'BUY', self.features[1], yQuantity, row[2], row[0], 'OPEN']) #y
            orders.append([None, 'SELL', self.features[0], xQuantity, row[1], row[0], 'OPEN']) #x
        
        #Short signal, sell y in quantity of beta[0] (dynamichedge ratio), long x tin proportion to 1
        if e_t > np.sqrt(Q_t)and lengthOpen < self.simOpen:
        # if e_t < -np.sqrt(Q_t) and lengthOpen < self.simOpen:
            
            #Base Ratios
            yRatio = -1        #belongs to row[2], or y
            xRatio = hedgeRatio               #belongs to row[1], or x
            
            #Multiply by the asset factor
            yQuantity= yRatio*quantity
            xQuantity= xRatio*quantity
            
            # Short y, long x
            orders.append([None, 'SELL', self.features[1], yQuantity, row[2], row[0], 'OPEN']) #y
            orders.append([None, 'BUY', self.features[0], xQuantity, row[1], row[0], 'OPEN']) #x
          
        
        if len(self.y)>100 and self.plot==True:
            window = 20
            self.axes[1].clear()
            self.axes[0].clear()

            self.axes[1].plot(range(window),self.y[-window:])
            self.axes[1].plot(range(window),self.yhatPlusQ[-window:])
            self.axes[1].plot(range(window),self.yhatMinusQ[-window:])

            self.axes[0].plot(range(window),self.dfHistory.iloc[-window:])
            if len(orders)>0:
                for order in orders:
                    feature = order[2]
                    if order[6]=='OPEN':
                        if order[1]=='BUY':
                            self.axes[0].scatter(window-1,self.dfHistory[feature].iloc[-1],marker='^',color='green')
                        elif order[1]=='SELL':
                            self.axes[0].scatter(window-1,self.dfHistory[feature].iloc[-1],marker='v',color='red')
            plt.pause(0.5)

        self.tick(row, orders)