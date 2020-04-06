
import pandas as pd
import numpy as np
import random
import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')




class backtest(object):

    def __init__(self, 
            timeSeries, 
            toTrade,
            shape, 
            printBool, 
            plotBool,
            decimalPlaces = 8,
            lookback = None,
            transactionFeesPercent = 0.008,
            minimumComission = 6,
            InitialCapital = 0,
            logDir = None):


        #Create Logs to keep track of order history, capital and portfolio balance over time
        self.logDir = logDir
        self.loggerOrderHistory = setup_logger('order_history', self.logDir+'/orderHistory.log')
        self.loggerOrderHistory.info('ORDER HISTORY\n')

        self.portfolioHistory = setup_logger('portfolio_history', self.logDir+'/portfolioHistory.log')
        self.portfolioHistory.info('PORTFOLIO HISTORY\n')

        self.capitalHistory = setup_logger('capital_history', self.logDir+'/capitalHistory.log')
        self.capitalHistory.info('CAPITAL HISTORY\n')

        #Decimal Places should be assigned globally for transparency
        self.decimalPlaces = decimalPlaces
        
      
        #The orderbook keeps track of all past orders made, stored in a data frame with the columns declared below
        self.orderbook = None
        self.orderbookColumns = ['ID','signal','asset','quantity','price','date','openClosed']
        

        self.totalComish = 0
        #We create a portfolio that keeps track of the quantity of each asset we're trading and the corresponding market value at time t (n.b. market value = price(t) * quantity(t))
        #Should look like
        '''

                    quantity        marketValue
        asset1          5               420
        asset2          11              69
        ...

        '''
        portfolioColumns = ['quantity','marketValue']
        portfolioRows = [[toTrade[i]]+[0]*len(portfolioColumns) for i in range(len(toTrade))]
        self.portfolio = pd.DataFrame(portfolioRows)
        self.portfolio.columns = ['asset','quantity','marketValue']
        self.portfolio.set_index(['asset'], inplace=True)
        self.portfolio = self.portfolio.astype(float)
        

        #Declare the initial amount of capital, create arrays to track the capital at each tick, alongside the returns, pnl, and value of investments
        self.capital = InitialCapital
        self.cumulativeCapital = pd.DataFrame(index=['capital'])
        self.returns = pd.DataFrame(index=['returns'])
        self.investmentValues = [0]

        #Transaction Fee Valie
        self.transactionFeesPercent = transactionFeesPercent
        self.minimumComission = minimumComission


        #Tracks the number of trades made by the stragety.
        self.numberOfTrades = 0
        
        #Historical [timestamp, prices... ] array
        self.dfHistory = None   #close prices
        self.dfOpenHistory = None
        self.dfLowHistory = None

    #Format the dates to UNIX TIME.
    #We'll be fetching data from a whole range of sources, so we wanna make sure the dates are consistently formatted as unix objects
    def formatDates(self, timeSeries):
        if type(timeSeries) is not pd.DataFrame:
            raise Exception('Data needs to be a dataframe to format dates')
        
        if type(timeSeries.index[0]) is pd._libs.tslibs.timestamps.Timestamp:
            print("Dates inputted as timestamps. Converting to unix")
            timeSeries.index = pd.to_datetime(timeSeries.index).astype(np.int64)//10**9
            
        return timeSeries



    # The tick function gets fed an array of orders unique to each strategy, as well as the current prices of each asset. 
    # prices = [date, feature1.price, feature2.price, feature3.price, ...]
    # orders = [[order1],[order2],[...],... ]
    #        =  [[int(ORDER_ID), str(BUY/SELL), str(fFEATURE_NAME), floata(QUANTITY), float(PRICE), unix(DATE), str(OPEN/CLOSED)], .....]
    #     e.g.  [[420, 'BUY', GOLD.AX, 1, 100, 1583987952, 'OPEN'],...]
    
    # In essence, the tick function executes all of the orders fed in. The main things achieved are:
    # - Update historical price dataframes
    # - for all closed and newly opened trades:
    #     - calculate the investment value of the order  = previousQuantity*previousPrice = market value of the trade OPENED
    #     - add order to orderbook
    #     - update the portfolio quantity and market value
    #     - update capital using the pnl of the trade         
    # - uniquely to the closed trades we:
    #     - calculate pnl = abs(quantityTrade)*(previousPrice-price)
    #     - calculate the returns = pnl/investmentValue


    def tick(self, prices, orders, closeAll = False):
        



        ## CLOSED ## We close the orders that are being closed here
        
        closedOrders = [x for x in orders if x[6] == 'CLOSED']          #Retrieves the closed orders
                                             

        # Iterate through closed orders
        for order in closedOrders:

            #Create variables for transparency
            ID = order[0]  
            signal = order[1]
            asset = order[2]
            quantityTrade = round(order[3],6)
            price = round(order[4], 8)
            
            previousQuantity = list(self.orderbook.loc[self.orderbook['ID'] == ID, 'quantity'])[0]
            previousPrice = round(list(self.orderbook.loc[self.orderbook['ID'] == ID, 'price'])[0],8)
            
            pnl = 0
            investmentValue = 0   

            date = order[5]

            #If the order is a 'CLOSED' trade, then turn the corresponding 'OPEN' trade into an 'DEPRICATED_OPEN' trade
            if order[6] == 'CLOSED':
                self.orderbook.loc[self.orderbook['ID'] == ID,'openClosed'] = 'DEPRICATED_OPEN'
            
            # Find the investment value so we can calculate the overall returns for the closed trade.
            investmentValue += abs(previousQuantity*previousPrice)


            # Update the orderbook
            if self.orderbook is None:
                    self.orderbook = pd.DataFrame([order])
                    self.orderbook.columns = self.orderbookColumns
            else:
                temp = pd.DataFrame([order])
                temp.columns = self.orderbookColumns
                self.orderbook = pd.concat([self.orderbook, temp])



            #COMISSION OF THE CLOSE ORDER
            if abs(quantityTrade*price)*self.transactionFeesPercent < self.minimumComission:
                comissionClose = self.minimumComission
            else:
                comissionClose = abs(quantityTrade*price)*self.transactionFeesPercent

            #COMISSION OF THE OPEN ORDER
            if abs(previousQuantity*previousPrice)*self.transactionFeesPercent < self.minimumComission:
                comissionOpen = self.minimumComission
            else:
                comissionOpen = abs(previousQuantity*previousPrice)*self.transactionFeesPercent

            self.totalComish += comissionOpen + comissionClose


            # If the signal is a buy, add to orderbook and adjust portfolio
            if signal == 'BUY':
                self.portfolio.loc[asset].quantity += round(float(quantityTrade), self.decimalPlaces)                           
                self.portfolio.loc[asset].marketValue = round(self.portfolio.loc[asset].quantity*price, self.decimalPlaces)
                self.capital += -abs(quantityTrade*price)- comissionClose - comissionOpen
                

                # Calculate returns and pnl. n.b. since we're buying back a short contract, our profit will be previous price - current price multiplied quantity (implies drop in price gives profit)
                prof = round(abs(quantityTrade)*(previousPrice-price),8)

                pnl += prof- comissionClose - comissionOpen

                self.loggerOrderHistory.info(order+[prof])

            elif signal == 'SELL':
                self.portfolio.loc[asset].quantity += round(quantityTrade, self.decimalPlaces)
                self.portfolio.loc[asset].marketValue = round(self.portfolio.loc[asset].quantity*price, self.decimalPlaces)
                self.capital += abs(quantityTrade*price) -comissionClose - comissionOpen
                
                # Find the investment value so we can calculate the overall returns for the closed trade.
                prof = round(abs(quantityTrade)*(price - previousPrice),9)
                # Calculate returns and pnl
                pnl += prof- comissionClose- comissionOpen

                self.loggerOrderHistory.info(order+[prof- comissionClose - comissionOpen])


            #Calculate returns
            if investmentValue == 0:
                if date in self.returns.columns:     
                    self.returns[date] += 0
                else:
                    self.returns[date] = 0
            else:
                if date in self.returns.columns:     
                    self.returns[date] += pnl/investmentValue
                else:
                    self.returns[date] = pnl/investmentValue
            #We update the capital after closing all trades, so we don't observe huge "jumps" in capital
            self.cumulativeCapital[date] = self.capital

        self.portfolio = self.portfolio.astype(float)



       


        
        ## OPENED ##
        # Open new orders here
        
        investmentValue = 0
        
        openOrders = [x for x in orders if x[6] == 'OPEN']        
        for order in openOrders:
            
            self.numberOfTrades +=1


            #Create unique ID
            if self.orderbook is None and order[0] == None:
                ID = random.choice([x for x in random.sample(range(1, 10000),100)])
                order[0] = ID
            elif order[0] == None:
                ID = random.choice([x for x in random.sample(range(1, 10000),100) if x not in list(self.orderbook.ID.values)])
                order[0] = ID
            elif order[0] is not None:
                ID == order[0]
            
            
            
            #Create variables for transparency
            signal = order[1]
            asset = order[2]
            quantityTrade = round(order[3], self.decimalPlaces)
            price = round(order[4], self.decimalPlaces)
            #Adjust Postfolios again
            if signal == 'BUY' :
                if self.orderbook is None:
                    self.orderbook = pd.DataFrame([order])
                    self.orderbook.columns = self.orderbookColumns
                else:
                    temp = pd.DataFrame([order])
                    temp.columns = self.orderbookColumns
                    self.orderbook = pd.concat([self.orderbook, temp])
                    
                self.portfolio.loc[asset].quantity += round(float(quantityTrade), self.decimalPlaces)
                self.portfolio.loc[asset].marketValue = round(self.portfolio.loc[asset].quantity*price, self.decimalPlaces)
                self.capital += -abs(quantityTrade*price)
                investmentValue += abs(quantityTrade*price)
                
            elif signal == 'SELL':
                
                if self.orderbook is None:
                    self.orderbook = pd.DataFrame([order])
                    self.orderbook.columns = self.orderbookColumns
                else:
                    temp = pd.DataFrame([order])
                    temp.columns = self.orderbookColumns
                    self.orderbook = pd.concat([self.orderbook, temp])
                    
                    
                self.portfolio.loc[asset].quantity += round(quantityTrade, self.decimalPlaces)
                self.portfolio.loc[asset].marketValue = round(self.portfolio.loc[asset].quantity*price, self.decimalPlaces)
                self.capital += abs(quantityTrade*price) 
                investmentValue += abs(quantityTrade*price)

            #Log the orders
            self.loggerOrderHistory.info(order)
                
                


        

        #Log The Portfolio and capital over time 
        self.portfolioHistory.info(self.portfolio)
        self.capitalHistory.info(self.capital)

        self.investmentValues.append(investmentValue)
        
        #If we're closing all of the trades, we don't want to create an extra element in returns because this causes a length mismatch when we go to plot vs. dates
        if closeAll == True:
            self.returns[-2] = self.returns[-2]+self.returns[-1]
            self.returns = self.returns[:-1]

            self.cumulativeCapital[-2] = self.cumulativeCapital[-2]+self.cumulativeCapital[-1]
            self.cumulativeCapital = self.cumulativeCapital[:-1]

        # if len(self.pnl) > 15 and self.plot == True:
        #     print('\n\n')
        #     plt.plot(Strategy.yhat_val[-15:], color = 'black')
        #     plt.plot(Strategy.yhatPlusQ[-15:], color = 'green')
        #     plt.plot(Strategy.yhatMinusQ[-15:], color = 'green')
        #     plt.plot(Strategy.y[-15:], color = 'red')
        #     plt.show()
        #     plt.plot()
        


#         plt.plot(Strategy.e[5:], linewidth=0.2)
#         plt.plot(Strategy.Q[5:])
#         plt.show()

#             if len(self.orderbook)>15:
#                 print('\nPast Orders:')
#                 print(self.orderbook.loc[-15:])
            
#             time.sleep(1)
#             clear_output()
        
#         plt.plot(Strategy.hedges)
#         plt.show()
#         print([x[0].tolist()[0] for x in Strategy.beta])
#         plt.plot([x[0].tolist()[0] for x in Strategy.beta])
#         plt.show()
#         plt.plot([x[1].tolist()[0] for x in Strategy.beta])
#         plt.show()



#Setup log files
def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger