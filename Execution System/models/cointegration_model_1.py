import sys
sys.path.append('C:/Users/Billy/Documents/PRISMO/Executioner/')
import datetime as dt
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta, timezone

from util.functionSrc import *
import os

from models.base_model import BaseModel
from util import dt_util
from pandas_datareader import data as pdr

from datetime import datetime
import random
import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

class coint1(BaseModel):
	def __init__(self, directories, delta = 0.01, ve = 0.01, quantityPerTrade=1, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.df_hist = None  # stores mid prices in a pandas DataFrame

		#Comission report and (unique) directory creation
		self.executions_comission = pd.DataFrame(columns = ['Execution','ComissionReport'])  #Store the execution and comissions report from filled trades
		self.comissionDirectory = directories['Comissions']
		index = 0
		date = datetime.now().strftime('%Y_%m_%d')
		self.directory_comission = self.comissionDirectory+''.join(self.symbols)+'_comissions_'+date+'_'+str(index)+'.pickle'
		while os.path.exists(self.directory_comission):
			index+=1
			self.directory_comission = self.comissionDirectory+''.join(self.symbols)+'_comissions_'+date+'_'+str(index)+'.pickle'


		self.pending_order_ids = set()
		self.is_orders_pending = False

		# Input params
		self.trade_qty = 0

		# Strategy params
		self.closeTradingWindow = dt.timedelta(seconds = 60*5)  		# in seconds, how close to the market close do we want to open a trade
		self.data_directory =directories['Data']

		self.delta = delta
		self.Ve = ve
		self.lastTime =None

		self.openTrades = 0
		self.openTradeIDs = []

		self.capitalPerTrade = 7500
		self.directories = directories

		self.orderBookDir = directories['Orderbook'] 
		self.guiDirecory = directories['GUI']
		self.dataDirectory =  directories['Data']
		self.priceHistories = directories['Price_Histories']

		self.signalString = ''
		self.lastPrinted = 1 #initialise with odd number


		#Setup Logger
		# Notes
		# -We call a pair trade 'short' when we short asset A and long asset B. Likewise, a pair trade 'long' for the opposite pair config
		# -We only open trades if there are no currently open pair trades. This lowers transaction costs, but can be changed to allow several simeltaneous trades for a single pair.



	def run(self, to_trade=[], trade_qty=0):
		""" Entry point """

		print('[{time}]started'.format(
			time=str(pd.to_datetime('now')),
		))

		# Initialize model based on inputs
		#to_trade = (TICK, contract(tick))
		self.init_model(to_trade, self.directories)
		print(self.symbols)

		self.trade_qty = trade_qty
		self.df_hist = pd.DataFrame(columns=self.symbols)

		# Establish connection to IB
		self.connect_to_ib()
		self.set_delayed_data()
		self.request_pnl_updates()
		self.request_position_updates()
		# self.request_historical_data()
		self.requestOpenOrders() 

		#Get physical version of orderbook
		self.getOrderBook()

		#Get the open and close times of the market
		self.marketOpen, self.marketClose = dt_util.marketOpenCloseTimes('Australia/Sydney', open_ = 10, close = 16, delay = 20)

		#Get the data from a pickle file
		self.get_historical_data_backend()

		#Initialize the kalman filter parameters
		self.initialize_kalman()


		self.request_all_contracts_data(self.on_tick)

		# Recalculate and/or print account updates at intervals
		while self.ib.waitOnUpdate():
			self.ib.sleep(3)
			self.print_strategy_params()
			self.print_prices()
			self.comissions_execution_matching()
			if not self.is_position_flat:
				self.print_account()

			if int(datetime.now().minute) in [0] and int(datetime.now().second) <5:
				self.backupMetadata()


	def on_tick(self, tickers):
		""" When a tick data is received, store it and make calculations out of it """
		for ticker in tickers:
			self.get_incoming_tick_data(ticker)

		self.perform_trade_logic()
		self.updateGUIdata()

	def backupMetadata(self):
		date = datetime.now().strftime('%Y_%m_%d')


		#DATA BACKUP
		directory_temp = self.priceHistories+''.join(self.symbols)+'_priceData_'+date+'.pickle'
		if os.path.exists(directory_temp):
			df_old = pickle.load(open(directory_temp,'rb'))
			self.df_hist = pd.concat([df_old,self.df_hist],axis = 0)
			self.df_hist.drop_duplicates(inplace = True)

		pickle.dump(self.df_hist, open(directory_temp,'wb'))

		#COMISSION BACKUP
		pickle.dump(self.executions_comission, open(self.directory_comission,'wb'))

	def comissions_execution_matching(self):
		#FIX ID #0001
		#Add the comission report to the ececutions recieved from TWS
		for noComission in self.executions_comission[pd.isnull(list(self.executions_comission['ComissionReport']))].iterrows():
		    for fill in self.ib.fills():
		        if fill[1] == self.executions_comission.loc[(noComission[0], 'Execution')]:
		            self.executions_comission.loc[(noComission[0], 'ComissionReport')]=fill[2]
		            break

	def mainLogFunction(self, orderStringFromCode, orderString):
		resampled = self.df_hist.ffill().resample('30s').ffill().dropna().iloc[-1]
		self.loggerOrderHistory.info(('EXECUTING TRADE: {} \nSignal:' +orderStringFromCode+'.\nAsset A: {}, Asset B: {}').format(self.inClosingWindow, self.assetA, self.assetB))
		self.loggerOrderHistory.info(orderString)
		self.loggerOrderHistory.info('[{time}][account]{symbol_a} price={price1}|'
			  '{symbol_b} price={price2}'.format(
			time=str(datetime.now()),
			symbol_a=self.assetA,
			price1=round(resampled[self.assetA],5),
			symbol_b=self.assetB,
			price2=round(resampled[self.assetB],5)
			))

		self.loggerOrderHistory.info('[{time}][strategy params]|rpnl={rpnl:.2f}|last_beta={betas}|forecast_error={error}|forecast_error_deviation={deviation}\n\n'.format(
			time=str(pd.to_datetime('now')),
			rpnl=self.pnl.realizedPnL,
			betas = self.beta[-1].tolist,
			error = self.e_t_last,
			deviation = self.q_t_last
		))


	def print_prices(self):
		#Show Price summary every 5 minutes
		if self.lastTime is not None and self.lastTime.minute%5 == 0 and self.lastTime.minute != self.lastPrinted:
			self.lastPrinted = self.lastTime.minute
			resampled = self.df_hist.ffill().resample('30s').ffill().dropna().iloc[-1]
			print('[{time}][account]{symbol_a} price={price1}|'
			  '{symbol_b} price={price2}'.format(
			time=str(datetime.now()),
			symbol_a=self.assetA,
			price1=round(resampled[self.assetA],5),
			symbol_b=self.assetB,
			price2=round(resampled[self.assetB],5)
			))

			







	def perform_trade_logic(self):
		
		#Calculate Signals
		self.calculate_signals()

		if self.is_orders_pending or self.check_and_enter_orders():
			return  # Do nothing while waiting for orders to be filled

	def print_account(self):
		if self.lastTime is not None and self.lastTime.minute%5 == 0 and self.lastTime.minute != self.lastPrinted:
			self.lastPrinted = self.lastTime.minute
			[symbol_a, symbol_b] = self.symbols
			position_a, position_b = self.positions.get(symbol_a), self.positions.get(symbol_b)

			print('[{time}][account]{symbol_a} pos={pos_a} avgPrice={avg_price_a}|'
				  '{symbol_b} pos={pos_b}|rpnl={rpnl:.2f} upnl={upnl:.2f}'.format(
				time=str(pd.to_datetime('now')),
				symbol_a=symbol_a,
				pos_a=position_a.position if position_a else 0,
				avg_price_a=position_a.avgCost if position_a else 0,
				symbol_b=symbol_b,
				pos_b=position_b.position if position_b else 0,
				avg_price_b=position_b.avgCost if position_b else 0,
				rpnl=self.pnl.realizedPnL,
				upnl=self.pnl.unrealizedPnL,
			))



	def getOrderBook(self):

		if os.path.exists(self.orderBookDir+'orderBook'+''.join(self.symbols)+'.pickle'):
			self.orderBook = pickle.load(open(self.orderBookDir+'orderBook'+''.join(self.symbols)+'.pickle', 'rb'))
		else:
			self.orderBook = pd.DataFrame(columns=['date','id','assetName','exchange','dateOpen','actionLast','dateClose','quantity','priceOpen','priceClose','marketValueOpen','marketValueClose','pnl','returns','openClosed'])
			self.orderBook.set_index(['id'],inplace=True)

	def print_strategy_params(self):
		# print('[{time}][strategy params]|rpnl={rpnl:.2f}|last_5_beta={betas}|forecast_error={error}|forecast_error_deviation={deviation}'.format(
		# 	time=str(pd.to_datetime('now')),
		# 	rpnl=self.pnl.realizedPnL,
		# 	betas = self.beta[-5:],
		# 	error = self.e_t_last,
		# 	deviation = self.q_t_last
		# ))
		pass

	def updateGUIdata(self):
		position_obj_A = self.positions.get(self.symbols[0])
		position_obj_B = self.positions.get(self.symbols[1])
		positions = [position_obj_A, position_obj_B]

		#Take the y, and yhat plus/minusQ until yesterday's close, plot against the current values, and update every 5 seconds

		error_values = self.error_list[-30:]+ [self.e_t_last ]
		plus_Q = self.PlusQ[-30:] + [self.q_t_last]
		minus_Q = self.MinusQ[-30:] + [-self.q_t_last]

		resampled_hist = self.df_hist[self.df_hist.index < pd.Timestamp.today(tz=dt_util.LOCAL_TIMEZONE).normalize()].iloc[-30:]
		resampled_current = pd.DataFrame(self.df_hist[self.df_hist.index > pd.Timestamp.today(tz=dt_util.LOCAL_TIMEZONE).normalize()].iloc[-30:].ffill().resample('5s').ffill().dropna())
		if len(resampled_current)>0:
			resampled_current = pd.DataFrame(self.df_hist[self.df_hist.index > pd.Timestamp.today(tz=dt_util.LOCAL_TIMEZONE).normalize()].iloc[-30:].ffill().resample('1s').ffill().dropna().iloc[-1]).T
			resampled = pd.concat([resampled_hist.iloc[-30:],resampled_current])
		else:
			resampled=resampled_hist

		
		pickle.dump([positions,
			error_values,
			minus_Q, 
			plus_Q, 
			resampled, 
			datetime.now().strftime('%d-%m-%Y %H-%M-%S'), 
			self.signalString,
			self.marketClose,
			self.marketClose.astimezone()],open(self.guiDirecory+'data'+''.join(self.symbols)+'.pickle','wb'))




	def check_and_enter_orders(self):
		#Only open trades if there are no currently "open" trades
		self.inClosingWindow =  self.lastTime >= self.marketClose.astimezone() - self.closeTradingWindow and self.lastTime < self.marketClose.astimezone() 
		
		if self.openTrades == 0:
			if self.is_position_flat and self.open_long_B_short_A:
				self.signalString = 'Trade {} units of {}, trade {} units of {}.\nIn market close window: {}.'.format(-self.hedgeRatio*self.baseTradeQuantity,
																													self.assetA,
																													self.baseTradeQuantity,
																													self.assetB,
																													self.inClosingWindow)
				#Logs 
				self.mainLogFunction( 'open_long_B_short_A', self.signalString)

				#Only execute trade if in closing window
				if self.inClosingWindow:
					self.place_pair_trade(-self.hedgeRatio*self.baseTradeQuantity, self.baseTradeQuantity, 'O')
				return True




			if self.is_position_flat and self.open_short_B_long_A:

				self.signalString = 'Trade {} units of {}, trade {} units of {}.\nIn market close window: {}.'.format(self.hedgeRatio*self.baseTradeQuantity,
																												self.assetA,
																												-self.baseTradeQuantity,
																												self.assetB,
																												self.inClosingWindow)
				#Logs 
				self.mainLogFunction('open_short_B_long_A', self.signalString)

				if self.inClosingWindow:
					self.place_pair_trade(self.hedgeRatio*self.baseTradeQuantity, -self.baseTradeQuantity, 'O')
				return True
				#Signals


		# self.open_long_B_short_A = e_t > np.sqrt(Q_t)
		# self.close_long_B_short_A= e_t < np.sqrt(Q_t)

		# self.open_short_B_long_A = e_t < -np.sqrt(Q_t)
		# self.close_short_B_long_A = e_t > -np.sqrt(Q_t)


		if self.is_position_short and self.close_long_B_short_A:						#Uncommented because we want to close all orders even if self.is_position_short=False (can happen if one asset has flat positions and the other doesnt)
			open_trades_long_B_short_A = self.orderBook.loc[self.orderBook.assetName.isin([self.symbols[0],self.symbols[1]])].loc[self.orderBook.openClosed=='O']
			quantity_A = sum(list(open_trades_long_B_short_A.loc[open_trades_long_B_short_A.assetName == self.symbols[0]].quantity))
			quantity_B = -sum(list(open_trades_long_B_short_A.loc[open_trades_long_B_short_A.assetName == self.symbols[1]].quantity))

			self.signalString = 'Closing Trades. Trade {} units of {}, trade {} units of {}.\nIn market close window: {}.'.format(quantity_A,
																													self.assetA,
																													quantity_B,
																													self.assetB,
																													self.inClosingWindow)

			#Logs 
			self.mainLogFunction('close_long_B_short_A', self.signalString)


			if self.inClosingWindow:
				self.place_pair_trade(quantity_A , quantity_B, 'C')
			return True

		if self.is_position_long and self.close_short_B_long_A:
			open_trades_long_A_short_B = self.orderBook.loc[self.orderBook.assetName.isin([self.symbols[0],self.symbols[1]])].loc[self.orderBook.openClosed=='O']
			quantity_A = -sum(list(open_trades_long_A_short_B.loc[open_trades_long_A_short_B.assetName == self.symbols[0]].quantity))
			quantity_B = sum(list(open_trades_long_A_short_B.loc[open_trades_long_A_short_B.assetName == self.symbols[1]].quantity))
			self.signalString = 'Closing Trades. Trade {} units of {}, trade {} units of {}.\nIn market close window: {}.'.format(quantity_A,
																													self.assetA,
																													quantity_B,
																													self.assetB,
																													self.inClosingWindow)


			#Logs 
			self.mainLogFunction('close_short_B_long_A', self.signalString)
				


			if self.inClosingWindow:
				self.place_pair_trade(quantity_A , quantity_B, 'C')

			return True

		return False


	#PLACE PAIR TRADE ON MARKET CLOSE (MOC)
	def place_pair_trade(self, qtyA, qtyB, orderType):

		#Want to close all currently open MOC trades
		[contract_a, contract_b] = self.contracts
		print(qtyA, qtyB)
		trade_a = self.place_market_order(contract_a, qtyA, self.on_filled, orderType)
		print('Order placed:', trade_a)

		self.pending_order_ids.add(trade_a.order.orderId)

		trade_b = self.place_market_order(contract_b, qtyB, self.on_filled, orderType)
		print('Order placed:', trade_b)
	
		self.pending_order_ids.add(trade_b.order.orderId)
		print('Order IDs pending execution:', self.pending_order_ids)

		self.is_orders_pending = True

		self.pending_order_ids.add(trade_a.order.orderId)
		self.pending_order_ids.add(trade_b.order.orderId)


	#What happens when a trade is filled
	def on_filled(self, trade):

		self.loggerOrderHistory.info('\nFilled\n'+str(trade)+'\n\n')

		for fill in trade.fills:
			execution = fill[1]
			self.executions_comission.loc[(execution.execId, 'Execution')] = execution

		print('Order filled:', trade.order)
		self.pending_order_ids.remove(trade.order.orderId)
		print('Order IDs pending execution:', self.pending_order_ids)

		self.openTradeIDs.append(trade.order.orderId)

		if len(self.openTradeIDs)==2:
			self.openTrades = 1

		self.updateOrderBook(trade)


		#We want to log the order
		# Update flag when all pending orders are filled
		if not self.pending_order_ids:
			self.is_orders_pending = False

	def updateOrderBook(self, trade):
		print(trade.order.openClose)
		if trade.order.openClose== 'O':

			ID = random.randint(1e10,1e11)
			while ID in self.orderBook.index:
				ID = random.randint(1e10,1e11)

			price = trade.order.lmtPrice
			quantity = trade.order.totalQuantity
			averageFillPrice = trade.orderStatus.avgFillPrice
			marketValue = averageFillPrice*quantity
			action = trade.order.action
			asset = trade.contract.symbol
			exchange = trade.contract.exchange
			openClosed = trade.order.openClose

			self.orderBook.loc[ID] =  [self.lastTime, 
										asset,
										exchange,
										self.lastTime,
										action,
										None,
										quantity,
										averageFillPrice,
										None,
										marketValue,
										None,
										None,
										None,
										openClosed] #''date','assetName','exchange','dateOpen','actionLast','dateClose','quantity','priceOpen','priceClose','marketValueOpen','marketValueClose','marketValue','pnl','returns','openClosed'

			pickle.dump(self.orderBook, open(self.orderBookDir +'orderBook'+''.join(self.symbols)+'.pickle','wb'))


		if trade.order.openClose== 'C':
			quantity = trade.order.totalQuantity
			asset = trade.contract.symbol

			trades = self.orderBook.loc[self.orderBook.assetName == asset].loc[self.orderBook.quantity == quantity].loc[self.orderBook.openClosed == 'O'].iloc[[0]]  #pass a single valued list to avoid returning a series (we want dataframe!!)
			ID = trades.index

			price = trade.order.lmtPrice
			quantity = trade.order.totalQuantity
			averageFillPrice = trade.orderStatus.avgFillPrice
			marketValue = averageFillPrice*quantity
			action = trade.order.action

			self.orderBook.loc[ID,'priceClose'] = price
			self.orderBook.loc[ID,'marketValueClose'] = marketValue
			self.orderBook.loc[ID,'dateClose'] = self.lastTime
			self.orderBook.loc[ID,'actionLast']=action
			self.orderBook.loc[ID,'openClosed']='C'

			pnl = self.orderBook.loc[ID,'marketValueClose'] -  self.orderBook.loc[ID,'marketValueOpen'] 
			returns = self.orderBook.loc[ID,'marketValueOpen']*(self.orderBook.loc[ID,'priceClose']-self.orderBook.loc[ID,'priceOpen'])
			#Means we're closing a short contract
			if action == 'BUY':
				pnl = -pnl
				returns = -returns

			self.orderBook.loc[ID,'returns']=returns
			self.orderBook.loc[ID,'pnl']=pnl

			pickle.dump(self.orderBook, open(self.orderBookDir +'orderBook'+''.join(self.symbols)+'.pickle','wb'))


	def recalculate_strategy_params(self, init_kalman = False):

		#The set of values we want to pass through the kalman filter. If we're initialising the filter, its all historical data. Otherwise, it's the most recent price from IB
		df_recalculate = None
		if init_kalman == True:
			df_recalculate = self.df_hist
		elif self.df_hist is not None and len(self.df_hist)>0:
			resampled = self.df_hist.ffill().resample('30s').ffill().dropna()
			mean = resampled.mean()
			df_recalculate = pd.DataFrame(resampled.iloc[-1]).T


		if df_recalculate is not None and len(df_recalculate)>0:
			df_recalculate = df_recalculate[[self.assetA, self.assetB]]
			for row in df_recalculate.iterrows():
				row = list(row[1])
				featureX = self.symbols[0]
				featureY = self.symbols[1]

				#Extract x and y from the row, put them into numpy form. We include a constant for x so we can fit with a constant.
				x = np.matrix([[row[0]],[1]])
				y = np.matrix(row[1])

				## 1 STEP AHEAD PREDICTION ##
				beta = self.beta[-1]                                            # beta(t|t-1) = beta(t-1|t-1)
				self.R = self.P + self.Vw                                       # R(t|t-1) = R(t-1|t-1) + Vw
				yhat = np.dot(x.T, beta)                                        # yhat = x.beta
				e_t = y - yhat                                                  # e(t) = y(t) - yhat(t)
				Q_t = np.dot( np.dot(x.T, self.R) , x ) + self.Ve               # Q(t) = var(e(t)) = var(y(t) - yhat(t)) 
				#                                                                      = var(y(t)) + var(yhat(t)) + cov[y(t), yhat(t)]
				#                                                                      = x . R(t|t-1) + Ve


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
				self.y.append(y.tolist()[0][0])
				# self.hedgedPortfolio.append(y.tolist()[0][0]-hedgeRatio*row[0])

				self.error_list.append(e_t.tolist()[0][0])
				self.PlusQ.append((np.sqrt(Q_t)).tolist()[0][0])
				self.MinusQ.append((-np.sqrt(Q_t)).tolist()[0][0])


			self.e_t_last = e_t
			self.q_t_last = np.sqrt(Q_t)
			self.yhat_last = yhat
        


	def calculate_signals(self):
		resampled = self.df_hist.ffill().resample('30s').ffill().dropna()
		resampled = resampled[[self.assetA, self.assetB]]
		row = list(resampled.iloc[-1])

		#We want the total value traded to equal approximately the capitalPerTrade value

		self.baseTradeQuantity = self.capitalPerTrade/(max(self.hedges[-1]*abs(row[0]),abs(row[1])))
		self.hedgeRatio = self.hedges[-1]
		x = np.matrix([[row[0]],[1]])  #X IS ASSET A
		y = np.matrix(row[1])			#Y IS ASSET B

		## 1 STEP AHEAD PREDICTION ##
		beta = self.beta[-1]                                            # beta(t|t-1) = beta(t-1|t-1)
		self.R = self.P + self.Vw                                       # R(t|t-1) = R(t-1|t-1) + Vw
		yhat = np.dot(x.T, beta)                                        # yhat = x.beta
		e_t = y - yhat                                                  # e(t) = y(t) - yhat(t)
		Q_t = np.dot( np.dot(x.T, self.R) , x ) + self.Ve               # Q(t) = var(e(t)) = var(y(t) - yhat(t)) 
		#                                                                      = var(y(t)) + var(yhat(t)) + cov[y(t), yhat(t)]
		#                                                                      = x . R(t|t-1) + Ve



		self.e_t_last = e_t
		self.q_t_last = np.sqrt(Q_t)


		# print('TRUE: ',self.symbols,'\t Prices',x,y,'\tbeta: ', beta, e_t, yhat, np.sqrt(Q_t))


		#Validated 08/01/2020
		self.open_long_B_short_A = e_t < -np.sqrt(Q_t)
		self.close_long_B_short_A= e_t > -np.sqrt(Q_t)

		self.open_short_B_long_A = e_t > np.sqrt(Q_t)
		self.close_short_B_long_A = e_t < np.sqrt(Q_t)

		# print('open_long_B_short_A',self.open_long_B_short_A, 
		# 	'\nclose_long_B_short_A',self.close_long_B_short_A,
		# 	'\nopen_short_B_long_A',self.open_short_B_long_A,
		# 	'\nclose_short_B_long_A',self.close_short_B_long_A)

		# x = np.matrix([[row[0]-0.03],[1]])
		# yhat_t = np.dot(x.T, beta)
		# e_t_t = y - yhat  
		# print('\tTEST: ',self.symbols,'\t Prices',x,y,'\tbeta: ', beta, e_t_t, yhat_t, np.sqrt(Q_t))

		# self.open_long_B_short_A = e_t > np.sqrt(Q_t)
		# self.close_long_B_short_A= e_t < np.sqrt(Q_t)

		# self.open_short_B_long_A = e_t < -np.sqrt(Q_t)
		# self.close_short_B_long_A = e_t > -np.sqrt(Q_t)

		# print('open_long_B_short_A',self.open_long_B_short_A, 
		# 	'\nclose_long_B_short_A',self.close_long_B_short_A,
		# 	'\nopen_short_B_long_A',self.open_short_B_long_A,
		# 	'\nclose_short_B_long_A',self.close_short_B_long_A)




	def get_incoming_tick_data(self, ticker):
		"""
		Stores the midpoint of incoming price data to a pandas DataFrame `df_hist`.

		:param ticker: The incoming tick data as a Ticker object.
		"""
		symbol = self.get_symbol(ticker.contract)
		dt_obj = dt_util.convert_utc_datetime(ticker.time)

		self.lastTime = dt_obj

		bid = ticker.bid
		ask = ticker.ask
		mid = (bid + ask) / 2
		try:
			self.df_hist.loc[(dt_obj, symbol)] = mid
		except Exception as e:
			print(e)

	
	def get_historical_data_backend(self):
		data = pickle.load(open(self.dataDirectory, "rb" ) )
		close = data.xs('CLOSE', axis = 1, level = 1)
		toTrade = [x for x in self.symbols]

		if self.exchange == 'ASX':
			toTrade_ = [x+ '.AX' for x in self.symbols]
		elif self.exchange == 'SEHK':
			toTrade_ = [x+ '.HK' for x in self.symbols]
			for index, tick in enumerate(toTrade_):
				if len(tick)==6:
					toTrade_[index] = '0'+toTrade_[index]
		else:
			toTrade_ = toTrade

		#Fill missing (recent) data with yahoo data
		price = pdr.get_data_yahoo(toTrade_)
		df = price['Adj Close'].dropna()
		df.columns = [self.assetA, self.assetB]
		df = df[[not x for x in  df.index.duplicated()]]
		missingDates = df[[dx not in close.index and dx > close.index[-1] for dx in df.index ]]
		#Don't include the price for today
		missingDates = missingDates[missingDates.index<pd.Timestamp.today().normalize()]

		# Simply to ensure that the column order is conistent
		if  all([x in df.columns for x in close.columns]):
				close = close[toTrade_]
				close.columns = toTrade
				close = pd.concat([close,missingDates], sort=True).dropna()
				close = close[[self.assetA, self.assetB]]
		else:
			close=missingDates
			close = close[[self.assetA, self.assetB]]
		if close is not None:
			self.df_hist = close
			del data
			del close
			self.df_hist.index = [dt_util.convert_utc_datetime(x) for x in self.df_hist.index]
		else:
			print('No Histocial Data.')


	def initialize_kalman(self):
		#Kalman Filter Parametes
		self.yhat = [np.matrix(0)]          #stores the measurement prediction of the linear model y (t) = x(t)*β(t) + e(t). Store for each time step
		self.Q = [None]						#The variance in the measurement prediction. Store for each time step.
		self.e = [None]						#The prediction error at each time step.
		self.beta = [np.matrix([[0],[0]])]  #Store the beta values
		self.R = np.zeros([len(self.symbols),len(self.symbols)])		#Covariance Arrays,  measuring the covariance of the error of the hidden variable, β,  estimates
		self.P = np.zeros([len(self.symbols),len(self.symbols)])		#For clarity, we denote R(t|t) by P, and R(t|t-1) as R. R(t | t − 1) is cov(β(t) − βhat(t|t-1))
		self.Vw = self.delta/(1-self.delta)*np.diag(np.ones(len(self.symbols)))
		self.hedges = []					#Store the hedge ratios

		# # For validation
		self.y = []
		self.PlusQ = []
		self.MinusQ = []
		self.error_list = []

		self.recalculate_strategy_params(init_kalman = True)

	def request_historical_data(self):
		"""
		Bootstrap our model by downloading historical data for each contract.

		The midpoint of prices are stored in the pandas DataFrame `df_hist`.
		"""
		for contract in self.contracts:
			self.set_historical_data(contract)

	def set_historical_data(self, contract):
		symbol = self.get_symbol(contract)

		bars = self.ib.reqHistoricalData(
			contract,
			endDateTime=time.strftime('%Y%m%d %H:%M:%S'),
			durationStr='3600 S',
			barSizeSetting='5 secs',
			whatToShow='MIDPOINT',
			useRTH=True,
			formatDate=1
		)
		for bar in bars:
			dt_obj = dt_util.convert_local_datetime(bar.date)
			self.df_hist.loc[dt_obj, symbol] = bar.close

	@property
	def is_position_flat(self):
		position_obj = self.positions.get(self.symbols[0])
		if not position_obj:
			return True

		return position_obj.position == 0

	@property
	def is_position_short(self):
		position_obj_A = self.positions.get(self.symbols[0])
		position_obj_B = self.positions.get(self.symbols[1])
		return position_obj_A and position_obj_B  and position_obj_A.position < 0 and position_obj_B.position > 0

	@property
	def is_position_long(self):
		position_obj_A = self.positions.get(self.symbols[0])
		position_obj_B = self.positions.get(self.symbols[1])
		return position_obj_A and position_obj_B  and position_obj_A.position > 0 and position_obj_B.position < 0
