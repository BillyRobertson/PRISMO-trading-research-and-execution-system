from ib_insync import IB, Forex, Stock, MarketOrder
from functionSrc import *
import sys
from util import order_util
import itertools

"""
A base model containing common IB functions. 

For other models to extend and use.
"""
import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


class BaseModel(object):
	def __init__(self, host='127.0.0.1', port=7497, client_id=1):
		self.host = host
		self.port = port
		self.client_id = client_id

		self.__ib = None
		self.pnl = None  # stores IB PnL object
		self.positions = {}  # stores IB Position object by symbol

		self.symbol_map = {}  # maps contract to symbol
		self.symbols, self.contracts = [], []

	def init_model(self, to_trade, directories):
		"""
		Initialize the model given inputs before running.
		Stores the input symbols and contracts that will be used for reading positions.

		:param to_trade: list of a tuple of symbol and contract, Example:
			[('EURUSD', Forex('EURUSD'), ]
		"""
		self.symbol_map = {str(contract): ident for (ident, contract) in to_trade}
		self.contracts = [contract for (_, contract) in to_trade]

		self.exchange = self.contracts[0].exchange

		self.symbols = list(self.symbol_map.values())
		self.assetA = self.symbols[0]
		self.assetB= self.symbols[1]
		print(self.symbols)
		shortable = getShortableStocksASX()
		if not all([x in shortable for x in self.symbols]):
			print([y for y in itertools.compress(self.symbols, [x in shortable for x in self.symbols])],' are not shortable')
			sys.exit(1)

		self.logDir = directories['logs']
		self.loggerOrderHistory = setup_logger('log', self.logDir+'/log_'+''.join(self.symbols)+'_'+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.log')
		self.loggerOrderHistory.info('LOGS \n')
		self.loggerOrderHistory.info('Script run at ' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
		

	def connect_to_ib(self):
		self.ib.connect(self.host, self.port, clientId=self.client_id)

	def request_pnl_updates(self):
		account = self.ib.managedAccounts()[0]
		self.ib.reqPnL(account)
		
		self.ib.pnlEvent += self.on_pnl

	def on_pnl(self, pnl):
		""" Simply store a copy of the latest PnL whenever where are changes """
		self.pnl = pnl

	def request_position_updates(self):
		self.ib.reqPositions()
		self.ib.positionEvent += self.on_position

	def on_position(self, position):
		""" Simply store a copy of the latest Position object for the provided contract """
		symbol = self.get_symbol(position.contract)
		if symbol not in self.symbols:
			print('[warn]symbol not found for position:', position)
			return

		self.positions[symbol] = position

	def set_delayed_data(self):
		self.ib.reqMarketDataType(3)

	def request_all_contracts_data(self, fn_on_tick):
		for contract in self.contracts:
			try:
				self.ib.reqMktData(contract)
			except Exception as e:
				print(e)

		self.ib.pendingTickersEvent += fn_on_tick

	def requestOpenOrders(self):
		return self.ib.reqOpenOrders() 

	def place_market_order(self, contract, qty, fn_on_filled, orderType):
		order = MarketOrder(order_util.get_order_action(qty), abs(qty), openClose = orderType)
		trade = self.ib.placeOrder(contract, order)
		self.loggerOrderHistory.info('\nOpened\n'+str(trade)+'\n\n')
		trade.filledEvent += fn_on_filled
		return trade

	def get_symbol(self, contract):
		"""
		Finds the symbol given the contract.

		:param contract: The Contract object
		:return: the symbol given for the specific contract
		"""
		symbol = self.symbol_map.get(str(contract), None)
		if symbol:
			return symbol

		symbol = ''
		if type(contract) is Forex:
			symbol = contract.localSymbol.replace('.', '')
		elif type(contract) is Stock:
			symbol = contract.symbol

		return symbol if symbol in self.symbols else ''

	@property
	def ib(self):
		if not self.__ib:
			self.__ib = IB()

		return self.__ib

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger