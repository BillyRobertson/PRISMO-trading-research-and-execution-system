import os

from ib_insync import Forex, Stock
import sys
sys.path.append('C:/Users/Billy/Documents/PRISMO/Executioner/')
from models.cointegration_model_1 import coint1


from multiprocessing import Process


if __name__ == '__main__':
	hotspot = '192.168.43.7'
	local = '127.0.0.1'
	TWS_HOST = os.environ.get('TWS_HOST', local)
	TWS_PORT = os.environ.get('TWS_PORT', 7497)

	print('Connecting on host:', TWS_HOST, 'port:', TWS_PORT)
	directories ={}
	directories['Path'] = 'C:/Users/Billy/Documents/PRISMO/Executioner/'
	directories['GUI'] = 'C:/Users/Billy/Documents/PRISMO/Executioner/gui/'
	directories['Data'] = 'D:/PRISMO/historicalData/data/allOrdsnobiasFINAL.pickle'
	directories['Orderbook'] = 'C:/Users/Billy/Documents/PRISMO/Executioner/orderMetaData/orderBooks/'
	directories['Comissions'] = 'C:/Users/Billy/Documents/PRISMO/Executioner/orderMetaData/comission/'
	directories['Price_Histories'] = 'C:/Users/Billy/Documents/PRISMO/Executioner/orderMetaData/priceHistories/'
	directories['logs'] = 'C:/Users/Billy/Documents/PRISMO/Executioner/logs/'
	model = coint1(
		directories = directories,
		delta = 9e-06,
		ve = 0.002,
		host=TWS_HOST,
		port=TWS_PORT,
		client_id=2
	)

	to_trade = [
 		('ABP',Stock("ABP","ASX")),
		('SPK', Stock("SPK",'ASX'))
	]
			
	model.run(to_trade=to_trade, trade_qty=100)

# to_trade = [
 		# ('IOO',Stock("IOO","ASX")),
        # ('IZZ', Stock("IZZ",'ASX'))
