import os

from ib_insync import Forex, Stock

from models.cointegration_model_1 import coint1


from multiprocessing import Process


if __name__ == '__main__':
	hotspot = '192.168.43.7'
	local = '127.0.0.1'
	TWS_HOST = os.environ.get('TWS_HOST', local)
	TWS_PORT = os.environ.get('TWS_PORT', 7497)

	print('Connecting on host:', TWS_HOST, 'port:', TWS_PORT)
	data_directory = 'D:/PRISMO/historicalData/data/allOrdsnobiasFINAL.pickle'
	model = coint1(
		data_directory = data_directory,
		delta = 4e-07,
		ve = 2e-05,
		host=TWS_HOST,
		port=TWS_PORT,
		client_id=2
	)

	to_trade = [
 		('NKE',Stock("NKE","NYSE")),
		('IBM', Stock("IBM",'NYSE'))
	]
			
	model.run(to_trade=to_trade, trade_qty=100)

# to_trade = [
 		# ('IOO',Stock("IOO","ASX")),
        # ('IZZ', Stock("IZZ",'ASX'))
# 	]