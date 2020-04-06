#EClient is outgoing messages from the code to the TWS station
from ibapi.client import EClient
# Ewrapper takes in messages from the tws
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum

import pandas as pd
from datetime import datetime
from ibapi.utils import iswrapper

from datetime import time, tzinfo, timedelta


def ASX_contract(ticker):
    contract = Contract()
    contract.symbol = ticker
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'AUD'
    contract.primaryExchange = 'ASX'
    
    return contract

def getShortableStocksASX():
    import requests
    import re

    website_url = requests.get('https://www.interactivebrokers.com.au/en/index.php?f=4587&cntry=australia&tag=Australia&ib_entity=hk&ln=').text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(website_url,'lxml')
    My_table = soup.find('table',{'class':'table table-striped table-bordered'})

    shortableStocks = []
    for row in My_table.findAll('tr')[1:]:
        string = re.findall(r'>[\w]+</a>', str(row.findAll('a')))[0]
        string = string[1:string.index('<')]
        shortableStocks.append(string)
    return shortableStocks