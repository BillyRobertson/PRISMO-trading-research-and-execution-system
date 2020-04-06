import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta
import pytz 

UTC_TIMEZONE = tz.tzutc()
LOCAL_TIMEZONE = tz.tzlocal()


def convert_utc_datetime(datetime):
	utc = datetime.replace(tzinfo=UTC_TIMEZONE)
	local_time = utc.astimezone(LOCAL_TIMEZONE)
	return pd.to_datetime(local_time)


def convert_local_datetime(datetime):
	local_time = datetime.replace(tzinfo=LOCAL_TIMEZONE)
	return pd.to_datetime(local_time)

def marketOpenCloseTimes(timezone,open_,close,delay):
    current_market = datetime.now(pytz.timezone(timezone))
    marketOpen = current_market.replace(hour=open_, minute = 0, second = 0,microsecond = 0) + timedelta(minutes=delay)
    marketClose = current_market.replace(hour=close, minute = 0, second = 0,microsecond = 0)  + timedelta(minutes=delay) 
    return marketOpen, marketClose
