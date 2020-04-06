import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk

import sys

import matplotlib.animation as animation
from matplotlib import style
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import pickle

import matplotlib.pyplot as plt

from multiprocessing import Process

import time
from datetime import datetime, timedelta

import sys
sys.path.append('C:/Users/Billy/Documents/PRISMO/Executioner/')
from util.functionSrc import *
from util import dt_util

marketOpen, marketClose = dt_util.marketOpenCloseTimes('Australia/Sydney', open_ = 10, close = 16, delay = 20)
marketOpenLocal = marketOpen.astimezone()
marketCloseLocal = marketClose.astimezone()
marketOpenLocalDelayed =  marketOpenLocal+timedelta(minutes=20)
marketCloseLocalDelayed =  marketCloseLocal+timedelta(minutes=20)


LastVisibleFrame = {}



#Some basic styling
style.use("ggplot")
#style.use("dark_background")
LARGE_FONT=("Verdana", 12)

pairs = [['TLS','SKC'],['ABP','SPK']]
strategyType = ['pairs trading (Kalman, bollinger)','pairs trading (Kalman, bollinger)']

#PAIRS1
fig1 = plt.figure()
gs = fig1.add_gridspec(2,3)
timeSeries1 = fig1.add_subplot(gs[0,:-1])
marketVal1 = fig1.add_subplot(gs[1,:-1],sharex=timeSeries1)
openPositions1  = fig1.add_subplot(gs[0,-1])
secondTimeSeries1 = timeSeries1.twinx()
returns1 = fig1.add = fig1.add_subplot(gs[1,-1])

homePageFig1 = plt.figure(figsize=(3,3))

gs = homePageFig1.add_gridspec(1,1)
returns1 = homePageFig1.add_subplot(gs[0,0])


#PAIRS2
fig2 = plt.figure()
gs = fig2.add_gridspec(2,3)
timeSeries2 = fig2.add_subplot(gs[0,:-1])
marketVal2 = fig2.add_subplot(gs[1,:-1],sharex=timeSeries2)
openPositions2  = fig2.add_subplot(gs[0,-1])
secondTimeSeries2 = timeSeries2.twinx()
returns2 = fig1.add = fig2.add_subplot(gs[1,-1])

homePageFig2 = plt.figure(figsize=(3,3))
gs = homePageFig2.add_gridspec(1,1)
returns2 = homePageFig2.add_subplot(gs[0,0])


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

marketsOpenClose =  plt.figure(figsize=(6,3))
gs = marketsOpenClose.add_gridspec(1,1)
openOrClose = marketsOpenClose.add_subplot(gs[0,0])
openOrClose.set_xticks([0,1,2])
openOrClose.set_yticks([0,1,2])

openOrderBook=  plt.figure(figsize=(12,6))
gs = openOrderBook.add_gridspec(1,1)
orderBook = openOrderBook.add_subplot(gs[0,0])
# openOrClose.set_xticks([0,1,2])
# openOrClose.set_yticks([0,1,2])


figures_ = [fig1, fig2]

homePageFigures_ = [ homePageFig1, homePageFig2, marketsOpenClose, openOrderBook]

def animateHomePage(i, pairIndex, figures):

	returnsArray =[   ['a',[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,0,11]],
				 ['b',[5,4,3,2,5,6,2,3],[2,3,4,5,6,7,8,9,0,11]]]
	if pairIndex < 2:
		pair = pairs[0]
		[positions,y_values,yhat_minus_Q, yhat_plus_Q, resampled, lastUpdate, signal, marketClose, marketCloseTZ] = pickle.load(open("C:/Users/Billy/Documents/PRISMO/Executioner/gui/data"+''.join(pair)+".pickle", "rb"))
	else:
		#If it's homepage 2, we only care about the marketClose and close TZ
		[positions,y_values,yhat_minus_Q, yhat_plus_Q, resampled, lastUpdate, signal, marketClose, marketCloseTZ] = pickle.load(open("C:/Users/Billy/Documents/PRISMO/Executioner/gui/data"+''.join(pairs[0])+".pickle", "rb"))



	if pairIndex == 3:
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		orderBooks = None
		for pair in pairs:
			orderB = pickle.load(open('C:/Users/Billy/Documents/PRISMO/Executioner/orderMetaData/orderBooks/orderBook'+''.join(pair)+'.pickle','rb'))
			#Filter to openOrders
			orderB = orderB[orderB['openClosed']=='O']
			
			if orderBooks is None:
				orderBooks = orderB
			else:
				orderBooks = pd.concat([orderBooks, orderB])

		figures = homePageFigures_[pairIndex].axes
		figures[0].clear()
		figures[0].text(0.05,0.25,orderBooks.to_string(),bbox=props, verticalalignment='bottom')
		figures[0].set_axis_off()

	elif pairIndex == 2:

		exchange = 'ASX'
		figures = homePageFigures_[pairIndex].axes
		figures[0].clear()
		if datetime.now().astimezone()>marketOpenLocalDelayed:
			props = dict(boxstyle='round', facecolor='green', alpha=0.5)
			exchange += ': OPEN'
		else:
			props = dict(boxstyle='round', facecolor='red', alpha=0.5)
			exchange += ': CLOSED'

		figures[0].text(0.05,0.25,exchange +'\n\nCurrent Time:'+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S%z')+ '\n\nOpen: '
			+str(marketOpen)+'\nOpen (local, incl. delay): '+str(marketOpenLocalDelayed.astimezone())+'\n\nClose: '+str(marketClose)+'\nClose (local, incl. delay): '
			+str(marketCloseLocalDelayed.astimezone()),bbox=props, verticalalignment='top')

		# figures[0].text(0,0,str(marketOpen)+'\n'+str(marketClose)+'\n'+str(marketClose.astimezone())+'\n'+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S%z'),bbox=props)
		figures[0].set_axis_off()





	else:
		figures = homePageFigures_[pairIndex].axes
		figures[0].clear()
		[pair, projectedReturns, realisedReturns] = returnsArray[pairIndex]
		figures[0].plot(projectedReturns)




def animate(i, pairIndex,figures):
	pair = pairs[pairIndex]
	[positions,y_values,yhat_minus_Q, yhat_plus_Q, resampled, lastUpdate, signal, marketClose, marketCloseTZ] = pickle.load(open("C:/Users/Billy/Documents/PRISMO/Executioner/gui/data"+''.join(pair)+".pickle", "rb"))

	# = [timeSeries, marketVal, openPos, SecondTs]
	figures[0].clear()
	figures[1].clear()
	figures[2].clear()
	figures[3].clear()

	dates = [datetime.strftime(x,'%d-%m-%Y') for x in resampled.index]

	figures[0].axvspan(len(y_values)-2, len(y_values), facecolor='g', alpha=0.3)
	figures[0].plot(list(resampled[resampled.columns[0]]), color = 'red')
	figures[0].set_ylabel(resampled.columns[0], color='red')
	figures[0].set_title('Historical data plus most recent price point')
	figures[0].set_xticklabels(dates,rotation=40,ha='right')
	figures[3].plot(list(resampled[resampled.columns[1]]), color = 'blue')
	figures[3].set_ylabel(resampled.columns[1], color='blue')

	figures[1].axvspan(len(y_values)-2, len(y_values), facecolor='g', alpha=0.3)
	figures[1].plot(y_values,color='black')
	figures[1].plot(yhat_minus_Q, color='green')
	figures[1].plot(yhat_plus_Q,color='green')
	figures[1].set_title('Hedged Portfolio Value with entry and exit bands')
	dates = [datetime.strftime(x,'%d-%m-%Y') for x in resampled.index]
	figures[1].set_xticklabels(dates,rotation=40,ha='right')
	
	position_ = 'Last Update: '+lastUpdate+'\n\n'+'Positions:\n'
	for position in positions: position_+=str(position).replace(',','\n')+'\n'
	position_ +='\n\n' +'Current Signal:\n'+signal
	figures[2].text(0,0,position_,bbox=props)
	figures[2].set_axis_off()




class SeaofPrismoApp (tk.Tk):
	
	def __init__(self,*args,**kwargs):
		
		self._running_anim = None

		tk.Tk.__init__(self, *args,**kwargs)
		tk.Tk.iconbitmap(self)
		tk.Tk.wm_title(self, "Prismo")

		container = tk.Frame(self)
		# container2 = tk.Frame(self)

		#hack - bad - just shoves shit in to a window from chosen side
		container.pack(side="top", fill="both", expand=True)
		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		self.frames = {}
		#self.frames2 = {}

		#add additional class names here for addition pages.
		for F in (StartPage, Pairs1, Pairs2):

			frame = F(container, self)
			frame.configure(background='white')
			print(F.__name__)
			self.frames[F] = frame
			frame.grid(row=0, column=0, sticky="nsew")

		self.show_frame(StartPage)

	def show_frame(self, cont):
		
		frame = self.frames[cont]
		qf(cont)
		LastVisibleFrame=cont
		canvass=[x for x in frame.canvas]

		if all([x != None for x in canvass]):
			[x.draw_idle() for x in canvass]

		frame.tkraise()
		# self.updateAnimate()


def qf(param):
	print(param)



class StartPage(tk.Frame):

	def __init__(self, parent, controller, *args, **kwargs):
		tk.Frame.__init__(self, parent)

		boxes = [] 										# contains [label, button, canvas]
		self.canvas = []


		#START PAGE FOR STRATEGY ONE
		index = 0
		pair = pairs[index]
		boxes.append([	ttk.Label(self, text=' '.join(pair) +' '+ strategyType[index], font = LARGE_FONT, background = 'white'),
							ttk.Button(self, text= "Details"+' '.join(pair), command =lambda:controller.show_frame(Pairs1)),
							FigureCanvasTkAgg(homePageFigures_[index], self)])

		boxes[-1][0].grid(row = 0, column = index)
		boxes[-1][1].grid(row=1, column = index)
		boxes[-1][2].draw()
		boxes[-1][2].get_tk_widget().grid(row=2, column = index,pady = 10, padx = 50)

		self.canvas.append(boxes[-1][2])

		#START PAGE FOR STRATEGY TWO

		index = 1
		pair = pairs[index]
		boxes.append([	ttk.Label(self, text=' '.join(pair) +' '+ strategyType[index], font = LARGE_FONT, background = 'white'),
							ttk.Button(self, text= "Details"+' '.join(pair), command =lambda:controller.show_frame(Pairs2)),
							FigureCanvasTkAgg(homePageFigures_[index], self)])

		boxes[-1][0].grid(row = 0, column = index)
		boxes[-1][1].grid(row=1, column = index)
		boxes[-1][2].draw()
		boxes[-1][2].get_tk_widget().grid(row=2, column = index,pady = 10, padx = 50)

		self.canvas.append(boxes[-1][2])

		#MARKETS OPEN OR CLOSED SUMMARY
		index = 2
		boxes.append([	ttk.Label(self, text="Markets", font = LARGE_FONT, background = 'white'),
						FigureCanvasTkAgg(homePageFigures_[index], self)])

		boxes[-1][0].grid(row = 0, column = index)
		boxes[-1][1].draw()
		boxes[-1][1].get_tk_widget().grid(row=2, column = index,pady = 10, padx = 50)

		self.canvas.append(boxes[-1][1])

		#OPEN ORDERBOOK SUMMARY
		index = 3
		boxes.append([	ttk.Label(self, text="OrderBook", font = LARGE_FONT, background = 'white'),
						FigureCanvasTkAgg(homePageFigures_[index], self)])

		boxes[-1][0].grid(row = 5, column = 0)
		boxes[-1][1].draw()
		boxes[-1][1].get_tk_widget().grid(row=5)

		self.canvas.append(boxes[-1][1])

class Pairs1(tk.Frame):
	def __init__(self, parent, controller, *args, **kwargs):
		tk.Frame.__init__(self,parent)
		self.controller = controller
		label = ttk.Label(self, text="Pairs trading", font = LARGE_FONT)
		#because only 1 label: use pack
		label.pack(pady = 10, padx = 10)

		button2 = ttk.Button(self, text= "Go back to Home", 
			command =lambda:controller.show_frame(StartPage))
		button2.pack()


		canvas1 = FigureCanvasTkAgg(figures_[0], self)
		canvas1.draw()
		canvas1.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
		self.canvas =[canvas1]
		

class Pairs2(tk.Frame):
	def __init__(self, parent, controller, *args, **kwargs):
		tk.Frame.__init__(self,parent)
		self.controller = controller
		label = ttk.Label(self, text="Pairs trading", font = LARGE_FONT)
		#because only 1 label: use pack
		label.pack(pady = 10, padx = 10)

		button2 = ttk.Button(self, text= "Go back to Home", 
			command =lambda:controller.show_frame(StartPage))
		button2.pack()


		canvas1 = FigureCanvasTkAgg(figures_[1], self)
		canvas1.draw()
		canvas1.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
		self.canvas = [canvas1]
		

app = SeaofPrismoApp()
app.state("zoomed")

detailedWindows = [animation.FuncAnimation(f, animate, interval=5000, blit=False, fargs = [pairIndex] +[f.axes]) for f, pairIndex in zip(figures_, range(len(pairs)))]
homePageWindows = [animation.FuncAnimation(f_h, animateHomePage, interval = 5000, blit = False,fargs = [pairIndex] + [f_h.axes]) for f_h, pairIndex in zip(homePageFigures_, range(len(homePageFigures_)))]
app.mainloop()

# fargs = [timeSeries,marketVal]
#  fargs = [timeSeries2,marketVal2]