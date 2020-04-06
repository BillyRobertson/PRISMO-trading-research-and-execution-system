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


openOrderBook=  plt.figure(figsize=(20,6))
gs = openOrderBook.add_gridspec(1,1)
orderBook = openOrderBook.add_subplot(gs[0,0])
# openOrClose.set_xticks([0,1,2])
# openOrClose.set_yticks([0,1,2])


homePageFigures_ = [openOrderBook]

def animateHomePage(i, pairIndex, figures):

	if pairIndex == 0:
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
		figures[0].text(0,0,orderBooks.to_string(),bbox=props, verticalalignment='bottom')
		figures[0].set_axis_off()



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
		for F in (StartPage,):

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

		#OPEN ORDERBOOK SUMMARY
		index = 0
		boxes.append([	ttk.Label(self, text="Open OrderBook", font = LARGE_FONT, background = 'white'),
						FigureCanvasTkAgg(homePageFigures_[index], self)])

		boxes[-1][0].grid(row = 0, column = 0)
		boxes[-1][1].draw()
		boxes[-1][1].get_tk_widget().grid(row=1,column=0)

		self.canvas.append(boxes[-1][1])


app = SeaofPrismoApp()
ws = app.winfo_screenwidth() # width of the screen
hs = app.winfo_screenheight() # height of the screen
# calculate x and y coordinates for the Tk root window
fractionCovering = 3/5
x = 0
y = (1-fractionCovering)*hs 

# set the dimensions of the screen 
# and where it is placed
app.geometry('%dx%d+%d+%d' % (ws, fractionCovering*hs, x, y))

homePageWindows = [animation.FuncAnimation(f_h, animateHomePage, interval = 5000, blit = False,fargs = [pairIndex] + [f_h.axes]) for f_h, pairIndex in zip(homePageFigures_, range(len(homePageFigures_)))]
app.mainloop()

# fargs = [timeSeries,marketVal]
#  fargs = [timeSeries2,marketVal2]