import slicematrixIO
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import Isomap
import numpy as np
import pickle

df = pickle.load(open('D:/PRISMO/historicalData/data/asx200nobiasFINAL.pickle','rb'))
df = df.xs('CLOSE', axis = 1, level = 1)
# df = pickle.load(open("D:/PRISMO/historicalData/ETFSnobiasFINAL.pickle",'rb'))
#Find all shortable stocks currently in the SNP200
shortable_stocks = list(pd.read_csv('C:/Users/Billy/Documents/PRISMO/data/ASX/shortable.csv',header=None)[0])
asxTickers = pd.read_csv('C:/Users/Billy/Documents/PRISMO/docs/20190901-asx200.csv')
snp200_stocks = list(asxTickers['S&P/ASX 200 Index (1 September 2019)'][1:])

snp200_stocks = [x + '.AX' for x in snp200_stocks]
shortable_stocks = [x + '.AX' for x in shortable_stocks]

allPastStocksSnp200 = df.columns.tolist()
currentSnp200 = [x for x in allPastStocksSnp200 if x in snp200_stocks]
current_shortable_snp200 =  [x for x in currentSnp200 if x in shortable_stocks]

#Filter Down Dataframe, fill na with 0, transpose for isomap
df = df[current_shortable_snp200]
# df = df.pct_change()
df = df.fillna(0)
df = df.iloc[int(4.2*len(df)/5):]
df = df.T
model = Isomap(n_components = 2)
proj = model.fit_transform(df)


#format data to lists
data = [x.tolist() for x in proj]
projectionData = []
for index, dist in enumerate(data):
    projectionData.append(dist+[current_shortable_snp200[index]])
    
# Find the closest pairs in the projection
pairs = []
for asset in projectionData:
    asset_name = asset[2]
    x = asset[0]
    y = asset[1]

    distances = []

    for index, otherAsset in enumerate(projectionData):
        x_2 = otherAsset[0]
        y_2 = otherAsset[1]
        distance = np.sqrt((x-x_2)**2 + (y-y_2)**2)
        distances.append([distance, otherAsset[2]])

    closest = sorted(distances, key = itemgetter(0))[1:6]
    closest = [c[1] for c in closest]
    for asset2 in closest:
        pairs.append([asset_name,asset2])

import pickle
pickle.dump(pairs, open('C:/Users/Billy/Documents/PRISMO/data/pairsFromManifoldsStonk.pickle','wb'))