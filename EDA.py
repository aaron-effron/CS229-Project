from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### Get and Examine Data ###########################################
def importData(path):
    data = pd.DataFrame.from_csv(path)
    return data

df = importData("../wine/winemag-data_first150k.csv")

#data overview
df.head()
df.index #150,930 rows
df.columns #10 columns
df.count() #to see missing data

#numeric variables: points, price
df.describe()
df.corr()


#### Get Top Varieties  & Plot 8 : Red vs White ######################
#select important cols
data=df.loc[:,['variety','description']]

variety_freq = data.groupby('variety').count() #632 varieties
variety_freq.columns = ['count']
variety_freq.sum() #all descriptions available
variety_freq = variety_freq.sort_values(by = 'count',ascending = False)

topVarieties = variety_freq[variety_freq['count']>=1000]
topVarieties.count() #31 wine varieties with more than 1000 examples
topVarieties['count'].sum()/data['description'].count() #top 31 represents 82% of data

#topVarieties.index
ColorMap = {u'Chardonnay': 'White', 
            u'Pinot Noir': 'Red',  #champagne is listed separately so Pinot Noir should all be reds
            u'Cabernet Sauvignon': 'Red', 
            u'Red Blend': 'Red', 
            u'Bordeaux-style Red Blend': 'Red', 
            u'Sauvignon Blanc': 'White', 
            u'Syrah': 'Red', 
            u'Riesling': 'White', 
            u'Merlot': 'Red', 
            u'Zinfandel': 'Red', 
            u'Sangiovese': 'Red', 
            u'Malbec': 'Red', 
            u'White Blend': 'White', 
            u'RosÃ©': 'Red', 
            u'Tempranillo': 'Red',  
            u'Nebbiolo': 'Red', 
            u'Portuguese Red': 'Red', 
            u'Sparkling Blend': 'N/A', 
            u'Shiraz': 'Red', 
            u'Corvina, Rondinella, Molinara': 'Red', 
            u'RhÃ´ne-style Red Blend': 'Red', 
            u'Barbera': 'Red', 
            u'Pinot Gris': 'White',  
            u'Cabernet Franc': 'Red', 
            u'Sangiovese Grosso': 'Red', 
            u'Pinot Grigio': 'White',  
            u'Viognier': 'White', 
            u'Bordeaux-style White Blend': 'White', 
            u'Champagne Blend': 'White', 
            u'Port': 'Red', 
            u'GrÃ¼ner Veltliner': 'White', 
            }
topVarieties['index'] = topVarieties.index
topVarieties['color'] = topVarieties.apply(lambda row : ColorMap.get(row['index'],0), axis=1)

redVwhite = topVarieties.groupby('color').sum()
redVwhite = redVwhite.loc[['Red','White']]

plt.figure()
redVwhite.plot.pie(subplots=True,autopct='%.2f')
plt.show()


#### Plot 3: Highest Scoring Grape Varieties ########################
dtemp=df.loc[:,['variety','points']] #note: no points are missing; see df.count()
dtemp=dtemp[dtemp['variety'].isin(topVarieties.index)] #restrict to top varieties

summary_overall = dtemp.describe() #overall
summary_byvariety = dtemp.groupby('variety').describe()

plt.figure()
plt.title("Boxplot of Point Scores Across All Examples")
dtemp.boxplot()
plt.show()

plt.figure()
(summary_byvariety[('points','mean')]).sort_values(ascending=False).plot(kind='bar') #no big differences in score by grape
plt.title("Average Point Score by Grape Variety")
plt.show()

del dtemp, summary_overall, summary_byvariety



#### Plot 7: Countries per Grape Variety ########################
dtemp=df.loc[:,['variety','country']]
dtemp=dtemp[dtemp['variety'].isin(topVarieties.index)] #restrict to top varieties
dtemp['counter'] = 1

dtemp.groupby(('variety','country')).size()
dtemp = dtemp.groupby(('variety','country')).prod()
dtemp = dtemp.groupby('variety').count()
dtemp = dtemp.sort_values(by='counter',ascending=False)


plt.figure()
dtemp.plot(kind='bar')
plt.title("Number of Countries Producing a Grape Variety")
plt.show()

del dtemp