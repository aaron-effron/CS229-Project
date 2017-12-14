from util import importData
from util import CLASS_MAP
import pandas as pd

# Datasets for machine learning tasks
#---------------------------------------------------------------------------------------------------------------------

#Import and process raw data
df = importData("data-raw/winemag-data_first150k.csv",censor=True,filter=True,processDescriptions=True) 

#Keep only necessary columns
df_keep = df[['index','color','class','description']]

#Parameters to subset data
num_test = 20000
num_dev = 20000
num_train = df_keep.shape[0] - num_test - num_dev
assert num_train > 0

#Subset Data
data = df_keep.sample(frac=1,replace=False,random_state=1415926).reset_index(drop=True)
data_test = data.iloc[0:num_test]
data_dev = data.iloc[num_test:(num_test+num_dev)]
data_train = data.iloc[(num_test+num_dev):(num_test+num_dev+num_train)]

#Save to disk
data_test.to_csv('data-processed/data.test',index=False)
data_dev.to_csv('data-processed/data.dev',index=False)
data_train.to_csv('data-processed/data.train',index=False)


# Dataset for human benchmark
#---------------------------------------------------------------------------------------------------------------------

#Import raw data without processing text, except for censoring
df_human = importData("data-raw/winemag-data_first150k.csv",censor=True,filter=True,processDescriptions=False)

#Keep only necessary columns
df_human_keep = df_human[['index','color','class','description']]

#Subset Data
data_human = df_human_keep.sample(frac=1,replace=False,random_state=1415926).reset_index(drop=True)
data_test_h = data_human.iloc[0:num_test]

#Resample 100 examples
data_test_h_sample = data_test_h.sample(n=100,replace=False,random_state=71659).reset_index(drop=True)

#Save to disk
data_test_h_sample.to_csv('data-processed/data.testh',index=False)

#Save CLASS_MAP to disk
map = pd.DataFrame({'class':[i for i in range(len(CLASS_MAP))],'variety':CLASS_MAP})
map.to_csv('data-processed/variety.class',index=False)