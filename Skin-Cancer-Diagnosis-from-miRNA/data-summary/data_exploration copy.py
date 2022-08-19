import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.preprocessing import MinMaxScaler

"""
Compiles data from multiple files into a single file

Author: Kostas Mateer
"""

#creation of dataframe
data = pd.read_csv("data/Breast Invasive Carcinoma/0a2a3cf2-8c9e-4d2a-9da3-7e04ffb1a032/d243560f-fec1-4803-b704-dde9f10f7aa2.mirbase21.mirnas.quantification.txt", delimiter='\t')
headers = np.array(data['miRNA_ID'])
headers = np.append(headers, 'cancer')
df = pd.DataFrame(columns=headers)

#compiling the data into one data frame
print("---------------------- running data compilation ----------------------")
count = 0
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if('.txt' in os.path.join(root, name) and not 'MANIFEST' in os.path.join(root, name) 
        and not 'annotations' in os.path.join(root, name)):
            print('-----------------------------------------')
            print('FILE: '+ os.path.join(root,name))
            print('-----------------------------------------')
            count+=1
            if('Breast Invasive Carcinoma' in os.path.join(root, name)):
                data = pd.read_csv(os.path.join(root, name), delimiter='\t')
                array = np.array(data['reads_per_million_miRNA_mapped'])
                array = np.append(array, 0)
                df.loc[len(df.index)] = array
            elif ('Kidney Renal Clear Cell Carcinoma' in os.path.join(root, name)):
                data = pd.read_csv(os.path.join(root, name), delimiter='\t')
                array = np.array(data['reads_per_million_miRNA_mapped'])
                array = np.append(array, 1)
                df.loc[len(df.index)] = array
            elif ('Lung Adenocarcinoma' in os.path.join(root, name)):
                data = pd.read_csv(os.path.join(root, name), delimiter='\t')
                array = np.array(data['reads_per_million_miRNA_mapped'])
                array = np.append(array, 2)
                df.loc[len(df.index)] = array
            elif ('Lung Squamous Cell Carcinoma' in os.path.join(root, name)):
                data = pd.read_csv(os.path.join(root, name), delimiter='\t')
                array = np.array(data['reads_per_million_miRNA_mapped'])
                array = np.append(array, 3)
                df.loc[len(df.index)] = array
            elif ('Pancreatic Adenocarcinoma' in os.path.join(root, name)):
                data = pd.read_csv(os.path.join(root, name), delimiter='\t')
                array = np.array(data['reads_per_million_miRNA_mapped'])
                array = np.append(array, 4)
                df.loc[len(df.index)] = array
            elif ('Uveal Melanoma' in os.path.join(root, name)):
                data = pd.read_csv(os.path.join(root, name), delimiter='\t')
                array = np.array(data['reads_per_million_miRNA_mapped'])
                array = np.append(array, 5)
                df.loc[len(df.index)] = array
print("amount of files compiled: ", count)
print()

print("shape of dataframe before feature removal: ", df.shape)
#removing uneccessary features with no values
print()
print("---------------------- running feature removal ----------------------")
count = 0
for column in df:
    column_array = np.array(df[column])
    if np.sum(column_array) == 0:
        count +=1
        df = df.drop(column, axis=1)
print("amount of 0 sum features: ", count)
print("shape of dataframe after feature removal: ", df.shape)

#saving a csv file to use
df.to_csv('data/compiled_data.csv', index=False)

# 0 = Breast Invasive Carcinoma
# 1 = Kidney Renal Clear Cell Carcinoma
# 2 = Lung Adenocarcinoma
# 3 = Lung Squamous Cell Carcinoma
# 4 = Pancreatic Adenocarcinoma
# 5 = Uveal Melanoma

