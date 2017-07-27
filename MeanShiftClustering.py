
# Analyze Titanic data using the MeanShift algorithms
# from sklearn.cluster
#
 
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import MeanShift
from sklearn import preprocessing

# clustering groups of data can provide insight 
# into how and why group data appear in certain
# clusters
#
style.use('ggplot')

'''
pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)  - 3rd class was at bottom of ship
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British Pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourgl Q = Queenstown; S = Southhampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''


# now: whether or not we can find some insights from this data
#
# we have the dataset: whether or not the pass survived
#
# separate these people into two groups:
# Q1: did they survive or not
# Q2: will K means sep two groups into live / death

cwd = os.getcwd()
#
os.chdir("C:\\ML_Data\\_Titanic")

df = pd.read_excel('titanic.xls')

# make a full copy of the orig. dataset
# will want to pass through Meanshift
# keeping orig. text feature set data
#
original_df = pd.DataFrame.copy(df) 

#print(df.head(5))

# which features will be important to cluster?
# name could be, but we'd need NLP to check for prestigious names
# sex could be, but it's non-numeric
# cabin is important, but it's non-numeric
# embarked is important, but it's non-numeric
# home.dest may be important, but it's non-numeric

# so what do we do with this data, since ML requires
# numerical data
# generally what we do is: take text column: take 
# set of sex.values is going to be female=0, male=1
# set of home.dest we just assign 0=city1, 1=city2
#
# with a lot of values, we may end up with a lot of outliers
#
# plus we have a lot of missing data that needs to be filled in
#

# first let's drop some non important columns
df.drop(['body','name'], 1, inplace=True)

#print(df.head(5))

# converts all of the cols to numeric
#
df.convert_objects(convert_numeric=True) 

df.fillna(0, inplace=True)

#print(df.head(5))


# define a function to handle non numeric data
#
def handle_non_numerical_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}  # exm: {'Female':0}
        
        def convert_to_int(val):
            return text_digit_vals[val] # returns numeric equiv to text
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

# reset the values of df{col] by mapping this function
# to the value that's in that column
#
            df[column] = list(map(convert_to_int, df[column])) 
    return df # return our modified dataframe
                    
# call our text to int converter
#
df = handle_non_numerical_data(df)

#print(df.head(5))
            
# we could have run this clustering before the ship
# set sail to determine beforehand who would survive
# and who would not
#
# after we've trained, we can then add new data 
# values to predict whether the outcome would be:
# survived or died
#

# we're doing unsupervised learning, so we don't
# need to do cross_validation;
#

# df.drop(['boat'], 1, inplace=True)

print(df.head(5))

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)

y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# now add a column to our original dataframe
#
original_df['cluster_group'] = np.nan # init with empty data

# now iterate through labels to 
# populate this new column
# where iloc[i] is the ith row of 
# our original dataset
#
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]
    

n_clusters_ = len(np.unique(labels)) 

# We've created cluster groups in the 
# past with 70% accuracy

survival_rates = {}  # key will be cluster_group index followed by the surv. rate

for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
    survival_cluster = temp_df[ (temp_df['survived']==1) ]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate
                 
print("survival_rates: ", survival_rates)

# Note: MeanShift() algos will produce slightly diff.
# results each time it's run

# Example output:
# survival_rates:  {0: 0.3780202650038971, 1: 0.8333333333333334, 2: 0.0}
# so group #1 (0 - 2), shows the highest survival rate of 83% which is very good
#
# Most likely: 
# group #0 is 2nd class passengers
# group #1 is 1st class passengers
# group #2 is 3rd class passengers

group_id = 2

# next, printout to confirm which class of 
# passengers were in group #1 (likely 1st class)
print("Cluster_grp #", group_id, ": ", original_df[ (original_df['cluster_group']==group_id)].describe() )


# What would be the survival rate of 1st class
# passengers in group # 0
#

cluster_0 = original_df[ (original_df['cluster_group']==group_id)]
cluster_0_fc = cluster_0[ (cluster_0['pclass']==1) ]
#cluster_0_fc.describe()
print("cluster_0_f: ", cluster_0_fc.head())


#### Output example:

"""



   pclass  survived  sex      age  sibsp  parch  ticket      fare  cabin  \
0       1         1    1  29.0000      0      0     742  211.3375    121   
1       1         1    0   0.9167      1      2     488  151.5500     12   
2       1         0    1   2.0000      1      2     488  151.5500     12   
3       1         0    0  30.0000      1      2     488  151.5500     12   
4       1         0    1  25.0000      1      2     488  151.5500     12   

   embarked  boat  home.dest  
0         2     2        242  
1         2    11        259  
2         2     0        259  
3         2     0        259  
4         2     0        259  

C:\Anaconda3\lib\site-packages\pandas\core\indexing.py:128: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  self._setitem_with_indexer(indexer, value)

survival_rates:  {0: 0.38492063492063494, 1: 0.041666666666666664, 2: 0.8666666666666667, 3: 0.1}

#Note that group #2 had the highest survival rate: 87%
#

Cluster_grp #1:             pclass     survived          age        sibsp        parch  \
count  1260.00000  1260.000000  1007.000000  1260.000000  1260.000000   
mean      2.29127     0.384921    29.968388     0.391270     0.294444   
std       0.83419     0.486770    14.227078     0.715262     0.640046   
min       1.00000     0.000000     0.166700     0.000000     0.000000   
25%       2.00000     0.000000    21.000000     0.000000     0.000000   
50%       3.00000     0.000000    28.000000     0.000000     0.000000   
75%       3.00000     1.000000    38.250000     1.000000     0.000000   
max       3.00000     1.000000    80.000000     4.000000     4.000000   

              fare        body  cluster_group  
count  1259.000000  118.000000         1260.0  
mean     29.397210  160.355932            0.0  
std      39.637802   97.339175            0.0  
min       0.000000    1.000000            0.0  
25%       7.895800   72.750000            0.0  
50%      13.500000  155.500000            0.0  
75%      28.856250  255.750000            0.0  
max     263.000000  328.000000            0.0  

"""

