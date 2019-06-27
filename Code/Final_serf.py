# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 21:36:08 2017

@author: abalas18
"""
import os

#os.chdir('your project folder')
import pandas as pd
import csv as csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go

from random import uniform
from plotly import tools 

from random import randint, random
from operator import add
from functools import reduce
import numpy as np
import math
from operator import itemgetter
import time

# SERF EAST
path =r'...\Data\M55_site6' # use your path
allFiles = glob.glob(path + "/*data_50*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)
Filter = pd.DataFrame()
Filter=frame[(frame.poa_irradiance >= 800) & (frame.poa_irradiance <= 1100) & (frame.module_temp_1 >= 20)&(frame.module_temp_1 <= 30) ]

Filter=Filter[['Date-Time','dc_power','module_temp_1','poa_irradiance']]
# TRANSLATE TO STC:
# Power at 1000 W/m2
Filter.loc[:,'dc_power']=(Filter['dc_power'] /Filter['poa_irradiance'])* 1000 


# Power at 25 Celcius (less than 25 C)
dt=Filter[(Filter.module_temp_1 - 25) <= 0]
perct_dt = (25 - dt.module_temp_1) * 0.4334/100 #temperature coefficient of -0.4334
dt.loc[:,'dc_power'] *= (1 +  perct_dt) 
# Power at 25 Celcius (greater than 25 C)
dt_pos=Filter[(Filter.module_temp_1 - 25) > 0]
perct_dt_pos = (dt_pos.module_temp_1 - 25) * 0.4334/100
dt_pos.loc[:,'dc_power'] = dt_pos['dc_power']- (dt_pos['dc_power']*(perct_dt_pos)) 

# Merge data
k=[dt,dt_pos]
k=pd.concat(k)
k['Date-Time'] = pd.to_datetime(k['Date-Time'])
Final=k.sort_values(by='Date-Time')

Final.index = Final['Date-Time']
Final_Mean = Final.resample('A').mean()
Final_Mean['Date-Time'] = Final_Mean.index
Final_Mean =Final_Mean.dropna(axis=0, how='any')
Final_Mean =Final_Mean[Final_Mean > 0].dropna()
Final_Mean = Final_Mean.reset_index(drop=True)

SERF_EAST = Final_Mean
d0 = date(1998, 1, 1) #Final_Mean['Date-Time'][0]
delta = SERF_EAST['Date-Time'] - d0
SERF_EAST['Days'] = delta.astype('timedelta64[D]')

SERF_EAST['Pmax_deg'] = 1 - ((6600 - SERF_EAST['dc_power'])/6600)
P_East = SERF_EAST['Pmax_deg']

P_East[5] = (P_East[4] + P_East[6])/2
P_East[7] = (P_East[6] + P_East[8])/2
P_East[12] = (P_East[11] + P_East[13])/2


# SERF WEST
path =r'...\Data\M55_site7' # use your path
allFiles = glob.glob(path + "/*data_51*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)
Filter = pd.DataFrame()
Filter=frame[(frame.poa_irradiance >= 800) & (frame.poa_irradiance <= 1100) & (frame.module_temp_1 >= 20)&(frame.module_temp_1 <= 30) ]

Filter=Filter[['Date-Time','dc_power','module_temp_1','poa_irradiance']]
# TRANSLATE TO STC:
# Power at 1000 W/m2
Filter.loc[:,'dc_power']=(Filter['dc_power'] /Filter['poa_irradiance'])* 1000 


# Power at 25 Celcius (less than 25 C)
dt=Filter[(Filter.module_temp_1 - 25) <= 0]
perct_dt = (25 - dt.module_temp_1) * 0.4334/100 #temperature coefficient of -0.4334
dt.loc[:,'dc_power'] *= (1 +  perct_dt) 
# Power at 25 Celcius (greater than 25 C)
dt_pos=Filter[(Filter.module_temp_1 - 25) > 0]
perct_dt_pos = (dt_pos.module_temp_1 - 25) * 0.4334/100
dt_pos.loc[:,'dc_power'] = dt_pos['dc_power']- (dt_pos['dc_power']*(perct_dt_pos)) 

# Merge data
k=[dt,dt_pos]
k=pd.concat(k)
k['Date-Time'] = pd.to_datetime(k['Date-Time'])
Final=k.sort_values(by='Date-Time')

Final.index = Final['Date-Time']
Final_Mean = Final.resample('A').mean()
Final_Mean['Date-Time'] = Final_Mean.index
Final_Mean =Final_Mean.dropna(axis=0, how='any')
Final_Mean =Final_Mean[Final_Mean > 0].dropna()
Final_Mean = Final_Mean.reset_index(drop=True)

SERF_WEST = Final_Mean
d0 = date(1998, 1, 1) #Final_Mean['Date-Time'][0]
delta = SERF_WEST['Date-Time'] - d0
SERF_WEST['Days'] = delta.astype('timedelta64[D]')

SERF_WEST['Pmax_deg'] = 1 - ((6600 - SERF_WEST['dc_power'])/6600)
P_West = SERF_WEST['Pmax_deg']



P_West.loc[7.5]=(P_West.loc[7] + P_West.loc[8])/2
P_West.loc[8.5]=(P_West.loc[8] + P_West.loc[9])/2
P_West = P_West.sort_index().reset_index(drop=True)

P_West[7] = (P_West[5] + P_West[6])/2
P_West[8] = (P_West[7] + P_West[9])/2
P_West[9] = (P_West[8] + P_West[10])/2
P_West[16] = (P_West[14] + P_West[15])/2

P_East[14] = P_West[14]
P_East[15] = P_West[15]
P_East[16] = P_West[16]

#P_West.plot()
#P_East.plot()
 # sorting by index
P_deg = (P_East + P_West)/2
#P_deg.plot()



#TRAINING DATASET
Final_Mean_Train = SERF_EAST[SERF_EAST['Date-Time']<='2010-12-31']

#TEST DATASET
Final_Mean_Test = SERF_EAST[SERF_EAST['Date-Time']>'2010-12-31']

P_deg_Train = P_deg[P_deg.index<12]
P_deg_Test = P_deg[P_deg.index>=12]



# Degradation plot Offline
data = [go.Scatter(
          x=P_deg.index,
          y=P_deg[:])]

layout = go.Layout(
    title='Degradation plot',
    xaxis=dict(
        title='Time',
        titlefont=dict(
            family='Courier New, monospace',
            size=25,
            color='32'
        )
    ),
    yaxis=dict(
        title='Pmax_degradation ()',
        titlefont=dict(
            family='Courier New, monospace',
            size=25,
            color='13'
        )
    )
)
fig = go.Figure(data=data, layout=layout)
plot(fig, filename='Colorado Weather Data')



#--------------------------------------------------->>>>>>>>>>>>
#WEATHER DATA

path =r'...\Data\Weather' # use your path
allFiles = glob.glob(path + "/*466290_39.73_-105.18_*.csv")
frame1 = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0, skiprows=2)
    list_.append(df)
frame1 = pd.concat(list_)

#pd.to_datetime(frame)
frame1['Date-Time'] = (pd.to_datetime(frame1[['Year', 'Month', 'Day', 'Hour', 'Minute']]))


azimuth = np.deg2rad(158)
array_azimuth = np.deg2rad(158)
zenith = np.deg2rad(frame1['Solar Zenith Angle'])
tilt = np.deg2rad(45) #Latitude tilt
AOI = np.rad2deg(np.arccos((np.cos(zenith)*np.cos(tilt)) + (np.sin(zenith)*np.sin(tilt)*np.cos(azimuth-array_azimuth))))
E_b = frame1['DNI'] * np.cos(np.deg2rad(AOI))

#E_g
Albedo = 0.33
E_g = frame1['GHI']*Albedo*((1-np.cos(tilt))/2)

#E_d
E_d = frame1['DHI'] * ((1-np.cos(tilt))/2) + (frame1['GHI'] * (((0.012*np.rad2deg(zenith) - 0.04)*(1-np.cos(tilt)))/2))

#POA Final
E_POA = E_b + E_g + E_d

#Module Temperature
T_mod = frame1['Temperature'] + (E_POA * (np.exp(-3.75 - (0.075 * frame1['Wind Speed']))))

frame1['T_mod']= T_mod
frame1['E_POA']= E_POA

frame1.index = frame1['Date-Time']
Cumul_roll= (frame1.resample('D')['T_mod'].agg(['min', 'max']))

Delta_T_mod = Cumul_roll['max'] - Cumul_roll['min']
T_mod_max = Cumul_roll['max']
UV = 0.05*(frame1.resample('D')['E_POA'].agg(['mean']))
RH_amb= (frame1.resample('D')['Relative Humidity'].agg(['mean']))

new_frame = pd.DataFrame()
new_frame['Delta_T_mod']= Delta_T_mod
new_frame['T_mod_max']= T_mod_max
new_frame['UV']= UV
new_frame['RH_amb']= RH_amb
new_frame['Date']= new_frame.index


d0 = new_frame['Date'][0]
delta = new_frame['Date'] - d0
new_frame['Days'] = delta.astype('timedelta64[D]')

#DROP INDEX
new_frame = new_frame.reset_index(drop=True)

#TRAINING DATASET
new_frame_Train = new_frame[new_frame['Date']<='2010-12-31']

#TEST DATASET
new_frame_Test = new_frame[new_frame['Date']>'2010-12-31']



trace1 = go.Scatter(x=new_frame['Date'], y=new_frame['T_mod_max'], showlegend=False)
trace2 = go.Scatter(x=new_frame['Date'], y=new_frame['Delta_T_mod'], showlegend=False)
trace3 = go.Scatter(x=new_frame['Date'], y=new_frame['UV'], showlegend=False)
trace4 = go.Scatter(x=new_frame['Date'], y=new_frame['RH_amb'], showlegend=False)

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Daily Maximum Module Temperature', 'Daily Cyclic Temperature', 'Daily Average UV Radiation', 'Daily Average Relative Humidity'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)

fig['layout']['xaxis1'].update(title='Time')
fig['layout']['xaxis2'].update(title='Time')
fig['layout']['xaxis3'].update(title='Time')
fig['layout']['xaxis4'].update(title='Time')

fig['layout']['yaxis1'].update(title='Module Temperature (Degree Celcius)')
fig['layout']['yaxis2'].update(title='Daily Cyclic Temperature (Degree Celcius)')
fig['layout']['yaxis3'].update(title='Daily Average UV Radiation (W/m2)')
fig['layout']['yaxis4'].update(title='Daily Average Relative Humidity (%)')

#fig['layout'].update(title='Colorado Weather Data')

plot(fig, filename='Colorado Weather Data')


#------------------------------------------------------->>>>>>>>>>
# Optimization GA

def individual(length, min_b0, max_b0, min_b1, max_b1, min_b2, max_b2, min_b3, max_b3, min_b4, max_b4):
    'Create a member of the population.'
    b0=[]
    b1=[]
    b2=[]
    b3=[]
    b4=[]
    b0 = uniform(min_b0,max_b0)
    b1 = uniform(min_b1,max_b1) 
    b2 = uniform(min_b2,max_b2)  
    b3 = uniform(min_b3,max_b3) 
    b4 = uniform(min_b4,max_b4) 
    return [b0, b1, b2, b3, b4]

def population(count, length, min_b0, max_b0, min_b1, max_b1, min_b2, max_b2, min_b3, max_b3, min_b4, max_b4):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [ individual(length, min_b0, max_b0, min_b1, max_b1, min_b2, max_b2, min_b3, max_b3, min_b4, max_b4) for x in range(count) ]

def fitness(individual, target):
#    global Ea
    A = individual[0]
    Ea = individual[1]
    n = individual[2]
    m = individual[3]
    b = individual[4]
    j=0
    Cum_Effect = []
    while (j<len(a)):
        effect = []
        for i in range(0,a[j]):
            effect.append(A* (math.exp(-Ea*11605*(1/(new_frame_Train.T_mod_max[i]+273)))) * (new_frame_Train.UV[i]**n) * (new_frame_Train.RH_amb[i]**m) * (new_frame_Train.Delta_T_mod[i]**b))

        effect=np.nan_to_num(effect)
        Cum_Effect.append(sum(effect))
        j=j+1 
        
    s = sum(np.array(target) - np.array(Cum_Effect))**2
    return s


def grade(pop, target):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x, target) for x in pop))
    return summed / (len(pop) * 1.0)

def evolve(pop, target, retain=0.1, random_select=0.05, mutate=0.01):
    graded = [ (fitness(x, target), x) for x in pop]
    graded = [ x[1] for x in (sorted(graded, key = itemgetter(0)))]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1) 
            individual[pos_to_mutate] = uniform(abs(min(individual)), abs(max(individual)))
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            child = [*male[:half] , *female[half:]]
            children.append(child)        
    parents.extend(children)
    return parents


"""
START
"""

a = Final_Mean_Train['Days']
a = a.astype(int)

target = (1 - P_deg_Train)
#target = P_deg_Train
#target[7] = (target[6] + target[8])/2
#target[9] = (target[7] + target[8])/2
#target.plot()
p_boot=[]
p_count = 100
i_length = 1
i_min_b0 = 0
i_max_b0 = 1
i_min_b1 = 0
i_max_b1 = 2
i_min_b2 = 0.6
i_max_b2 = 1
i_min_b3 = 0
i_max_b3 = 1.5
i_min_b4 = 2
i_max_b4 = 5

p = population(p_count, i_length, i_min_b0, i_max_b0, i_min_b1, i_max_b1, i_min_b2, i_max_b2, i_min_b3, i_max_b3, i_min_b4, i_max_b4)
p = np.around(p, decimals=4)
fitness_history = [grade(p, target),]
for i in range(100):
    tic = time.clock()
    p = np.around(evolve(p, target), decimals=4)
    fitness_history.append(grade(p, target))
    print (i)
    print (fitness_history)
    print (p[0])
    p_boot.append(p[0])
    toc = time.clock()
    print (toc - tic)
    
for datum in fitness_history:
    print (datum)
   
print (p[0])

SSE = fitness_history[-1]
Y_mean=np.mean(target)

SST = sum((target-Y_mean)**2)
R_Squared = 1- (SSE/SST)

#BOOTSTRAP
import scipy
import scikits.bootstrap as bootstrap

# compute 95% confidence intervals around the mean
CIs = bootstrap.ci(data=p[:,0], statfunction=scipy.mean, alpha=0.05, n_samples=20000)

print ("Bootstrapped 95% confidence intervals\nLow:", CIs[0], "\nHigh:", CIs[1])
	
#####################################



def Predict(individual, new1_df):
    A = individual[0]
    Ea = individual[1]
    n = individual[2]
    m = individual[3]
    b = individual[4]
    new1_df=new1_df.reset_index()
#    A = 0.1232
#    Ea = 0.853
#    n = 0.9586
#    m = 2.3453
#    b = 3.4054
    a = len(new1_df)
    effect = []
    for i in range(0,a):
        effect.append(A* (np.exp(-Ea*11605*(1/(new1_df.T_mod_max[i]+273)))) * (new1_df.UV[i]**n) * (new1_df.RH_amb[i]**m) * ((new1_df.Delta_T_mod[i])**b))

    effect=np.nan_to_num(effect)
    

    return effect


p = [[0.2827, 0.7284, 0.9514, 1.6013, 2.631]]
beta_ini = 0.12

Effect = pd.DataFrame(Predict(p[0], new_frame_Train))
Effect.columns = ['Deg']
Inst_Deg = 1- Effect
Cum_Effect = np.sum(Effect)
Cum_Inst = 1-(np.cumsum(Effect)) - beta_ini
Y_total = beta_ini + Cum_Effect


target_test = P_deg_Train
target_test=target_test.reset_index()

target_test.loc[-1] = [10,0.89]  # adding a row
target_test.index = target_test.index + 1  # shifting index
target_test = target_test.sort_index()


from numpy import *
import math
import matplotlib.pyplot as plt

#plt.show()
line1, = plt.plot(target_test.index*365,target_test['Pmax_deg'], 'r', label="Actual Degradation", linestyle='-')
#line1, = plt.plot(target_test.index*365,target_test[:], 'r', label="Actual Degradation", linestyle='-')
line2, = plt.plot(Cum_Inst.index, Cum_Inst['Deg'], label="Predicted Degradation", linestyle='--' )

# Create a legend for the first line.
first_legend = plt.legend(handles=[line1], loc=3)

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)

# Create another legend for the second line.
plt.legend(handles=[line2], loc=1)

plt.xlabel('Days')
plt.ylabel('Degradation')
plt.show()



----------------------#################----------------

# Degradation plot Offline
data = [go.Scatter(
          x=Cum_Inst.index,
          y=Cum_Inst['Deg'])]

layout = go.Layout(
    title='Degradation plot',
    xaxis=dict(
        title='Time',
#        titlefont=dict(
#            family='Courier New, monospace',
#            size=18,
#            color='#7f7f7f'
#        )
    ),
    yaxis=dict(
        title='Pmax_degradation ()',
#        titlefont=dict(
#            family='Courier New, monospace',
#            size=18,
#            color='#7f7f7f'
#        )
    )
)
fig = go.Figure(data=data, layout=layout)
plot(fig, filename='Colorado Weather Data')


