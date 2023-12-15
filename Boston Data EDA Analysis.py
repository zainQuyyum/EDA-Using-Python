#!/usr/bin/env python
# coding: utf-8

# ***Introduction***

# *This Exploratory Data Analysis (EDA) examines crime incident reports in the city of Boston from June 2015 to October 2018.*

# In[1]:


# Load Necessery libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Load Boston Crime Dataset 
data = pd.read_csv("crime.csv",encoding = 'latin-1')
data


# ***About Dataset***

# *The data is originally provided by Boston's open data hub*

# In[3]:


print(len(data))


# In[4]:


data.info()


# In[5]:


data.describe()


# *Initially, let's streamline and simplify this dataset. I'll concentrate on the two years with comprehensive data (2016 and 2017). Furthermore, I'll zero in on UCR Part One offenses, encompassing solely the gravest crimes*

# In[6]:


# Keep only data from complete years (2016, 2017)
data = data.loc[data['YEAR'].isin([2016,2017])]

# Keep only data on UCR Part One offenses
data = data.loc[data['UCR_PART'] == 'Part One']

# Remove unused columns
data = data.drop(['INCIDENT_NUMBER','OFFENSE_CODE','UCR_PART','Location'], axis=1)

# Convert OCCURED_ON_DATE to datetime
data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'])

# Fill in nans in SHOOTING column
data.SHOOTING.fillna('N', inplace=True)

# Convert DAY_OF_WEEK to an ordered category
data.DAY_OF_WEEK = pd.Categorical(data.DAY_OF_WEEK, 
              categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
              ordered=True)

# Replace -1 values in Lat/Long with Nan
data.Lat.replace(-1, None, inplace=True)
data.Long.replace(-1, None, inplace=True)

# Rename columns to something easier to type (the all-caps are annoying!)
rename = {'OFFENSE_CODE_GROUP':'Group',
         'OFFENSE_DESCRIPTION':'Description',
         'DISTRICT':'District',
         'REPORTING_AREA':'Area',
         'SHOOTING':'Shooting',
         'OCCURRED_ON_DATE':'Date',
         'YEAR':'Year',
         'MONTH':'Month',
         'DAY_OF_WEEK':'Day',
         'HOUR':'Hour',
         'STREET':'Street'}
data.rename(index=str, columns=rename, inplace=True)

# Check
data.head()


# ***Tpyes of Violations***

# *Let's begin by examining the occurrence of various crime categories. As we've narrowed down our focus to only 'severe' offenses, we now have just 9 distinct types of violations – significantly more manageable compared to the initial 67*

# In[7]:


# A few more data checks
data.dtypes
data.isnull().sum()
data.shape


# In[8]:


# Countplot for crime types
sns.catplot(y='Group',
           kind='count',
            height=8, 
            aspect=1.5,
            order=data.Group.value_counts().index,
           data=data);


# ***Timing Patterns of Major Offenses***

# *We can examine trends across multiple timeframes, including daily hours, weekdays, and months throughout the year*

# In[9]:


# Crimes by hour of the day
sns.catplot(x='Hour',
           kind='count',
            height=8.27, 
            aspect=3,
            color='red',
           data=data)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel('Hour', fontsize=40)
plt.ylabel('Count', fontsize=40)
plt.show()


# In[ ]:





# In[10]:


# Crimes by day of the week
sns.catplot(x='Day',
           kind='count',
            height=8, 
            aspect=3,
           data=data)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel('')
plt.ylabel('Count', fontsize=40);


# In[11]:


# Crimes by month of year
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
sns.catplot(x='Month',
           kind='count',
            height=8, 
            aspect=3,
            color='blue',
           data=data)
plt.xticks(np.arange(12), months, size=30)
plt.yticks(size=30)
plt.xlabel('')
plt.ylabel('Count', fontsize=40);


# *Crime rates exhibit a dip between 1-8 in the morning, gradually escalating throughout the day, reaching a peak around 6 pm. There is variability across weekdays, with Friday recording the highest crime rate and Sunday the lowest. Additionally, months appear to play a role, as the winter period from February to April reports lower crime rates, while the summer and early fall months from June to October exhibit the highest rates. Notably, there is a spike in crime rates in January.
# 
# Are there any other time-related factors correlated with crime? According to crime analysts, certain offenses tend to surge during holidays, especially larceny and robbery. This can be attributed to various factors: crowded shopping centers provide more opportunities for thieves, travelers' homes become more susceptible to burglary, and heightened alcohol and drug consumption can elevate the likelihood of conflict-related crime. Let's investigate if our data supports these findings, focusing on the year 2017. I've also included a couple of non-official holidays known for rowdiness in Boston: St. Patrick's Day and the Boston Marathon*

# In[12]:


# Create data for plotting
data['Day_of_year'] = data.Date.dt.dayofyear
data_holidays = data[data.Year == 2017].groupby(['Day_of_year']).size().reset_index(name='counts')

# Dates of major U.S. holidays in 2017
holidays = pd.Series(['2017-01-01', # New Years Day
                     '2017-01-16', # MLK Day
                     '2017-03-17', # St. Patrick's Day
                     '2017-04-17', # Boston marathon
                     '2017-05-29', # Memorial Day
                     '2017-07-04', # Independence Day
                     '2017-09-04', # Labor Day
                     '2017-10-10', # Veterans Day
                     '2017-11-23', # Thanksgiving
                     '2017-12-25']) # Christmas
holidays = pd.to_datetime(holidays).dt.dayofyear
holidays_names = ['NY',
                 'MLK',
                 'St Pats',
                 'Marathon',
                 'Mem',
                 'July 4',
                 'Labor',
                 'Vets',
                 'Thnx',
                 'Xmas']

import datetime as dt
# Plot crimes and holidays
fig, ax = plt.subplots(figsize=(11,6))
sns.lineplot(x='Day_of_year',
            y='counts',
            ax=ax,
            data=data_holidays)
plt.xlabel('Day of the year')
plt.vlines(holidays, 20, 80, alpha=0.5, color ='r')
for i in range(len(holidays)):
    plt.text(x=holidays[i], y=82, s=holidays_names[i])


# ***Locations of Major Offenses***

# *We can utilize the latitude and longitude columns to visualize the crime locations in Boston. When we set the alpha parameter to a very small value, we can identify certain areas with a high frequency of crimes, often referred to as 'hotspots*

# In[13]:


# Simple scatterplot
sns.scatterplot(x='Lat',
               y='Long',
                alpha=0.01,
               data=data);


# *That indeed resembles Boston. If you have any familiarity with the city, it won't be too unexpected to observe that the downtown area exhibits the most concentrated points of darkness. However, there are also certain areas beyond the city center with notably elevated crime rates. Let's create an additional scatterplot, but now we'll assign colors to points based on districts to identify which districts experience the highest crime rates*

# In[14]:


# Plot districts
sns.scatterplot(x='Lat',
               y='Long',
                hue='District',
                alpha=0.01,
               data=data)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2);


# *We can now link elevated crime rates to specific districts, notably A1 and D4, which align with the bustling areas of downtown Boston. Another region with significantly high crime rates is evident in district D14.*
# 
# *To enhance visual appeal, let's utilize Folium to generate an interactive heatmap illustrating Boston's crime distribution. For this visualization, I'll focus exclusively on the data from 2017*

# ***Key Findings***

# ***1-Larceny stands out as the predominant type of severe crime.***
# 
# ***2-Serious offenses are more prone to happen during the afternoon and evening.***
# 
# ***3-Fridays witness the highest likelihood of serious crimes, while Sundays experience the lowest.***
# 
# ***4-The summer and early fall months record the highest incidence of serious crimes, contrasting with lower rates during winter (excluding January, which resembles summer rates).***
# 
# ***5-There is no evident correlation between major holidays and crime rates.***
# 
# ***6-City center regions, particularly districts A1 and D4, exhibit the highest occurrence of serious crimes***

# ***Conclusion***

# *This exploratory data analysis provides only a preliminary examination of the dataset. Additional analyses could delve into the temporal and spatial variations of different crime types. I didn't delve into the less severe UCR Part Two and Part Three crimes, which are more prevalent than Part One crimes but encompass intriguing categories like drug-related offenses. Another intriguing avenue would be to integrate this data with other Boston-related information, such as demographics or even weather data, to explore the factors that predict crime rates over time and in different locations.*

# In[ ]:




