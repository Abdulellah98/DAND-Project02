#!/usr/bin/env python
# coding: utf-8

# # Project 2: TMDB Movies 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# In this project, I will analyze TMDB movies dataset based on multiple criteria which are genres, year of release, and more.
# I will walk through TMDB Movies dataset to answer multiple questions:
# - What is the most "genres" who has the highest average vote?
# - What is the most frequent average voting ?
# - Which genres has the highes and lowest run time ?
# - Which year has highest / lowest release movies 
# - Which year has highest / lowest average vote ?
# - Which movie has the highest profit ?
# - What is the movie who has the highest lost ?
# 

# In[125]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# - Upload the data and try take an overview on it and see its attribute
# - Check how many columns and rows there are 
# - Chech the data type and if there are a nulls values 
# - See the summary statistics of each attribute 
# - Drop the duplicated row and make sure its deleted by see how many row we have 
# - See the disribution of each attribute 
# - Drop some attributes that i will not use in my analysis e.g. id and imdb_id etc...
# - Replace the null with N/A.  
# - Droped the nulls in 'genres'.
# - Changed release_year data type to datetime. 
# - Replace the zero values in revenue_adj and budget_adj columns with mean.
# - Make sure everything is changed and it's clean. 

# In[126]:


df = pd.read_csv(r'C:\Users\Hp\Desktop\Data Analysis\tmdb-movies.csv')
df.head()


# In[127]:


df.shape


# In[128]:


df.info()


# In[129]:


df.describe()


# In[130]:


sum(df.duplicated())


# In[131]:


df.drop_duplicates(keep = 'first', inplace =True)


# In[132]:


df.shape


# In[133]:


df.hist(figsize=(10,10))


# ## Data Cleaning

# In[134]:


df.drop(['id', 'imdb_id', 'homepage', 'tagline', 'budget', 'revenue', 'keywords'], 
        axis = 1, inplace = True)
df.head()


# In[135]:


df['director'].fillna('N/A', inplace = True)
df['production_companies'].fillna('N/A', inplace = True)
df['overview'].fillna('N/A', inplace = True)
df['cast'].fillna('N/A', inplace = True)


# In[136]:


df.dropna(inplace = True)


# In[137]:


df['release_date'] = pd.to_datetime(df['release_date'])


# In[138]:


budget_mean = df['budget_adj'].mean()
revenue_mean = df['revenue_adj'].mean()
runtime_mean = df['runtime'].mean()
budget_mean, revenue_mean, runtime_mean


# In[139]:


df['budget_adj']= df['budget_adj'].replace(0, budget_mean)
df['revenue_adj']= df['revenue_adj'].replace(0, revenue_mean)
df['runtime']= df['runtime'].replace(0, runtime_mean)
df.describe()


# In[140]:


df.info()


# In[141]:


df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# ### Research Question 1: What is the most "genres" who has the highest average vote?

# In[142]:


df.groupby('genres').vote_average.mean()


# #### the most average vote is "War|Drama|Action|Adventure|History" has an average with 7.8 out of 10 and that makes a sense because a lot of people who prefer that kind of movies and the leats average vote is "Western|Horror|Thriller" and maybe because many people don't like horror movies

# ### Research Question 2: What is the most frequent average voting ?

# In[143]:


df.groupby('genres').vote_average.mean().plot(kind = 'hist', figsize = (8,8), fontsize = 10) 
plt.title('The most frequnet average voting based on genres', fontsize = 15)
plt.legend();


# ### We can notice the most frequent average voting is 6 and that makes sense because a lot of movies has this average.

# ### Research Question 3: Which genres has the highes and lowest run time ?

# In[144]:


df.groupby(['genres']).runtime.mean()


# ### The historical movies (War|Drama|Action|Adventure|History) its take the most average runtime 540 min and that because its a real story so it takes time to cover, and family & fantasy movies (Action|Adventure|Animation|Family|Fantasy) its take the lest average runtime 56 min and maybe that because they don't want the kids to get bored.

# ### Research Question 4: Which year has highest / lowest release movies 

# In[145]:


df.groupby(['release_year']).genres.count().plot(kind='barh', figsize=(12,12), fontsize = 10) 
plt.title('Which year has highest / lowest release movies', fontsize = 15)
plt.legend();


# ### The highest year whose release is 2014 by 699 movies and the lowest are 1961 and 1969 31 movies.

# ### Research Question 5: Which year has highest / lowest average vote ? 

# In[146]:


df.groupby('release_year').vote_average.mean()


# ### In 1973 has the highest average vote 6.7 and I have checked the movies in that year and nothing was very popular in that year but I noticed that a lot of movies the vote is between 7 and 8 and I think that the reason.       And 2012 has the lowest average vote 5.79 and also I have checked the movies and I was shocked by the greatest movies was released in that year e.g.(The Dark Knight Rises, Django Unchained, & The Avengers, etc...) and that because there are more than 100 (18%)out of 584 movies the vote in between 1 - 2.5. 

# In[147]:


df['Profit'] = df['revenue_adj'] - df['budget_adj']
df.head()


# ###  Research Question 6: Which movie has the highest profit ?

# In[148]:


high_profit = df['Profit'].idxmax
df.loc[high_profit]


# ### Star Wars 1977 is the highest profit its genres is Adventure|Action|Science with a profit $2,750,140,000

# ### Research Question 7: What is the movie who has the highest lost ?

# In[149]:


low_profit = df['Profit'].idxmin
df.loc[low_profit]


# ### The Warrior's Way 2010 has the highest lost -$413,912,000 and its genres are Adventure|Fantasy|Action|Western|Thriller

# ### Research Question 8: Does the popularity affected by the average vote?

# In[150]:


df.plot(x='vote_average', y='popularity', kind='scatter', figsize = (8,8), fontsize = 10, label= 'popularity') 
plt.title('Which year has highest / lowest release movies', fontsize = 15)
plt.legend();


# ### As we can see in the chart the answer is Yes because of the vote average increase the popularity will increase as well.

# ### Research Question 8: Does the popularity of the movies increase over the years?

# In[151]:


df.plot(x='release_year', y='popularity', kind='scatter', figsize = (8,8), fontsize = 10, label= 'popularity') 
plt.title('Does the popularity increasing over the years?', fontsize = 15)
plt.legend();


# ### We can notice the popularity is vacillating over the years but in general it's increasing

# <a id='conclusions'></a>
# ## Conclusions
#  
# first of all, I try to discover and understand the data by seeing its attributes and see how many columns and rows and see if there are any null values and check the data type and see the summary statistics of each attributes then a try to find if there is any duplicated row then I delete it finally I try to see the distribution of each attribute. 
# 
# secondly, I start cleaning by drop some attributes that will be useless such as id, imdb_id, homepage, tagline, keywords, etc...
# then I replace the nulls values with N/A and the one who has a few nulls I dropped them. 
# 
# finally, I start data exploratory by some asking questions such as see the most "genres" who has the highest average voting, Which genres has the highest and lowest run time, Which year has highest / lowest average vote and finally see the movie who has the highest profit and highest lost. 
# 
# ### Limitations
# 1- There is a duplicate row and I dropped it so the analysis be correct. 
# 
# 2- I dropped some extraneous columns in the data wangling process. Some of them may yield other useful results, e.g. id, tmdb_id, homepage, etc...
# 
# 3- There are a lot of attributes that have null values some of them I replace it with N/A because it has a high number of null and it doesn't affect the analysis directly and the other I dropped them because it few and it affects the analyzing.
# 
# 4- There are many of zero values in revenue_adj and budget_adj columns.I replace the zero value with mean. There might be alternative ways to better process to this kind of issue. 
# 
# 5- Some of the movies have more than one genre and that will affect the level of accuracy of the analysis.
