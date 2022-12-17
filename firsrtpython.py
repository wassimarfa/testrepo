import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
haberdata=pd.read_csv('haberman.csv')
#haberdata


print(haberdata.head())

print(haberdata.shape)
#will give us the number of rows and columns

print(haberdata.columns)
#will get to know the name of columns.

haberdata['status'].value_counts()

haberdata.plot(kind='scatter',x='age',y='nodes')
plt.title('Scatterplot of age vs nodes')
plt.show()


sns.set_style('whitegrid')
sns.FacetGrid(haberdata,hue='status',height=5) \
.map(plt.scatter,'age','nodes')\
.add_legend()
plt.title('Scatterplot using seasborn of age vs nodes')
plt.show()

sns.set_style('whitegrid')
sns.pairplot(haberdata,hue='status',height=3)
plt.show(
  
  status_1=haberdata.loc[haberdata['status']==1]
status_2=haberdata.loc[haberdata['status']==2]
plt.plot(status_1['nodes'],np.zeros_like(status_1['nodes']),'o')
plt.plot(status_2['nodes'],np.zeros_like(status_2['nodes']),'o')
plt.title('1-D plotting of nodes')

plt.show()
  
  
  #PDF for age
sns.FacetGrid(haberdata,hue='status', height=8)\
.map(sns.distplot,'age')\
.add_legend()
plt.title('Histogram of age')
  
  
  #PDF for nodes
sns.FacetGrid(haberdata,hue='status',height=8)\
.map(sns.distplot,'nodes')\
.add_legend()
plt.title('Histogram of nodes')
plt.show()
  
  
  #PDF for year
sns.FacetGrid(haberdata,hue='status',height=8)\
.map(sns.distplot,'year')\
.add_legend()
plt.title('Histogram of Year')
plt.show()
  
  
  
  counts, bin_edges = np.histogram(status_1['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
arr1,=plt.plot(bin_edges[1:],pdf);
arr2,=plt.plot(bin_edges[1:],cdf)
plt.title('CDF and PDF of people living more than 5 years')
plt.legend([arr1,arr2], ['PDF survived','CDF survived'])
plt.ylabel('count')
plt.xlabel('nodes')
  
  
  counts, bin_edges = np.histogram(status_2['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
arr1,=plt.plot(bin_edges[1:],pdf);
arr2,=plt.plot(bin_edges[1:],cdf)
plt.title('CDF and PDF of people living less than 5 years')
plt.legend([arr1,arr2], ['PDF survived','CDF survived'])
plt.ylabel('count')
plt.xlabel('nodes')
  
  
  print('The means are:- ')
print(np.mean(status_1['nodes']))
print(np.mean(status_2['nodes']))
print('The standard deviation are:- ')
print(np.std(status_1['nodes']))
print(np.std(status_2['nodes']))
  
  
  print('\nMedians are:- ')
print(np.median(status_1['nodes']))
print(np.median(status_2['nodes']))
print('\nThe Quantiles are:- ')
print(np.percentile(status_1['nodes'],np.arange(0,100,25)))
print(np.percentile(status_2['nodes'],np.arange(0,100,25)))
print('\nThe 95th% is:- ')
print(np.percentile(status_1['nodes'],95))
print(np.percentile(status_2['nodes'],95))

from statsmodels import robust
print ("\nMedian Absolute Deviation is:- ")
print(robust.mad(status_1["nodes"]))
print(robust.mad(status_2["nodes"]))
  
  
  
  
  #Box plot
sns.boxplot(x='status',y='nodes',data=haberdata)
plt.legend
plt.title('Box plot')
plt.show
  
  
  #Violin plot
sns.violinplot(x='status',y='nodes',data=haberdata)
plt.legend
plt.title('Violin plot')

plt.show

  
  
  
  #Contour plots
sns.jointplot(x='age',y='nodes',data=status_1,kind='kde')
plt.legend
plt.title('Joint plot or Contour plot')

plt.show
  
  
  
  
