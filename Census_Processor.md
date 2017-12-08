---
title: MERGE EDUCATION ATTAINMENT
notebook: Census_Processor.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}


The census data was downloaded from https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml. The following data was downloaded per msa:<br>
1. Age Group Demographics
2. Gender Demographics
3. Marital Status
4. Race Demographics
5. Education Levels
6. Income Statistics
7. Real Estate Vacancy Rates

### Importing Libraries and Defining functions



```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
import random
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
%matplotlib inline

import csv
from sklearn import ensemble
import math
from sklearn.metrics import confusion_matrix

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

sns.set_context('notebook')
sns.set_style("darkgrid")

import requests
from bs4 import BeautifulSoup
from IPython.display import IFrame, HTML

import warnings
warnings.filterwarnings('ignore')
```




```python
def get_axs(rows, columns, fig_size_width, fig_size_height):
    dims = (fig_size_width, fig_size_height)
    fig, axs = plt.subplots(rows, columns, figsize=dims)
    if(rows*columns>1):
         axs = axs.ravel()
            
def get_int(s):
    return_value = np.nan
    if s!=None:
        try:
            return_value = int(s)
            return return_value
        except ValueError:
            return return_value
    else:
        return return_value
    
def convert_to_int(row):
    return_value = []
    return_value = [get_int(i) for i in row]
    return return_value
```


### Read in the downloaded datasets



```python
age_gender_dict ={}
race_dict = {}
marital_dict = {}
vacancy_dict = {}
income_dict = {}
edu_dict = {}

for i in range(2006,2017):
    age_gender_dict[i] = pd.DataFrame(pd.read_csv('data/agegender_'+str(i)+'.csv', encoding='latin-1'))
    race_dict[i] = pd.DataFrame(pd.read_csv('data/race_'+str(i)+'.csv', encoding='latin-1'))
    marital_dict[i] = pd.DataFrame(pd.read_csv('data/marital_'+str(i)+'.csv', encoding='latin-1'))
    vacancy_dict[i] = pd.DataFrame(pd.read_csv('data/vacancy_'+str(i)+'.csv', encoding='latin-1'))
    income_dict[i] = pd.DataFrame(pd.read_csv('data/income_'+str(i)+'.csv', encoding='latin-1'))
    edu_dict[i] = pd.DataFrame(pd.read_csv('data/edu_'+str(i)+'.csv', encoding='latin-1'))
```


### Merging all years



```python
##### MERGE DATA

merged_2 = {}
merged_3 = {}
merged_4 = {}
merged_5 = {}
merged_all = {}

for i in range(2006,2017):
    merged_2[i] = pd.merge(age_gender_dict[i], race_dict[i], left_on=('Id2'), right_on=str(i)+'Id2', how='left')
    merged_3[i] = pd.merge(merged_2[i], marital_dict[i], left_on=(str(i)+'Id2'), right_on=str(i)+'Id2', how='left')
    merged_4[i] = pd.merge(merged_3[i], vacancy_dict[i], left_on=(str(i)+'Id2'), right_on=str(i)+'Id2', how='left')
    merged_5[i] = pd.merge(merged_4[i], income_dict[i], left_on=(str(i)+'Id2'), right_on=str(i)+'Id2', how='left')
    merged_all[i] = pd.merge(merged_5[i], edu_dict[i], left_on=(str(i)+'Id2'), right_on=str(i)+'Id2', how='left')
```




```python
merged_all[2006].head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Id2</th>
      <th>Geography</th>
      <th>total</th>
      <th>male_total</th>
      <th>male_under5</th>
      <th>male_5to9</th>
      <th>male_10-14</th>
      <th>male_15-17</th>
      <th>male_18-19</th>
      <th>...</th>
      <th>2006Id</th>
      <th>2006Geography</th>
      <th>2006Total25plus</th>
      <th>2006Less than 9th grade</th>
      <th>2006 9th to 12th grade, no diploma</th>
      <th>2006High school graduate (includes equivalency)</th>
      <th>2006Some college, no degree</th>
      <th>2006Associate's degree</th>
      <th>2006Bachelor's degree</th>
      <th>2006Graduate or professional degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3100000US10180</td>
      <td>10180</td>
      <td>Abilene, TX Metro Area</td>
      <td>158548</td>
      <td>78912</td>
      <td>5642</td>
      <td>4648</td>
      <td>6435</td>
      <td>3578</td>
      <td>2398</td>
      <td>...</td>
      <td>3100000US10180</td>
      <td>Abilene, TX Metro Area</td>
      <td>98495.0</td>
      <td>7387.0</td>
      <td>14282.0</td>
      <td>28170.0</td>
      <td>22457.0</td>
      <td>5910.0</td>
      <td>13888.0</td>
      <td>6501.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3100000US10380</td>
      <td>10380</td>
      <td>Aguadilla-Isabela-San Sebastián, PR Metro Area</td>
      <td>336502</td>
      <td>166686</td>
      <td>11601</td>
      <td>11870</td>
      <td>13496</td>
      <td>8074</td>
      <td>5566</td>
      <td>...</td>
      <td>3100000US10380</td>
      <td>Aguadilla-Isabela-San Sebastián, PR Metro Area</td>
      <td>216337.0</td>
      <td>70742.0</td>
      <td>25311.0</td>
      <td>54084.0</td>
      <td>19687.0</td>
      <td>12980.0</td>
      <td>26393.0</td>
      <td>7355.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3100000US10420</td>
      <td>10420</td>
      <td>Akron, OH Metro Area</td>
      <td>700943</td>
      <td>337619</td>
      <td>21106</td>
      <td>22114</td>
      <td>24253</td>
      <td>14953</td>
      <td>10045</td>
      <td>...</td>
      <td>3100000US10420</td>
      <td>Akron, OH Metro Area</td>
      <td>466484.0</td>
      <td>11196.0</td>
      <td>41517.0</td>
      <td>160470.0</td>
      <td>93763.0</td>
      <td>31721.0</td>
      <td>84900.0</td>
      <td>42917.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3100000US10500</td>
      <td>10500</td>
      <td>Albany, GA Metro Area</td>
      <td>165062</td>
      <td>78572</td>
      <td>6720</td>
      <td>6167</td>
      <td>6411</td>
      <td>4443</td>
      <td>4152</td>
      <td>...</td>
      <td>3100000US10500</td>
      <td>Albany, GA Metro Area</td>
      <td>101195.0</td>
      <td>7084.0</td>
      <td>12346.0</td>
      <td>31876.0</td>
      <td>23983.0</td>
      <td>6679.0</td>
      <td>12143.0</td>
      <td>6982.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3100000US10580</td>
      <td>10580</td>
      <td>Albany-Schenectady-Troy, NY Metro Area</td>
      <td>850957</td>
      <td>413205</td>
      <td>23804</td>
      <td>24299</td>
      <td>27849</td>
      <td>17305</td>
      <td>16144</td>
      <td>...</td>
      <td>3100000US10580</td>
      <td>Albany-Schenectady-Troy, NY Metro Area</td>
      <td>573079.0</td>
      <td>18339.0</td>
      <td>39542.0</td>
      <td>174789.0</td>
      <td>92266.0</td>
      <td>63039.0</td>
      <td>102008.0</td>
      <td>83670.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 95 columns</p>
</div>



### Drop Redundant Columns



```python
merged_all_clean = {}

for i in range(2006,2017):
    merged_all_clean[i] = merged_all[i].drop(['Unnamed: 0_x', str(i)+'Geography_x', 'Unnamed: 0_y', str(i)+'Id_y', str(i)+'Id2', str(i)+'Geography_y', 'Unnamed: 0_x',str(i)+'Id_x', str(i)+'Geography_x', str(i)+'Geography_y', 'Unnamed: 0_x', str(i)+'Unnamed: 0', str(i)+'Id_x', str(i)+'Geography_x', 'Unnamed: 0_y', str(i)+'Id_y', str(i)+'Geography_y'], axis =1)
```




```python
merged_all_clean[2016].head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Id2</th>
      <th>Geography</th>
      <th>total</th>
      <th>male_total</th>
      <th>male_under5</th>
      <th>male_5to9</th>
      <th>male_10-14</th>
      <th>male_15-17</th>
      <th>male_18-19</th>
      <th>...</th>
      <th>2016Id</th>
      <th>2016Geography</th>
      <th>2016Total25plus</th>
      <th>2016Less than 9th grade</th>
      <th>2016 9th to 12th grade, no diploma</th>
      <th>2016High school graduate (includes equivalency)</th>
      <th>2016Some college, no degree</th>
      <th>2016Associate's degree</th>
      <th>2016Bachelor's degree</th>
      <th>2016Graduate or professional degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>310M300US10180</td>
      <td>10180</td>
      <td>Abilene, TX Metro Area</td>
      <td>170860</td>
      <td>87459</td>
      <td>5571</td>
      <td>6315</td>
      <td>5340</td>
      <td>3113</td>
      <td>3663</td>
      <td>...</td>
      <td>310M300US10180</td>
      <td>Abilene, TX Metro Area</td>
      <td>109703.0</td>
      <td>5479.0</td>
      <td>9154.0</td>
      <td>40224.0</td>
      <td>25198.0</td>
      <td>7304.0</td>
      <td>15103.0</td>
      <td>7241.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>310M300US10380</td>
      <td>10380</td>
      <td>Aguadilla-Isabela, PR Metro Area</td>
      <td>309764</td>
      <td>153695</td>
      <td>7976</td>
      <td>9909</td>
      <td>9401</td>
      <td>7580</td>
      <td>4089</td>
      <td>...</td>
      <td>310M300US10380</td>
      <td>Aguadilla-Isabela, PR Metro Area</td>
      <td>217353.0</td>
      <td>50408.0</td>
      <td>20760.0</td>
      <td>62332.0</td>
      <td>26186.0</td>
      <td>19000.0</td>
      <td>27709.0</td>
      <td>10958.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>310M300US10420</td>
      <td>10420</td>
      <td>Akron, OH Metro Area</td>
      <td>702221</td>
      <td>341200</td>
      <td>19415</td>
      <td>19402</td>
      <td>21511</td>
      <td>13214</td>
      <td>10979</td>
      <td>...</td>
      <td>310M300US10420</td>
      <td>Akron, OH Metro Area</td>
      <td>482631.0</td>
      <td>12016.0</td>
      <td>27417.0</td>
      <td>156793.0</td>
      <td>92660.0</td>
      <td>44121.0</td>
      <td>94427.0</td>
      <td>55197.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>310M300US10500</td>
      <td>10500</td>
      <td>Albany, GA Metro Area</td>
      <td>152506</td>
      <td>72073</td>
      <td>4555</td>
      <td>4994</td>
      <td>6549</td>
      <td>3387</td>
      <td>1996</td>
      <td>...</td>
      <td>310M300US10500</td>
      <td>Albany, GA Metro Area</td>
      <td>99582.0</td>
      <td>5000.0</td>
      <td>10095.0</td>
      <td>32148.0</td>
      <td>23758.0</td>
      <td>10058.0</td>
      <td>10708.0</td>
      <td>7815.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>310M300US10540</td>
      <td>10540</td>
      <td>Albany, OR Metro Area</td>
      <td>122849</td>
      <td>61175</td>
      <td>4247</td>
      <td>4910</td>
      <td>3466</td>
      <td>2399</td>
      <td>1363</td>
      <td>...</td>
      <td>310M300US10540</td>
      <td>Albany, OR Metro Area</td>
      <td>85318.0</td>
      <td>2054.0</td>
      <td>5032.0</td>
      <td>23747.0</td>
      <td>29634.0</td>
      <td>6358.0</td>
      <td>12365.0</td>
      <td>6128.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>





```python
years = range(2006, 2017)
for year in years:
    merged_all_clean[year].columns = merged_all_clean[year].columns.str.replace(str(year), '')
```




```python
 merged_all_clean[2016].head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Id2</th>
      <th>Geography</th>
      <th>total</th>
      <th>male_total</th>
      <th>male_under5</th>
      <th>male_5to9</th>
      <th>male_10-14</th>
      <th>male_15-17</th>
      <th>male_18-19</th>
      <th>...</th>
      <th>Id</th>
      <th>Geography</th>
      <th>Total25plus</th>
      <th>Less than 9th grade</th>
      <th>9th to 12th grade, no diploma</th>
      <th>High school graduate (includes equivalency)</th>
      <th>Some college, no degree</th>
      <th>Associate's degree</th>
      <th>Bachelor's degree</th>
      <th>Graduate or professional degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>310M300US10180</td>
      <td>10180</td>
      <td>Abilene, TX Metro Area</td>
      <td>170860</td>
      <td>87459</td>
      <td>5571</td>
      <td>6315</td>
      <td>5340</td>
      <td>3113</td>
      <td>3663</td>
      <td>...</td>
      <td>310M300US10180</td>
      <td>Abilene, TX Metro Area</td>
      <td>109703.0</td>
      <td>5479.0</td>
      <td>9154.0</td>
      <td>40224.0</td>
      <td>25198.0</td>
      <td>7304.0</td>
      <td>15103.0</td>
      <td>7241.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>310M300US10380</td>
      <td>10380</td>
      <td>Aguadilla-Isabela, PR Metro Area</td>
      <td>309764</td>
      <td>153695</td>
      <td>7976</td>
      <td>9909</td>
      <td>9401</td>
      <td>7580</td>
      <td>4089</td>
      <td>...</td>
      <td>310M300US10380</td>
      <td>Aguadilla-Isabela, PR Metro Area</td>
      <td>217353.0</td>
      <td>50408.0</td>
      <td>20760.0</td>
      <td>62332.0</td>
      <td>26186.0</td>
      <td>19000.0</td>
      <td>27709.0</td>
      <td>10958.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>310M300US10420</td>
      <td>10420</td>
      <td>Akron, OH Metro Area</td>
      <td>702221</td>
      <td>341200</td>
      <td>19415</td>
      <td>19402</td>
      <td>21511</td>
      <td>13214</td>
      <td>10979</td>
      <td>...</td>
      <td>310M300US10420</td>
      <td>Akron, OH Metro Area</td>
      <td>482631.0</td>
      <td>12016.0</td>
      <td>27417.0</td>
      <td>156793.0</td>
      <td>92660.0</td>
      <td>44121.0</td>
      <td>94427.0</td>
      <td>55197.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>310M300US10500</td>
      <td>10500</td>
      <td>Albany, GA Metro Area</td>
      <td>152506</td>
      <td>72073</td>
      <td>4555</td>
      <td>4994</td>
      <td>6549</td>
      <td>3387</td>
      <td>1996</td>
      <td>...</td>
      <td>310M300US10500</td>
      <td>Albany, GA Metro Area</td>
      <td>99582.0</td>
      <td>5000.0</td>
      <td>10095.0</td>
      <td>32148.0</td>
      <td>23758.0</td>
      <td>10058.0</td>
      <td>10708.0</td>
      <td>7815.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>310M300US10540</td>
      <td>10540</td>
      <td>Albany, OR Metro Area</td>
      <td>122849</td>
      <td>61175</td>
      <td>4247</td>
      <td>4910</td>
      <td>3466</td>
      <td>2399</td>
      <td>1363</td>
      <td>...</td>
      <td>310M300US10540</td>
      <td>Albany, OR Metro Area</td>
      <td>85318.0</td>
      <td>2054.0</td>
      <td>5032.0</td>
      <td>23747.0</td>
      <td>29634.0</td>
      <td>6358.0</td>
      <td>12365.0</td>
      <td>6128.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



### Cleaning Data - Using percentages instead of actual values



```python
for i in range(2006, 2017):
    print(i, end='\r')
    merged_all_clean[i]['age15to19'] = merged_all_clean[i]['male_15-17'] + merged_all_clean[i]['male_18-19'] + merged_all_clean[i]['female_15-17'] + merged_all_clean[i]['female_18-19']
    merged_all_clean[i]['age15to19'] = merged_all_clean[i]['age15to19']/merged_all_clean[i]['total']
    
    merged_all_clean[i]['age20to24'] = merged_all_clean[i]['male_20'] + merged_all_clean[i]['male_21'] +  merged_all_clean[i]['male_22to24'] + merged_all_clean[i]['female_20'] + merged_all_clean[i]['female_21'] +  merged_all_clean[i]['female_22to24']
    merged_all_clean[i]['age20to24'] = merged_all_clean[i]['age20to24']/merged_all_clean[i]['total']

    merged_all_clean[i]['age25to29'] = merged_all_clean[i]['male_25to29'] + merged_all_clean[i]['female_25to29']
    merged_all_clean[i]['age25to29'] = merged_all_clean[i]['age25to29']/merged_all_clean[i]['total']
    
    merged_all_clean[i]['age30to34'] = merged_all_clean[i]['male_30to34'] + merged_all_clean[i]['female_30to34']
    merged_all_clean[i]['age30to34'] = merged_all_clean[i]['age30to34']/merged_all_clean[i]['total']
    
    merged_all_clean[i]['age35to44'] = merged_all_clean[i]['male_35to39'] + merged_all_clean[i]['male_40to44'] + merged_all_clean[i]['female_35to39'] + merged_all_clean[i]['female_40to44']
    merged_all_clean[i]['age35to44'] = merged_all_clean[i]['age35to44']/merged_all_clean[i]['total']
                                                                                            
    merged_all_clean[i]['age45to59'] = merged_all_clean[i]['male_45to49'] + merged_all_clean[i]['male_50to54'] + merged_all_clean[i]['male_55to59'] + merged_all_clean[i]['female_45to49'] + merged_all_clean[i]['female_50to54'] + merged_all_clean[i]['female_55to59']
    merged_all_clean[i]['age45to59'] = merged_all_clean[i]['age45to59']/merged_all_clean[i]['total']
    
    merged_all_clean[i]['age60plus'] = merged_all_clean[i]['male_60to61'] + merged_all_clean[i]['male_62to64'] + merged_all_clean[i]['male_65to66'] + merged_all_clean[i]['male_67to69'] + merged_all_clean[i]['male_70to74'] + merged_all_clean[i]['male_75to79'] + merged_all_clean[i]['male_80to84'] + merged_all_clean[i]['male_85plus'] + merged_all_clean[i]['female_60to61'] + merged_all_clean[i]['female_62to64'] + merged_all_clean[i]['female_65to66'] + merged_all_clean[i]['female_67to69'] + merged_all_clean[i]['female_70to74'] + merged_all_clean[i]['female_75to79'] + merged_all_clean[i]['female_80to84'] + merged_all_clean[i]['female_85plus']
    merged_all_clean[i]['age60plus'] = merged_all_clean[i]['age60plus']/merged_all_clean[i]['total']
```


    2016




```python
for i in range(2006, 2017):
    print(i, end='\r')
    merged_all_clean[i]['total'] = convert_to_int(merged_all_clean[i]['total'])
    merged_all_clean[i]['Now married (except separated)'] = convert_to_int(merged_all_clean[i]['Now married (except separated)'])
    merged_all_clean[i]['Widowed'] = convert_to_int(merged_all_clean[i]['Widowed'])
    merged_all_clean[i]['Divorced'] = convert_to_int(merged_all_clean[i]['Divorced'])
    merged_all_clean[i]['Separated'] = convert_to_int(merged_all_clean[i]['Separated'])
    merged_all_clean[i]['Never married'] = convert_to_int(merged_all_clean[i]['Never married'])
    
    merged_all_clean[i]['Now married (except separated)'] = merged_all_clean[i]['Now married (except separated)']/merged_all_clean[i]['total']
    merged_all_clean[i]['Widowed'] = merged_all_clean[i]['Widowed']/merged_all_clean[i]['total']
    merged_all_clean[i]['Divorced'] = merged_all_clean[i]['Divorced']/merged_all_clean[i]['total']
    merged_all_clean[i]['Separated'] = merged_all_clean[i]['Separated']/merged_all_clean[i]['total']
    merged_all_clean[i]['Never married'] = merged_all_clean[i]['Never married']/merged_all_clean[i]['total']
```


    2016




```python
for i in range(2006, 2017):
    print(i, end='\r')
    merged_all_clean[i]['White'] = merged_all_clean[i]['White']/merged_all_clean[i]['total']
    merged_all_clean[i]['Black or African American'] = merged_all_clean[i]['Black or African American']/merged_all_clean[i]['total']
    merged_all_clean[i]['American Indian and Alaska Native'] = merged_all_clean[i]['American Indian and Alaska Native']/merged_all_clean[i]['total']
    merged_all_clean[i]['Asian'] = merged_all_clean[i]['Asian']/merged_all_clean[i]['total']
    merged_all_clean[i]['Native Hawaiian and Other Pacific Islander'] = merged_all_clean[i]['Native Hawaiian and Other Pacific Islander']/merged_all_clean[i]['total']
    merged_all_clean[i]['Other'] = merged_all_clean[i]['Other']/merged_all_clean[i]['total']
    merged_all_clean[i]['Two or More'] = merged_all_clean[i]['Two or More']/merged_all_clean[i]['total']
```


    2016




```python
for i in range(2006, 2017):
    print(i, end='\r')
    merged_all_clean[i]['Less than 9th grade'] = merged_all_clean[i]['Less than 9th grade']/merged_all_clean[i]['Total25plus']
    merged_all_clean[i][' 9th to 12th grade, no diploma'] = merged_all_clean[i][' 9th to 12th grade, no diploma']/merged_all_clean[i]['Total25plus']
    merged_all_clean[i]['High school graduate (includes equivalency)'] = merged_all_clean[i]['High school graduate (includes equivalency)']/merged_all_clean[i]['Total25plus']
    merged_all_clean[i]['Some college, no degree'] = merged_all_clean[i]['Some college, no degree']/merged_all_clean[i]['Total25plus']
    merged_all_clean[i]["Associate's degree"] = merged_all_clean[i]["Associate's degree"]/merged_all_clean[i]['Total25plus']
    merged_all_clean[i]["Bachelor's degree"] = merged_all_clean[i]["Bachelor's degree"]/merged_all_clean[i]['Total25plus']
    merged_all_clean[i]['Graduate or professional degree'] = merged_all_clean[i]['Graduate or professional degree']/merged_all_clean[i]['Total25plus']

```


    2016




```python
for i in range(2006, 2017):
    print(i, end='\r')
    merged_all_clean[i] = merged_all_clean[i].drop(['male_under5', 'Total', 'male_5to9','female_5to9', 'male_10-14', 'male_15-17', 'male_18-19', 'male_20', 'male_21','male_22to24', 'male_25to29',  'male_30to34', 'male_35to39', 'male_40to44', 'male_45to49', 'male_50to54', 'male_55to59','male_60to61', 'male_62to64','male_65to66', 'male_67to69', 'male_70to74', 'male_75to79', 'male_80to84', 'male_85plus','female_under5', 'female_10-14', 'female_15-17', 'female_18-19', 'female_20', 'female_21',  'female_22to24',  'female_25to29',  'female_30to34', 'female_35to39', 'female_40to44', 'female_45to49', 'female_50to54', 'female_55to59', 'female_60to61', 'female_62to64', 'female_65to66', 'female_67to69', 'female_70to74', 'female_75to79', 'female_80to84', 'female_85plus'], axis=1)

```


    2016




```python
for i in range(2006, 2017):
    print(i, end='\r')
    merged_all_clean[i]['Some College'] = merged_all_clean[i]['Some college, no degree'] + merged_all_clean[i]["Associate's degree"]
    merged_all_clean[i] = merged_all_clean[i].drop(['Some college, no degree', "Associate's degree", 'Total25plus'], axis = 1)
```


    2016




```python
for i in range(2006, 2017):
    print(i, end='\r')
    merged_all_clean[i]['Vacancy Rate'] = merged_all_clean[i]['Bldg_Vacant'] / merged_all_clean[i]['Bldg_Total']
    merged_all_clean[i]['MtoF'] = merged_all_clean[i]['male_total'] / merged_all_clean[i]['female_total']
```


    2016




```python
for i in range(2006, 2017):
    print(i, end='\r')
    merged_all_clean[i] = merged_all_clean[i].drop(['Bldg_Vacant', 'Bldg_Occupied', 'Bldg_Total'], axis = 1)
    merged_all_clean[i] = merged_all_clean[i].drop(['male_total', 'female_total'], axis = 1)
```


    2016




```python
 merged_all_clean[2016].head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Id2</th>
      <th>Geography</th>
      <th>total</th>
      <th>White</th>
      <th>Black or African American</th>
      <th>American Indian and Alaska Native</th>
      <th>Asian</th>
      <th>Native Hawaiian and Other Pacific Islander</th>
      <th>Other</th>
      <th>...</th>
      <th>age15to19</th>
      <th>age20to24</th>
      <th>age25to29</th>
      <th>age30to34</th>
      <th>age35to44</th>
      <th>age45to59</th>
      <th>age60plus</th>
      <th>Some College</th>
      <th>Vacancy Rate</th>
      <th>MtoF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>310M300US10180</td>
      <td>10180</td>
      <td>Abilene, TX Metro Area</td>
      <td>170860</td>
      <td>0.781371</td>
      <td>0.082559</td>
      <td>0.007433</td>
      <td>0.020701</td>
      <td>0.000047</td>
      <td>0.078637</td>
      <td>...</td>
      <td>0.073587</td>
      <td>0.085257</td>
      <td>0.081459</td>
      <td>0.072006</td>
      <td>0.110640</td>
      <td>0.166288</td>
      <td>0.211670</td>
      <td>0.296273</td>
      <td>0.168903</td>
      <td>1.048656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>310M300US10380</td>
      <td>10380</td>
      <td>Aguadilla-Isabela, PR Metro Area</td>
      <td>309764</td>
      <td>0.704982</td>
      <td>0.034152</td>
      <td>0.000588</td>
      <td>0.000000</td>
      <td>0.000165</td>
      <td>0.226595</td>
      <td>...</td>
      <td>0.069133</td>
      <td>0.068317</td>
      <td>0.064365</td>
      <td>0.052343</td>
      <td>0.135936</td>
      <td>0.194167</td>
      <td>0.254862</td>
      <td>0.207892</td>
      <td>0.241089</td>
      <td>0.984789</td>
    </tr>
    <tr>
      <th>2</th>
      <td>310M300US10420</td>
      <td>10420</td>
      <td>Akron, OH Metro Area</td>
      <td>702221</td>
      <td>0.819179</td>
      <td>0.120844</td>
      <td>0.000964</td>
      <td>0.028414</td>
      <td>0.000212</td>
      <td>0.002700</td>
      <td>...</td>
      <td>0.067853</td>
      <td>0.075730</td>
      <td>0.065968</td>
      <td>0.060149</td>
      <td>0.113325</td>
      <td>0.211643</td>
      <td>0.236208</td>
      <td>0.283407</td>
      <td>0.092347</td>
      <td>0.945097</td>
    </tr>
    <tr>
      <th>3</th>
      <td>310M300US10500</td>
      <td>10500</td>
      <td>Albany, GA Metro Area</td>
      <td>152506</td>
      <td>0.421472</td>
      <td>0.539284</td>
      <td>0.001692</td>
      <td>0.009882</td>
      <td>0.000734</td>
      <td>0.010065</td>
      <td>...</td>
      <td>0.069643</td>
      <td>0.073394</td>
      <td>0.071092</td>
      <td>0.062384</td>
      <td>0.122566</td>
      <td>0.183514</td>
      <td>0.213415</td>
      <td>0.339579</td>
      <td>0.143845</td>
      <td>0.896063</td>
    </tr>
    <tr>
      <th>4</th>
      <td>310M300US10540</td>
      <td>10540</td>
      <td>Albany, OR Metro Area</td>
      <td>122849</td>
      <td>0.901383</td>
      <td>0.004843</td>
      <td>0.010159</td>
      <td>0.010574</td>
      <td>0.000602</td>
      <td>0.036720</td>
      <td>...</td>
      <td>0.060171</td>
      <td>0.055678</td>
      <td>0.071307</td>
      <td>0.061238</td>
      <td>0.121588</td>
      <td>0.185154</td>
      <td>0.255208</td>
      <td>0.421857</td>
      <td>0.055655</td>
      <td>0.991909</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>





```python
for i in range (2006,2017):
    merged_all_clean[i] = merged_all_clean[i].drop(['Id', 'Geography', 'Unnamed: 0', 'Id', 'Geography'], axis= 1)
```




```python
merged_all_clean[2011].shape
```





    (374, 31)



### Renaming columns for easier readability



```python

```




```python
new_columns = ['msa', 'pop', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'm1', 'm2', 'm3', 'm4', 'm5', 'i1', 'i2', 'e1', 'e2', 'e3', 'e4', 'e5', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'e6', 'vr', 'mtof']
```




```python
len(new_columns)
```





    31





```python
for i in range(2006, 2017):
     merged_all_clean[i].columns = new_columns
```




```python
for i in range(2006, 2017):
     merged_all_clean[i]['year'] = [i for j in range(0, len(merged_all_clean[i].msa))]
```


### Export Dataset



```python
dfs = []
for i in range(2006, 2017):
    dfs.append(merged_all_clean[i])
df = pd.concat(dfs)
df.index = list(range(0, df.shape[0]))
df.to_csv("census_data_stacked.csv")
```




```python

for i in range(2006, 2017):
    merged_result[str(i)+'Some College'] = merged_result[str(i)+'Some college, no degree'] + merged_result[str(i)+"Associate's degree"]
    
    merged_result = merged_result.drop([str(i)+'Some college, no degree', str(i)+"Associate's degree"], axis = 1)
```




```python
merged_result.to_csv('data/merged_census.csv')
```

