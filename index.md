---
title: Crime in the USA - Context
---

<!-- This is the home page -->

<!-- ## Lets have fun -->

<!-- >here is a quote

Here is *emph* and **bold**.

Here is some inline math $\alpha = \frac{\beta}{\gamma}$ and, of-course, E rules:

$$ G_{\mu\nu} + \Lambda g_{\mu\nu}  = 8 \pi T_{\mu\nu} . $$ -->

![Image of Crime in USA](Images/Background.jpg)
## Problem Statement and Motivation
**Data analysis in crime is not news.** Agencies around the world have used data in various ways to understand, get insights and eventually predict crime. <br>

Yet we have also come to acknowledge issues with these new techniques. The datasets have mirrored human biases and we continue to see correlations which make very little sense. <br>

Our motivation is, therefore, to better understand the context of crime in the US. We would like to combine data from multiple sources in a hope to prove/disprove possible correlations. We hope that this work can help start conversations about these issues and understand whether the context of a geography is correlated to crime within its boundaries. <br>

## Introduction and Description of Data
Our data comes from the following main sources:<br>
1. FBI Uniform Crime Reporting Database<br>
2. American Fact Finder - US Census Data<br>
3. Bureau of Alcohol, Tobacoo, Firearms and Exposives - US Firearms Data<br>
We have identified a few basic criteria including the demographic range in age, gender, and race. Additionally, we also hypothesize that marital status, income, and educational attainment of the given population may have an impact on the way people behave and/or commit crimes. We’ve also looked into external sources to find the total number of firearm licenses issued within each state to find additional correlation between crime rates and the availability of firearms. 
We have scraped the FBI website for the total numbers of reported violent crimes in each Metropolitan Statistical Area (within the U.S. and Puerto Rico) across 2006 to 2016. 

## Questions:
Our questions derive from our Literature Review (below):
1. Is there a direct correlation between high population density and crime rates?<br>
2. Is there a correlation between high crime rates and high density of a certain race, age group, or gender?<br>
3. Would an area populated with higher marital rates have lower crime rates? Or would an area populated with widowed or divorced population have higher crime rates?<br>
4. Would a higher educated population lead to higher or lower crime rates?<br>

## Literature Review/Related Work
**Marriage and Crime**
In an article we found around marriage and crime rates, the author mainly discusses the social/cultural effects of crime rates. 
Forrest, Walter. “Marriage Helps Reduce Crime.” The Conversation. Accessed December 7, 2017. [Marriage and Crime](http://theconversation.com/marriage-helps-reduce-crime-3576)

**Real Estate and Crime**
In an article we found around marriage and crime rates, the author mainly discusses the social/cultural effects of crime rates. 
Forrest, Walter. “Marriage Helps Reduce Crime.” The Conversation. Accessed December 7, 2017. [Real Estate and Crime](http://theconversation.com/marriage-helps-reduce-crime-3576)

**Firearms and Crime**
In an article we found around marriage and crime rates, the author mainly discusses the social/cultural effects of crime rates. 
Forrest, Walter. “Marriage Helps Reduce Crime.” The Conversation. Accessed December 7, 2017. [Firearms and Crime](http://theconversation.com/marriage-helps-reduce-crime-3576)

**Income, Education and Crime**
In an article we found around marriage and crime rates, the author mainly discusses the social/cultural effects of crime rates. 
Forrest, Walter. “Marriage Helps Reduce Crime.” The Conversation. Accessed December 7, 2017. [Income, Education and Crime](http://theconversation.com/marriage-helps-reduce-crime-3576)

## Modeling Approach and Project Trajectory
Given the predictors are mostly (if not all) are continuous variables, we have decided to go with the linear regression models rather than the logistic regression. 
With a basic linear regression as our baseline model, we also tested using a polynomial model (with and without interaction terms).

## Results, Conclusions, and Future Work

Some of the complications we’ve faced include inconsistent MSA names across different years.  There were a significant amount of data that was missing from the census data that needed to be imputed. Some of the data sets found on the census bureau website were missing Puerto Rico. Added complications came from overlapping data (i.e. data sets including two or more characteristics that are also present 