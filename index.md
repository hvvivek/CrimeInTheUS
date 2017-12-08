---
title: Crime in the USA - Context
---

<!-- This is the home page -->

<!-- ## Lets have fun -->

<!-- >here is a quote

Here is *emph* and **bold**.

Here is some inline math $\alpha = \frac{\beta}{\gamma}$ and, of-course, E rules:

$$ G_{\mu\nu} + \Lambda g_{\mu\nu}  = 8 \pi T_{\mu\nu} . $$ -->

![Image of Crime in USA](Images/Background.jpeg)
## Problem Statement and Motivation
With crime, we want to investigate how strongly correlated a context is to its crime rates. What are those strongest contributing factors that correlate to increased crime rates? 

## Introduction and Description of Data
We have identified a few basic criteria including the demographic range in age, gender, and race. Additionally, we also hypothesize that marital status, income, and educational attainment of the given population may have an impact on the way people behave and/or commit crimes. We’ve also looked into external sources to find the total number of firearm licenses issued within each state to find additional correlation between crime rates and the availability of firearms. 
We have scraped the FBI website for the total numbers of reported violent crimes in each Metropolitan Statistical Area (within the U.S. and Puerto Rico) across 2006 to 2016. 

## Question:
Is there a direct correlation between high population density and crime rates?
Is there a correlation between high crime rates and high density of a certain race, age group, or gender?
Would an area populated with higher marital rates have lower crime rates? Or would an area populated with widowed or divorced population have higher crime rates?
Would a higher educated population lead to higher or lower crime rates?

## Literature Review/Related Work
In an article we found around marriage and crime rates, the author mainly discusses the social/cultural effects of crime rates. 
Forrest, Walter. “Marriage Helps Reduce Crime.” The Conversation. Accessed December 7, 2017. http://theconversation.com/marriage-helps-reduce-crime-3576.

## Modeling Approach and Project Trajectory
Given the predictors are mostly (if not all) are continuous variables, we have decided to go with the linear regression models rather than the logistic regression. 
With a basic linear regression as our baseline model, we also tested using a polynomial model (with and without interaction terms).

## Results, Conclusions, and Future Work

Some of the complications we’ve faced include inconsistent MSA names across different years.  There were a significant amount of data that was missing from the census data that needed to be imputed. Some of the data sets found on the census bureau website were missing Puerto Rico. Added complications came from overlapping data (i.e. data sets including two or more characteristics that are also present 