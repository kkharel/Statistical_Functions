import math
import os
import random
import re
import sys
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext

# Mean

def Mean(Array):
  
  return sum(Array)/len(Array)


# Weighted Mean

def WeightedMean(Array, Weights):
    
    sum_product = 0
    for i,j in zip(Array,Weights):
        
        sum_product += i*j
        
    weights_sum = sum(Weights)
    
    weighted_mean = sum_product / weights_sum
    
    return round(weighted_mean, 2)
  
  # Median

def Median(Array):
  
  Array.sort()
  
  n = len(Array)
  
  if n % 2 == 0:
    
    value = (Array[n//2-1] + Array[n//2])/2
    
  else:
    
    value = Array[n//2]
    
  return round(value,2)

# Mode

def mode(dataset):
  
    if len(dataset) == 0:
      
        raise ValueError("Input list must not be empty.")
      
    freq_dict = {}
    
    for value in dataset:
      
        freq_dict[value] = freq_dict.get(value, 0) + 1

    max_frequency = max(freq_dict.values())

    modes = [x for x, frequency in freq_dict.items() if frequency == max_frequency]

    if len(modes) == len(freq_dict):
      
        return None

    return modes
  

# Variance

def Variance(Array):
  
  n = len(Array)
  
  mean = sum(Array)/n
  
  squared_difference = []
  
  for i in Array:
    
    difference = i - mean
    
    squared_difference.append(difference**2)
    
  if n < 2:
    
    raise ValueError("Sample size should be at least 2 to calculate variance.")
    
  variance = sum(squared_difference) / n 
  
  return round(variance,2)
  
  
# StandardDeviation
  
def StandardDeviation(Array):
  
  n = len(Array)
  
  mean = sum(Array)/n
  
  squared_difference = []
  
  for i in Array:
    
    difference = i - mean
    
    squared_difference.append(difference**2)
    
  if n < 2:
    
    raise ValueError("Sample size should be at least 2 to calculate variance.")
    
  variance = sum(squared_difference) / n 
  
  stdev = math.sqrt(variance)
  
  return round(stdev,2)
  
  
def skewness(dataset):
  
  n = len(dataset)
    
  if n < 3:
    
    raise ValueError("Sample size should be at least 4 to calculate kurtosis.")
    
  mean = Mean(dataset)
    
  std_dev = StandardDeviation(dataset)
    
  third_moment = sum((x - mean) ** 3 for x in dataset)
    
  skewness = (third_moment / (n * std_dev ** 3)) 
    
  return skewness


# Kurtosis

def kurtosis(dataset):
  
  n = len(dataset)
    
  if n < 4:
    
    raise ValueError("Sample size should be at least 4 to calculate kurtosis.")
    
  mean = Mean(dataset)
    
  std_dev = StandardDeviation(dataset)
    
  fourth_moment = sum((x - mean) ** 4 for x in dataset)
    
  kurtosis = (fourth_moment / (n * std_dev ** 4)) - 3
    
  return kurtosis


# Quartiles

def Quartiles(Array):
  
  Array.sort()
  
  n = len(Array)
  
  if n % 2 == 0:
    
    lower_half = Array[:n//2]
    
    upper_half = Array[n//2:]
    
    q1 = Median(lower_half)
    
    q3 = Median(upper_half)
    
  else:
    
    lower_half = Array[:n//2]
    
    upper_half = Array[n//2 + 1:]
    
    q1 = Median(lower_half)
    
    q3 = Median(upper_half)
    
  q2 = Median(Array)
  
  return {'Q1':q1, 'Q2':q2, 'Q3':q3}

# Inter-Quartile

def interQuartile(values, freqs):
  
  S = []
  
  for i in range(len(values)):
    
    new_list = [values[i]] * freqs[i]
        
    S.extend(new_list)
    
  S.sort()
    
  n = len(S)
    
  if n % 2 == 0:
    
    lower_half = S[:n // 2]
    
    upper_half = S[n // 2:]
    
    q1 = Median(lower_half)
    
    q3 = Median(upper_half)
  
  else:
    
    lower_half = S[:n // 2]
    
    upper_half = S[n // 2 + 1:]
    
    q1 = Median(lower_half)
    
    q3 = Median(upper_half)
    
  interquartile_range = q3 - q1
  
  return round(interquartile_range, 2)

  
 
# Distributions

 # Binomial Distribution   

# Factorial

def factorial(n):
  
  if n < 0:
    
    raise ValueError("Factorial not available for negative numbers")
  
  elif n == 0 or n == 1:
    
    return 1
  
  else:
    
    result = 1
    
    for i in range(2, n+1):
      
      result *= i
      
    return result
      
      
def n_choose_k(n,k):
  
  if k > n:
    
    raise ValueError("Cannot have more trials than total population")
  
  else:
    
    return factorial(n) / (factorial(k) * factorial(n-k))
  
  
  
def binomial_probability(n, p, k):
  
  return n_choose_k(n, k) * p ** k * (1 - p) ** (n - k)


def geometric_probability(p, k):
  
  probability = 0
    
  for i in range(1, k+1):
    
    probability += ((1 - p) ** (i - 1)) * p
    
  return probability
  

def poisson_probability(lmbda, k):
  
  return (math.e**(-lmbda) * lmbda ** k) / factorial(k)


def normal_probability(X, mu, sigma):
  
  return 0.5*(1+math.erf((X - mu)/(sigma * math.sqrt(2))))



def CI(n,z, mu, sigma):
  
  _mu = n*mu
  
  _sigma = math.sqrt(n)*sigma
  
  A = (_mu - _sigma * z)/100
  
  B = (_mu + _sigma * z)/100

  return A, B


def covariance(X, Y):
  
  mean_X = Mean(X)
    
  mean_Y = Mean(Y)
    
  covariance_sum = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(len(X)))
    
  cov = covariance_sum / len(X)
    
  return cov



def pearson_correlation(X, Y):
  
  cov_XY = covariance(X, Y)
    
  st_dev_X = StandardDeviation(X)
    
  st_dev_Y = StandardDeviation(Y)
    
  coefficient = cov_XY / (st_dev_X * st_dev_Y)
    
  return coefficient


  
def spearman_rank_correlation(X, Y, n):
  
  def rank(arr):
    
    sort = sorted(arr)
    
    rank = []
    
    for i in arr:
    
      rank.append(sort.index(i) + 1)
    
    return rank
    
  def correlation(X, Y, n):
    
    std_x = StandardDeviation(X)
        
    std_y = StandardDeviation(Y)
        
    m_x = Mean(X)
    
    m_y = Mean(Y)
    
    corr = 0
    
    for i in range(n):
      
      corr += (X[i] - m_x) * (Y[i] - m_y) / n
    
    p_coeff = corr / (std_x * std_y)
    
    return round(p_coeff, 3)
    
  rank_x = rank(X)
  rank_y = rank(Y)
    
  return correlation(rank_x, rank_y, n)



def correlation(X, Y):
  
  std_x = StandardDeviation(X)
    
  std_y = StandardDeviation(Y)
    
  m_x = Mean(X)
    
  m_y = Mean(Y)
    
  corr = 0
    
  for i in range(len(X)):
    
    corr += ((X[i] - m_x) * (Y[i] - m_y)) / len(X)
        
  return corr / (std_x * std_y)
    

# Linear Regression

def linear_regression(X, Y, _x):
  
  b = (StandardDeviation(Y) / StandardDeviation(X))*correlation(X,Y)
    
  a = Mean(Y) - (b * Mean(X))
    
  pred = a + b * _x
    
  return a, b, pred

