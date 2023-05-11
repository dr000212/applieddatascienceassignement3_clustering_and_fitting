# -*- coding: utf-8 -*-
"""
Created on Thu May 11 04:14:11 2023

@author: Deepak Raj
Student ID: 22010606
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

# Read the CSV file
def load_world_data_clim_change(file_path):
    """
     Load world climate change data from a CSV file.

    Parameters:
    -----------
    file_path : str
        The file path to the CSV file containing the data.

    Returns:
    --------
    pandas.DataFrame
        The loaded world climate change data.

    """
    worlddataclimchange = pd.read_csv(file_path)
    return worlddataclimchange

worlddataclimchange1 = load_world_data_clim_change('C:/Users/user/Downloads/adsassignment3.csv')


def clean_transpose(worlddataclimchange1):
    """
    Transpose a dataframe, remove any rows or columns that contain only NaN values,
    and replace any remaining NaN values with zero.

    Parameters:
    -----------
    worlddataclimchange : pandas.DataFrame
        The input dataframe to be cleaned and transposed.

    Returns:
    --------
    pandas.DataFrame
        The cleaned and transposed dataframe.
    """
    # Transpose the dataframe
    df1 = worlddataclimchange1.transpose()

    # Remove any rows or columns that contain only NaN values
    cleaned1 = df1.dropna(how='all').T.dropna(how='all')

    # Replace any remaining NaN values with zero
    cleaned2 = cleaned1.fillna(0)

    return cleaned2

# Clean and transpose the data
transposeddata = clean_transpose(worlddataclimchange1)
print(transposeddata)

# Get the years and indicator name of interest
datayears1 = [str(year) for year in range(2000, 2011)]
indicatorname = 'Agricultural land (sq. km)'

# Get the data for Aruba and the years of interest
data = transposeddata.loc[(transposeddata['Country Name'] == 'United States') & (transposeddata['Indicator Name'] == indicatorname), datayears1].values[0]

# Define the function to fit (quadratic)
def func(x, a, b, c):
    return a * x**2 + b * x + c

# Perform the curve fitting
popt, pcov = curve_fit(func, np.arange(len(datayears1)), data)

# Get the standard deviation of the residuals
residuals = data - func(np.arange(len(datayears1)), *popt)
perr = np.sqrt(np.diag(pcov))

# Create the line plot with the fitted curve and confidence interval
plt.plot(datayears1, data, label='Agricultural land (sq. km)')
plt.plot(datayears1, func(np.arange(len(datayears1)), *popt), label='Curve Fit')

plt.fill_between(datayears1, func(np.arange(len(datayears1)), *(popt + perr)),
                 func(np.arange(len(datayears1)), *(popt - perr)), alpha=0.2)

plt.title('Agricultural land (sq. km) in United States')
plt.xlabel('Year')
plt.ylabel('Agricultural land (sq. km)')
plt.legend()
plt.savefig('lineplot1.png', dpi=300)
plt.show()

datayears2 = [str(year) for year in range(2000, 2001)]

# Define the two indicators of interest
indicator1name = 'Population, total'
indicator2name = 'Urban population'

# Get the list of all countries in the data
countriesdata = transposeddata['Country Name'].unique()

# Initialize an empty array to store the concatenated data
concat_data1 = np.empty([0, 2])

# Loop over the countries and concatenate the data for the two indicators and the years of interest
for country in countriesdata:
    data1 = transposeddata.loc[(transposeddata['Country Name'] == country) & (transposeddata['Indicator Name'] == indicator1name), datayears2].values[0]
    data2 = transposeddata.loc[(transposeddata['Country Name'] == country) & (transposeddata['Indicator Name'] == indicator2name), datayears2].values[0]
    data_country1 = np.column_stack((data1, data2))
    concat_data1 = np.vstack([concat_data1, data_country1])

# Perform KMeans clustering with k=4
kmeanscluster = KMeans(n_clusters= 4, random_state=0).fit(concat_data1)

# Get the cluster labels
labels = kmeanscluster.labels_

# Set the plot title and axis labels
plt.title('Total Population vs. Urban Population for all Countries (2000)')
plt.xlabel('Population, total')
plt.ylabel('Urban population')

# Create a scatter plot for each cluster with a different color and label
for i in range(kmeanscluster.n_clusters):
    plt.scatter(concat_data1[labels == i, 0], concat_data1[labels == i, 1], label='Cluster {}'.format(i+1))

# Add a legend to the plot with the cluster labels
plt.legend()
plt.savefig('scatterplotads.png', dpi=300)
# Display the plot
plt.show()

