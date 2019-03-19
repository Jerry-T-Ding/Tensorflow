#coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data

sns.pairplot(data.data, x_vars=['D/G','L/G'], y_vars='LnGDP',kind='reg')
plt.show()
