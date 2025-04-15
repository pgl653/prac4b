
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = sb.load_dataset('mpg')

# Print the dataset
print(df)

# Describe the 'horsepower' column
print(df['horsepower'].describe())

# Describe the 'model_year' column
print(df['model_year'].describe())

# Define bins for horsepower
bins = [0, 75, 150, 240]
df['horsepower_new'] = pd.cut(df['horsepower'], bins=bins, labels=['l', 'm', 'h'])

# Show the new horsepower categories
c = df['horsepower_new']
print(c)

# Define bins for model year
ybins = [69, 72, 74, 84]
df['model_year_new'] = pd.cut(df['model_year'], bins=ybins, labels=['11', '12', '13'])

# Show the new model year categories
newyear = df['model_year_new']
print(newyear)

# Create a crosstab for horsepower and model year categories
df_chi = pd.crosstab(df['horsepower_new'], df['model_year_new'])

# Print the crosstab
print(df_chi)

# Perform chi-square test
chi2, p, dof, expected = stats.chi2_contingency(df_chi)

# Print the result of the chi-square test
print(f'Chi2 Statistic: {chi2}')
print(f'p-value: {p}')
print(f'Degrees of Freedom: {dof}')
print(f'Expected frequencies: \n{expected}')
