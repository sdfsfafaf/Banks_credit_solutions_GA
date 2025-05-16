import pandas as pd
import numpy as np
import utils as EGAUtils
import constants as EConstants
import matplotlib.pyplot as plt

# 'M%', 'P%', 'A%', 'D', 'POP_SIZE', 'AAA%', 'AA%', 'A%', 'BBB%', 'BB%', 'ACCEPTED_CUSTOMERS','GENERATION_SIZE'
df_results = pd.read_csv('results.csv', names=EConstants.get_cols_result())

df_results['M%'] = (df_results['M%'] * 100).round(2)
df_results['P%'] = (df_results['P%'] * 100).round(2)
df_results['A%'] = (df_results['A%'] * 100).round(2)
df_results['AAA%'] = (df_results['AAA%'] * 100).round(2)
df_results['AA%'] = (df_results['AA%'] * 100).round(2)
df_results['A%'] = (df_results['A%'] * 100).round(2)
df_results['BBB%'] = (df_results['BBB%'] * 100).round(2)
df_results['BB%'] = (df_results['BB%'] * 100).round(2)


print(df_results[['M%', 'P%', 'A%', 'ACCEPTED_CUSTOMERS']])
print(df_results[['AAA%', 'AA%', 'A%', 'BBB%', 'BB%', 'ACCEPTED_CUSTOMERS']])

# df_results[['M%', 'P%', 'A%']].plot()
# plt.legend()
# plt.show()