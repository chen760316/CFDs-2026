"""
Print the frequency of occurrence of the LHS attribute set in CFDs
The frequency of simultaneous occurrence of LHS and RHS attribute sets in printed CFDs
Print how often LHS appears in CFDs and the RHS attribute set does not
"""
from rich.console import Console
import pandas as pd
import numpy as np

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

initial_file = '../datasets/adult/adult.csv'
# initial_file = '../sampling/output.csv'
df = pd.read_csv(initial_file)

"""case 1 [relationship, capital-gain] => sex, (Husband, 7298 || Male)"""
lhs_count = df[df['relationship'] == 'Husband'].shape[0]
lhs_rhs_count = df[(df['relationship'] == 'Husband') & (df['sex'] == 'Male')].shape[0]
violated_rows = df[(df['relationship'] == 'Husband') & (df['sex'] != 'Male')]
#
# print(lhs_count)
# print(lhs_rhs_count)
# print(violated_rows)

"""case 2 [sex, marital-status, age] => relationship, (Male, Married-civ-spouse, 61 || Husband)"""
# lhs_count = df[(df['sex'] == 'Male') & (df['marital-status'] == 'Married-civ-spouse')].shape[0]
# lhs_rhs_count = df[(df['sex'] == 'Male') & (df['marital-status'] == 'Married-civ-spouse') & (df['relationship'] == 'Husband')].shape[0]
# violated_rows = df[(df['sex'] == 'Male') & (df['marital-status'] == 'Married-civ-spouse') & (df['relationship'] != 'Husband')]

"""case 3 [education-num, hours-per-week, relationship] => marital-status, (13, 40, Husband || Married-civ-spouse)"""
# lhs_count = df[(df['relationship'] == 'Husband') & (df['hours-per-week'] == 40)].shape[0]
# lhs_rhs_count = df[(df['relationship'] == 'Husband') & (df['hours-per-week'] == 40) & (df['marital-status'] == 'Married-civ-spouse')].shape[0]
# violated_rows = df[(df['relationship'] == 'Husband') & (df['hours-per-week'] == 40) & (df['marital-status'] != 'Married-civ-spouse')]

print(lhs_count)
print(lhs_rhs_count)
print(violated_rows)
print(lhs_rhs_count/lhs_count)