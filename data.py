import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('https://github.com/allenai/MacGyver/raw/refs/heads/main/data/MacGyver/problem_solution_pair.xlsx')
df_extra = pd.read_excel('https://github.com/allenai/MacGyver/raw/refs/heads/main/data/MacGyver/additional_human_solutions.xlsx')
df.to_csv('MacGyver.csv')
df_extra.info()