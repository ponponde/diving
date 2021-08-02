import scipy.io
import pandas as pd

mat = scipy.io.loadmat('diving_overall_scores.mat')
print(mat)
# data = mat['difficulty_level']
data = mat['overall_scores']
data
data = data.transpose(1,0)
data

df = pd.DataFrame(data)
df.to_csv('diving_overall_scores.csv')
