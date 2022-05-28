import math

import pandas as pd
import seaborn as sns
data = pd.read_csv(r'./diabetes.csv')
my_data = data.head(20)
my_data.corr()
sns.heatmap(my_data.corr())
new_data = my_data.drop('Pregnancies', axis=1)
new_data1 = my_data.drop('SkinThickness', axis=1)
new_data2 = my_data.drop('BloodPressure', axis=1)
new_data3 = my_data.drop('BMI', axis=1)
new_data4 = my_data.drop('Age', axis=1)
new_data5 = my_data.drop('DiabetesPedigreeFunction', axis=1)
my_newdata = new_data5.drop('Outcome',axis=1)
user_data = [160, 35]
columns = list(my_newdata)
data_arr = []
dist_arr = []
for i in range(0, 19):
    for j in columns:
        print(my_newdata[j][i])
        elem = my_newdata[j][i]
        data_arr.append(elem)
    print(data_arr)
    dist = ((user_data[0] - data_arr[0]) **2 + (user_data[1] - data_arr[1]) **2) **0.5
    dist_arr.append(dist)
    data_arr = []
result = pd.DataFrame(dist_arr)
result['Glucose'] = my_newdata['Glucose']
result['Insulin'] = my_newdata['Insulin']
#result['Diabetes'] = my_newdata['Outcome']
k = math.sqrt(20)           #k = 20 ** 0.5
k = 5
result.nsmallest(5, [0])


