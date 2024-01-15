from pyspark import SparkContext
import os
import json
import sys
import time
import numpy as np
from xgboost import XGBRegressor

### command line:   /usr/bin/python3 /Users/xiangyuanchi/Downloads/TestProgram/task1.py


folder = sys.argv[1] # /Users/xiangyuanchi/Downloads/TestProgram
input_val = sys.argv[2]
output_path = sys.argv[3]

time_start = time.time()

sc = SparkContext('local[*]', 'task2_2').getOrCreate()
sc.setLogLevel("ERROR")
train_file = folder + "/yelp_train.csv"
review_file = folder + "/review_train.json"
user_file = folder + "/user.json"
bus_file = folder + "/business.json"

### yelp_train.csv & yelp_val.csv
rdd = sc.textFile(train_file).filter(lambda x: x!= "user_id,business_id,stars").map(lambda x: x.split(","))
val_rdd = sc.textFile(input_val).filter(lambda x: x!= "user_id,business_id,stars").map(lambda x: x.split(","))


### json.loads()
def rdd_dict(input_rdd):
    out = {}
    for i,j in input_rdd.collect():
        out[i] = j
    return out

def f_list(x):
    return list(x)

user_rdd = sc.textFile(user_file).map(lambda x: json.loads(x)).map(lambda row: (row['user_id'], (float(row['average_stars']), float(row['review_count']), float(row['fans']))))
user_dic = rdd_dict(user_rdd)

bus_rdd = sc.textFile(bus_file).map(lambda x: json.loads(x)).map(lambda row:(row['business_id'], (float(row['stars']), float(row['review_count']))))
bus_dic = rdd_dict(bus_rdd)

# parameters for xgboost model
review_rdd = sc.textFile(review_file).map(lambda x:json.loads(x)).map(lambda row: (row['business_id'], (float(row['useful']), float(row['funny']), float(row['cool'])))).groupByKey().mapValues(f_list) 
review = rdd_dict(review_rdd)
review_dict = {}
for k, v in review.items():
    p1, p2, p3 = 0, 0, 0
    length = len(v)
    for val in v:
        tmp = list(val)
        p1 += tmp[0]
        p2 += tmp[1]
        p3 += tmp[2]
    review_dict[k] =  (p1/length, p2/length, p3/length)
#print(review_dict)



# Train model
def parameters(user,bus):
    if bus in review_dict:
        p1 = review_dict[bus][0]
        p2 = review_dict[bus][1]
        p3 = review_dict[bus][2]
    else:
        p1, p2, p3 = None, None, None
    if user in user_dic:
        user_rate = user_dic[user][0]
        user_review = user_dic[user][1]
        user_fans = user_dic[user][2]
    else:
        user_rate, user_review, user_fans = None, None, None
    if bus in bus_dic:
        bus_rate = bus_dic[bus][0]
        bus_review = bus_dic[bus][1]
    else:
        bus_rate, bus_review = None, None
    lst = [p1,p2,p3,user_rate,user_review,user_fans,bus_rate,bus_review]
    return lst


def xgb(input_model): # 1: rdd   2: val_rdd
    if input_model == rdd:
        X, Y = [], []
        for item in input_model.collect():
            Y.append(item[2]) 
            X.append(parameters(item[0], item[1]))
        return (X,Y)
    else:
        user_business_lst = []
        X = []
        for item in val_rdd.collect():
            u, b = item[0], item[1] 
            user_business_lst.append((u,b))
            X.append(parameters(u,b))
        return (X, user_business_lst)

x_train = xgb(rdd)[0]
x_train = np.array(x_train, dtype = "float32")
y_train = xgb(rdd)[1]
y_train = np.array(y_train, dtype = "float32")

x_val = xgb(val_rdd)[0]
x_val = np.array(x_val, dtype = "float32")
y_test = xgb(val_rdd)[1]


#param = {
        #'lambda': 9.92724463758443, 
        #'alpha': 0.2765119705933928, 
        #'colsample_bytree': 0.5, 
        #'subsample': 0.8, 
        #'learning_rate': 0.02, 
        #'max_depth': 17, 
        #'random_state': 2020, 
        #'min_child_weight': 101,
        #'n_estimators': 300,}

#xgb = XGBRegressor(**param)
xgb = XGBRegressor(verbosity=0, n_estimators=50, random_state=42, max_depth=10)
xgb.fit(x_train, y_train)
pred = xgb.predict(x_val)
#print(pred)

string = "user_id, business_id, prediction\n"
for idx in range(len(pred)):
    tmp_str = y_test[idx][0] + "," + y_test[idx][1] + "," + str(pred[idx]) + "\n"
    string += tmp_str

with open(output_path, "w") as file:
    file.writelines(string)

time_end = time.time()
duration = round(time_end - time_start, 2)
print('Duration:', duration)                                                                                                                  