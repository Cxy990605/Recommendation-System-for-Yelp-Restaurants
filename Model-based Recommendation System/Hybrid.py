from pyspark import SparkContext
import os
import sys
import time
import json
import numpy as np
from xgboost import XGBRegressor


folder = sys.argv[1]
input_val = sys.argv[2]
output_path = sys.argv[3]

##### item-based
def rmse(b,b_new,inter):
    l1, l2 = [], []
    for u in inter:
        l1.append(float(bus_user_r_dict[b][u]))
        l2.append(float(bus_user_r_dict[b_new][u]))
        avg1, avg2 = sum(l1) / len(l1), sum(l2) / len(l2)
        tmp1 = [i - avg1 for i in l1]
        tmp2 = [i - avg2 for i in l2]
        zip_lst = zip(tmp1,tmp2)
        x_lst = []
        for i, j in zip_lst:
            x_lst.append(i * j)
        x_val = sum(x_lst)
        y_val = ((sum([x ** 2 for x in tmp1])**(1/2)) * (sum([x ** 2 for x in tmp2])**(1/2)))
        if y_val != 0:
            w = x_val/y_val
        else:
            w = 0
    return w

def item_cf(business,user):
    if user not in user_bus_dict:
        return 3.75
    if business not in bus_user_dict:
        return avg_u_dict[user]
    w_lst = []
    for b1 in user_bus_dict[user]:
        tmp = sorted((b1,business))
        tmp = tuple(tmp)
        if tmp in w_dict:
            w = w_dict[tmp]
        else:
            #user_corr = bus_user_dict[business].intersection(bus_user_dict[b1])
            user_corr = bus_user_dict[business] & bus_user_dict[b1]
            if len(user_corr) == 2:
                user_corr = list(user_corr)
                weight1, weight2 = float((5 - abs(float(bus_user_r_dict[business][user_corr[0]]) - float(bus_user_r_dict[b1][user_corr[0]])))/5), float((5 - abs(float(bus_user_r_dict[business][user_corr[1]]) - float(bus_user_r_dict[b1][user_corr[1]])))/5)
                w = (weight1+weight2) / 2
            elif len(user_corr) <= 1:
                w = float((5 - abs(avg_b_dict[business] - avg_b_dict[b1])) / 5)
            else: 
                w = rmse(business,b1,user_corr)

            w_dict[tmp] = w
        w_lst.append((w, float(bus_user_r_dict[b1][user])))
    cand = sorted(w_lst, key = lambda x: -x[0])[:15]
    x_val, y_val = 0, 0
    for weight, rate in cand:
        x_val += weight * rate
        y_val += abs(weight)
    if y_val != 0:
        return x_val/y_val
    else:
        return 3.75 

sc = SparkContext('local[*]', 'task2_3').getOrCreate()
sc.setLogLevel("ERROR")

train_file = folder + "/yelp_train.csv"
review_file = folder + "/review_train.json"
user_file = folder + "/user.json"
bus_file = folder + "/business.json"

rdd = sc.textFile(train_file).filter(lambda x: x != "user_id,business_id,stars").map(lambda x: (x.split(",")[1], x.split(",")[0], x.split(",")[2]))
# bus[0], user[1], star[2]
#print(rdd.take(5))
def f_set(x):
    return set(x)
def f_list(x):
    return list(x)

def train_rdd(case):
    if case == 0:
        train = rdd.map(lambda row: (row[0], row[1])).groupByKey().mapValues(f_set)
    else:
        train = rdd.map(lambda row: (row[1], row[0])).groupByKey().mapValues(f_set)
    train_dict = {}
    for i, j in train.collect():
        train_dict[i] = j
    return train_dict

bus_user_dict = train_rdd(0)
user_bus_dict = train_rdd(1)
#print(bus_user_dict)


def average(input_index): # 0 for business and 1 for user
    avg = rdd.map(lambda x: (x[input_index], float(x[2]))).groupByKey().mapValues(f_list)
    avg = avg.map(lambda x: (x[0], sum(x[1]) / len(x[1])))
    avg_dict = {}
    for i,j in avg.collect():
        avg_dict[i] = j
    return avg_dict

avg_b_dict = average(0)
avg_u_dict = average(1)
#print(avg_b_dict)

bus_user_r = rdd.map(lambda row: (row[0], (row[1], row[2]))).groupByKey().mapValues(f_set)
bus_user_r_dict = {}
for i, j in bus_user_r.collect():
    tmp = {}
    for k in j:
        tmp[k[0]] = k[1]
    bus_user_r_dict[i] = tmp
#print(bus_user_r_dict) 

rdd_val = sc.textFile(input_val).filter(lambda x: x!= "user_id,business_id,stars").map(lambda x: (x.split(",")[1], x.split(",")[0]))
w_dict = {}
item_pred = []
for i in rdd_val.collect():
    pred = item_cf(i[0], i[1])
    item_pred.append(pred)


##### model-based

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

xgb = XGBRegressor(verbosity=0, n_estimators=50, random_state=42, max_depth=10)
xgb.fit(x_train, y_train)
pred = xgb.predict(x_val)
#print(pred)


time_start = time.time()
#sc = SparkContext('local[*]', 'task2_3').getOrCreate()
#sc.setLogLevel("ERROR")
train_set = sc.textFile(train_file).filter(lambda x: x != "user_id,business_id,stars").map(lambda x: (x.split(",")[1], x.split(",")[0], x.split(",")[2]))

bus_user_dict = train_rdd(0)
user_bus_dict = train_rdd(1)
avg_b_dict = average(0)
avg_u_dict = average(1)


bus_user_dict_new = bus_user_dict
user_bus_dict_new = user_bus_dict
avg_b_dict_new = avg_b_dict
avg_u_dict_new = avg_u_dict

bus_user_r_new = bus_user_r
bus_user_r_dict_new = bus_user_r_dict

val_set = rdd_val
w_dict_new = {}
### Apply item-based 
item_res = []
for i in val_set.collect():
    pred1 = item_cf(i[0], i[1])
    item_res.append(pred1)

### Apply model-based
user_bus_list = y_test
model_res = pred

### Hybrid method: final score = alpha * score_item + (1-alpha) * score_model
alpha = 0.1
string = "user_id, business_id, prediction\n"
for i in range(len(model_res)):
    final_score = float(alpha) * float(item_res[i]) + (1 - float(alpha)) * float(model_res[i])
    string += user_bus_list[i][0] + "," + user_bus_list[i][1] + "," + str(final_score) + "\n"

with open(output_path, "w") as file:
    file.writelines(string)

time_end = time.time()
duration = round(time_end - time_start, 2)
print('Duration:', duration)                                                                                                                  