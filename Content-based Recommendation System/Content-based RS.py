from pyspark import SparkContext
import os
import sys
import time
#from statistics import mean

input_train = sys.argv[1]
input_val = sys.argv[2]
output_file = sys.argv[3]

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


#time_start = time.time()
sc = SparkContext('local[*]', 'task2_1').getOrCreate()
sc.setLogLevel("ERROR")
rdd = sc.textFile(input_train).filter(lambda x: x != "user_id,business_id,stars").map(lambda x: (x.split(",")[1], x.split(",")[0], x.split(",")[2]))
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

                

time_start = time.time()
rdd_val = sc.textFile(input_val).filter(lambda x: x!= "user_id,business_id,stars").map(lambda x: (x.split(",")[1], x.split(",")[0]))
w_dict = {}
string = "user_id, business_id, prediction\n"
for i in rdd_val.collect():
    pred = item_cf(i[0], i[1])
    line = i[1] + "," + i[0] + "," + str(pred) + "\n"
    string += line

with open(output_file, "w") as file:
    file.writelines(string)

time_end = time.time()
duration = round(time_end - time_start, 2)
print('Duration:', duration)

