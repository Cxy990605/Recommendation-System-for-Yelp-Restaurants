"""
Method Description:
Referring to the HW3, I decided to use the model-based method (which balances the computation time and rmse) for my competition project. Compared with HW3, I fine-tuned the hyper-parameters and included more features to meet the RMSE threshold (from 0.987 to 0.975). 

Preprocessing:
param_pre.py is a file for the fillna values (mean, std, kurtosis, skewness, max and min). Bacause of the MemoryError on Vocareum, I have computed the file, and saved the result as intermediate.csv in work folder.
competition.py is a file for score

Error Distribution:
>=0 and <1: 
>=1 and <2: 
>=2 and <3: 
>=3 and <4: 
>=4: 

Execution Time:
1000-1500s

RMSE:
0.975+
"""


import time
import sys
from xgboost import XGBRegressor
from pyspark import SparkContext, SparkConf
import json
import math
import pandas as pd
import numpy as np
from param_pre import preprocess



### RDD common functions
def comb(U, V):
    return U[0] + V[0], U[1] + V[1]
    
def lst_fc(x):
    return list(x)
    
def add_fc(a,b):
    return a + b


### Dataframes computation functions
def feature_engineer(feature, input_type): #()
    rst = {}
    if len(feature) <= 0:
        return user_general if input_type == "usr" else bu_general
    array_list = pd.Series([float(i[1]) for i in feature])
    rst[f"{input_type}_avg"] = array_list.mean()
    rst[f"{input_type}_std"] = array_list.std()
    rst[f"{input_type}_kurt"] = array_list.kurt()
    rst[f"{input_type}_skew"] = array_list.skew()
    rst[f"{input_type}_max"] = array_list.max()
    rst[f"{input_type}_min"] = array_list.min()
    return rst


#def column_generation(user, business_id): #(user_dic,business_dic,features dic)
#    vector_1 = u_dic.get(user, [])
#    vector_2 = b_dic.get(business_id, [])
#    vector_1 = [i for i in vector_1 if i[0] != business_id] 
#    vector_2 = [i for i in vector_2 if i[0] != user] # get this business's remaining customers
#    dic1 = feature_engineer(vector_1, "usr")
#    dic2 = feature_engineer(vector_2, "bns")
#    dic3 = dict(dic1, **dic2)
#    dic0 = {"user_id": user, "business_id": business_id}
#    return dict(dic0, **dic3)

def map_cols(map_user, map_id):
    if map_user in u_dic:
        vec1 = u_dic[map_user]
    else:
        vec1 = []
    vec1 = [i for i in vec1 if i[0] != map_id]
    dic1 = feature_engineer(vec1, "usr")

    if map_id in b_dic:
        vec2 = b_dic[map_id]
    else:
        vec2 = []
    vec2 = [j for j in vec2 if j[0] != map_user]
    dic2 = feature_engineer(vec1, "bns")
    combined = {**dic1, **dic2}
    output_dict = {"user_id": map_user, "business_id": map_id, **combined}
    return output_dict



def combine_statistics(user,business_id,score,user_list,business_list):
    try:
        u_raw_list = user_list.remove(score)
    except:
        print(score)
        print(user_list)
    user_stat_dic = get_statistics(u_raw_list, temp=user_general, label='usr')
    bu_raw_list = business_list.remove(score)
    bu_stat_dic = get_statistics(bu_raw_list, temp=bu_general, label='bus')
    initial = {'user_id':user,'business_id':business_id,'y':score}
    return dict(initial,**user_stat_dic,**bu_stat_dic)


def get_statistics(raw_list,temp=None,input_type="usr"):
    np_list = pd.Series(raw_list)
    stat_dict = {}
    if raw_list and len(raw_list) > 0:
        stat_dict[f'{input_type}_avg'] = np_list.mean()
        stat_dict[f'{input_type}_std'] = np_list.std()
        stat_dict[f'{input_type}_kurt'] = np_list.kurt()
        stat_dict[f'{input_type}_skew'] = np_list.skew()
        stat_dict[f'{input_type}_max'] = np_list.max()
        stat_dict[f'{input_type}_min'] = np_list.min()
        return stat_dict
    else:
        return temp

    
def merge_data(train_data, feature_list, users):
    for features in feature_list:
        temp_data = pd.DataFrame(features)
        train_data = pd.merge(train_data, temp_data, on="business_id", how="left")

    temp_data = pd.DataFrame(users)
    train_data = pd.merge(train_data, temp_data, on="user_id", how="left")

    return train_data

def rmse(pred,truth):
        numerator = sum(pow((pred - truth), 2))/len(pred)
        return math.sqrt(numerator)

if __name__ == "__main__":
    sc = SparkContext("local[*]", "competition")
    sc.setLogLevel("WARN")
    
    folder = sys.argv[1]
    input_val = sys.argv[2]
    output_file = sys.argv[3]
   
    train_file = folder + "/yelp_train.csv"
    review_file = folder + "/review_train.json"
    user_file = folder + "/user.json"
    bus_file = folder + "/business.json"
    tip_file = folder + "/tip.json"
    photo_file = folder + "/photo.json"
    checkin_file = folder + "/checkin.json"

    time_start = time.time()
    
    
    
    
    # Read the preprocessing data for further fill in NaN value
    dff = sc.textFile("intermediate.csv").filter(lambda x: x[0].isalpha()==False).map(lambda x: x.split(",")).map(lambda x:(x[0], x[1])).collect()
    #print(dff)
    user_general, bu_general = {}, {}
    user_general["usr_avg"] = float(dff[0][0])
    bu_general["bns_avg"] = float(dff[0][1])
    user_general["usr_std"] = float(dff[1][0])
    bu_general["bns_std"] = float(dff[1][1])
    user_general["usr_kurt"] = float(dff[2][0])
    bu_general["bns_kurt"] = float(dff[2][1])
    user_general["usr_skew"] = float(dff[3][0])
    bu_general["bns_skew"] = float(dff[3][1])
    user_general["usr_max"] = float(dff[4][0])
    bu_general["bns_max"] = float(dff[4][1])
    user_general["usr_min"] = float(dff[5][0])
    bu_general["bns_min"] = float(dff[5][1])

    
    # Read the training data and valuation data: train_file and input_val
    train = sc.textFile(train_file)    
    header = train.first()
    data = train.filter(lambda item: item != header).map(lambda x: x.split(","))

    def train2rdd(case):
        if case == 0:
            train = data.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(lst_fc)
        else:
            train = data.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(lst_fc)
        train_dict = {}
        for i,j in train.collect():
            train_dict[i] = j
        return train_dict
    
    u_dic = train2rdd(0)
    b_dic = train2rdd(1)
    
   
    ### Iterate files for feature selections for XGB
    business_rdd = sc.textFile(bus_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], (x["stars"], x["review_count"], x['latitude']))).collect()
    business = []
    for i in business_rdd:
        bus_tmp = {}
        bus_tmp["business_id"] = i[0]
        bus_tmp["bus_stars"] = i[1][0]
        bus_tmp["review_cnt"] = i[1][1]
        bus_tmp["latitude"] = i[1][2]
        business.append(bus_tmp)
    
    

    # business data --- attributes data
    business_raw = sc.textFile(bus_file).map(lambda x: json.loads(x)).collect()
    attribute_dic = dict()
    attribute_cnt = dict()
    business_len = len(business_raw)
    for row in business_raw:
        if row.get('attributes',None):
            sub_dic = row['attributes']
            for j in sub_dic.keys():
                attribute_dic.setdefault(j, set())
                attribute_dic[j].add(sub_dic[j])
                attribute_cnt.setdefault(j, 0)
                attribute_cnt[j] += 1 / business_len  

    fea_needs = [key for key,value in attribute_dic.items() if len(value) <= 10 and attribute_cnt[key] >= 0.2]
    temp_dic = {i: None for i in fea_needs}
    business_attribute = []
    for row in business_raw:
        bus_record = temp_dic.copy()
        bus_record["business_id"] = row["business_id"]
        if row.get('attributes',None):
            sub_dic = row['attributes']
            for j in sub_dic.keys():
                if j in fea_needs:
                    bus_record[j] = sub_dic[j]
            business_attribute.append(bus_record.copy())
    del business_raw

  
    # tips_file: "likes"
    tips_rdd = sc.textFile(tip_file).map(lambda line: json.loads(line))\
        .map(lambda x:(x["business_id"], (0, 1))).reduceByKey(comb).collect()
    tips = []
    for pairs in tips_rdd:
        tips_dict = {}
        tips_dict["business_id"] = pairs[0]
        tips_dict["likes"] = pairs[1][1]
        tips.append(tips_dict)
    
    # photo_file: "labels"
    photo_rdd = sc.textFile(photo_file).map(lambda x: json.loads(x))\
    .map(lambda x: (x['business_id'], [x["label"]])).reduceByKey(add_fc)\
    .map(lambda x: (x[0], len(x[1]))).collect()
    photos = []
    for i in photo_rdd:
        photo_dict = {}
        photo_dict["business_id"] = i[0]
        photo_dict["labels"] = i[1]
        photos.append(photo_dict)

    # checkin_file: "customers"
    checks_rdd = sc.textFile(checkin_file).map(lambda x: json.loads(x)).map(lambda x: (x["business_id"], len(list(x["time"].values())))).collect()
    checks = []
    for items in checks_rdd:
        checks_dict = {}
        checks_dict["business_id"] = items[0]
        checks_dict["customers"] = items[1]
        checks.append(checks_dict)
   
    # user_file: based on the hw3 which only includes thre variables ("useful", "funny", "cool"), I add eight more variables in competition to imporve the performance
    
    user_raw = sc.textFile(input_val)
    header = user_raw.first()
    test_2 = user_raw.filter(lambda line: line != header).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    set1 = set(test_2.map(lambda x: x[0]).collect())
    set2 = set(list(u_dic.keys()))
    user_index = set1.union(set2)

    def str2num(input_str):
        return 0 if input_str == "None" else len(input_str.split(","))
      
    users_lst = sc.textFile(user_file).map(lambda x: json.loads(x)).filter(lambda x: x["user_id"] in user_index).map(lambda row: (row['user_id'], float(row['useful']), float(row['funny']), float(row['cool']), row["elite"], row["friends"],float(row['fans']), float(row['average_stars']), float(row['compliment_list']), float(row['compliment_note']), float(row['review_count']))).collect()
    users = []
    for i in users_lst:
        users_d = {}
        users_d["user_id"] = i[0]
        users_d["useful"] = i[1]
        users_d["funny"] = i[2]
        users_d["cool"] = i[3]
        users_d["elite"] = str2num(i[4])
        users_d["friends"] = str2num(i[5])
        users_d["fans"] = i[6]
        users_d["average_stars"] = i[7]
        users_d["compliment_list"] = i[8]
        users_d["compliment_note"] = i[9]
        users_d["review_count"] = i[10]
        users.append(users_d)
    
       
    ### Merge all the dataframes
    user_agg_rdd = data.map(lambda x:(x[0],float(x[2]))).groupByKey()
    bu_agg_rdd = data.map(lambda x:(x[1],float(x[2]))).groupByKey()

    
    def comb_map(input_x, stage):
        if stage == 0:
            return (input_x[1][0][0],(input_x[0], input_x[1][0][1],input_x[1][1]))
        else:
            return (input_x[1][0][0], input_x[0], input_x[1][0][1], input_x[1][0][2],input_x[1][1])

        
    train_rdd = data.map(lambda x:(x[0],(x[1],float(x[2])))).join(user_agg_rdd)\
        .map(lambda x: comb_map(x, 0)).join(bu_agg_rdd)\
        .map(lambda x: comb_map(x, 1)).repartition(30)\
        .map(lambda x: combine_statistics(x[0],x[1],x[2],list(x[3]),list(x[4]))).collect()
    

    feature_list = [business,business_attribute,tips,photos,checks]
    train_data = merge_data(pd.DataFrame(train_rdd),feature_list,users)

    
    for col in pd.DataFrame(business_attribute).columns:
        if col != "business_id":
            temp_data = pd.get_dummies(train_data[col].fillna("unk"), drop_first=True)
            new_cols = [col + i for i in temp_data.columns]
            temp_data.columns = new_cols
            train_data.drop(col, axis=1, inplace=True)
            train_data = pd.concat([train_data, temp_data], axis=1)
        

    def fill_nan(input_dataframe):
        global bu_general
        attrs_lst = list(bu_general.keys())
        for i in attrs_lst:
            input_dataframe[i].fillna(bu_general[i], inplace = True)
        input_dataframe.fillna(0, inplace = True)
        return input_dataframe
    
  
    train_data = fill_nan(train_data)

    

    ### Filter selected features
    test = sc.textFile(input_val).map(lambda x: x.split(","))
    test_header = test.first()
    test_2 = test.filter(lambda x: x != test_header)
    test_df = test_2.repartition(10).map(lambda x: map_cols(x[0], x[1])).collect()
    #test_df = test_2.repartition(10).map(lambda x: column_generation(x[0], x[1])).collect()
    test_data = merge_data(pd.DataFrame(test_df),feature_list,users)

    valid_cols = train_data.columns
    for col in pd.DataFrame(business_attribute).columns:
        if col != "business_id":
            temp_data = pd.get_dummies(test_data[col].fillna("unk"))
            cols = [i for i in temp_data.columns if col + i in valid_cols]
            temp_data = temp_data[cols]
            temp_data.columns = [col + i for i in temp_data.columns]
            test_data.drop(col, axis=1, inplace=True)
            test_data = pd.concat([test_data, temp_data], axis=1)
    
    test_data = fill_nan(test_data)
    

        
    ### Apply model-based: XGBoost
    train_data["y"] = train_data["y"].astype("float")
    train_cols = train_data.columns.difference(["y", "business_id", "user_id"])
    train_x = train_data[train_cols]
    train_y = train_data["y"]
    
    param = {
        "max_depth": 5,
        "min_child_weight": 1,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state":42,
        "n_estimators":400,
            }
    
    model = XGBRegressor(**param)
    model.fit(train_x, train_y,verbose = 50)  
    xgb_pred = model.predict(test_data[train_cols]) 
    dff_pred = pd.DataFrame(xgb_pred)
    dff_test = test_data[["user_id", "business_id"]]
    preds = pd.concat([dff_test, dff_pred], axis=1)
    preds.columns = ["user_id", "business_id", "prediction"]
    
    ### Save as output csv
    preds.to_csv(output_file, index=False)

    ### Compute RMSE
    val_df = pd.read_csv(input_val)
    valid_df = pd.merge(val_df, preds, on = ['user_id','business_id'])
    print(rmse(np.array(valid_df["stars"]), np.array(valid_df["prediction"])))
   

    time_end = time.time()
    duration = round(time_end - time_start, 2)    
    print('Duration:', duration)                                                                                                                                                                                                                                  