# import pandas as pd
# import sys
# df1 = pd.read_json("/Users/xiangyuanchi/Downloads/TestProgram/business.json", lines = True)
# # print(df1["stars"].mean())
# # print(df1["stars"].std())
# # print(df1["stars"].kurt())
# # print(df1["stars"].skew())


# df2 = pd.read_json("/Users/xiangyuanchi/Downloads/TestProgram/user.json", lines = True)
# #print((sum(df2["review_count"] * df2["average_stars"]))/sum(df2["review_count"]))
# print(df2["average_stars"].mean())
# print(df2["average_stars"].std())
# print(df2["average_stars"].kurt())
# print(df2["average_stars"].skew())
###/usr/bin/python3 /Users/xiangyuanchi/Downloads/TestProgram/param_pre.py
###/usr/bin/python3 /Users/xiangyuanchi/Downloads/TestProgram/param_pre.py /Users/xiangyuanchi/Downloads/TestProgram/user.json /Users/xiangyuanchi/Downloads/TestProgram/business.json


import pandas as pd
import sys

d1 = sys.argv[1]
d2 = sys.argv[2]


class preprocess:
    def default_dict(self, input_data1, input_data2):  # default_dict(self, user_file, bus_file)
        df_tmp1 = pd.read_json(input_data1, lines = True)
        df_tmp2 = pd.read_json(input_data2, lines = True)
        dict_tmp1 = {}
        dict_tmp1["usr_avg"] = df_tmp1["average_stars"].mean()
        dict_tmp1["usr_std"] = df_tmp1["average_stars"].std()
        dict_tmp1["usr_kurt"] = df_tmp1["average_stars"].kurt()
        dict_tmp1["usr_skew"] = df_tmp1["average_stars"].skew()
        dict_tmp1["usr_max"] = 5
        dict_tmp1["usr_min"] = 1

        dict_tmp2 = {}
        dict_tmp2["bns_avg"] = df_tmp2["stars"].mean()
        dict_tmp2["bns_std"] = df_tmp2["stars"].std()
        dict_tmp2["bns_kurt"] = df_tmp2["stars"].kurt()
        dict_tmp2["bns_skew"] = df_tmp2["stars"].skew()
        dict_tmp2["bns_max"] = 5
        dict_tmp2["bns_min"] = 1

        return dict_tmp1, dict_tmp2


if __name__ == '__main__':
    pre_pro = preprocess()
    user_dict, bus_dict = pre_pro.default_dict(d1, d2)
    user_values = list(user_dict.values())
    bus_values = list(bus_dict.values())
    d = {"user": user_values, "bus": bus_values}
    ddf = pd.DataFrame(data=d)
    ddf.to_csv("intermediate.csv", index = False)
    
    #print(ddf)
    
    #user_general = {"usr_avg": 3.54154,"usr_std": 1.38811,"usr_kurt": -0.88873,"usr_skew": -0.61507
    #    ,"usr_max":5,"usr_min":1}
    #bu_general = {"bns_avg": 3.63155,"bns_std": 1.01678,"bns_kurt": -0.3405,"bns_skew": 0.51992
    #    ,"bns_max": 5, "bns_min": 1}