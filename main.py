# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import pickle
import lightgbm as lgb
from math import radians, cos, sin, asin, sqrt

def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

start_all = datetime.datetime.now()


path_train = "train.csv"  # 训练文件路径
path_test = "test.csv"  # 测试文件路径
# path
# path_train = "/data/dm/train.csv"  # 训练文件路径
# path_test = "/data/dm/test.csv"  # 测试文件路径

# read train data
data = pd.read_csv(path_train)
train1 = []
alluser = data['TERMINALNO'].nunique()
# print(alluser)
# Feature Engineer, 对每一个用户生成特征:
# trip特征, record特征(数量,state等),
# 地理位置特征(location,海拔,经纬度等), 时间特征(星期,小时等), 驾驶行为特征(速度统计特征等)

for item in data['TERMINALNO'].unique():

    print('user NO:',item)
    temp = data.loc[data['TERMINALNO'] == item,:]
    temp.index = range(len(temp))
    # print(temp)

    # trip 特征
    num_of_trips = temp['TRIP_ID'].nunique()
    # print("trip:",num_of_trips)

    # record 特征
    num_of_records = temp.shape[0]
    # print("records",num_of_records)
    num_of_state = temp[['TERMINALNO','CALLSTATE']]
    # print(num_of_state)
    nsh = num_of_state.shape[0]
    # print("nsh",nsh)
    num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE']==0].shape[0]/float(nsh)
    # print("state0",num_of_state_0)
    num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE']==1].shape[0]/float(nsh)
    num_of_state_2 = num_of_state.loc[num_of_state['CALLSTATE']==2].shape[0]/float(nsh)
    num_of_state_3 = num_of_state.loc[num_of_state['CALLSTATE']==3].shape[0]/float(nsh)
    num_of_state_4 = num_of_state.loc[num_of_state['CALLSTATE']==4].shape[0]/float(nsh)
    del num_of_state
    # print("state0", num_of_state_0[0])

    ### 地点特征
    startlong = temp.loc[0, 'LONGITUDE']
    # print("startlong",startlong)
    startlat  = temp.loc[0, 'LATITUDE']
    hdis1 = haversine1(startlong, startlat, 113.9177317,22.54334333)  # 距离某一点的距离

    # 单次行程方向变化
    trip_hasNon_direction = 1 if len(temp.loc[
                                         (temp.DIRECTION < 0), 'DIRECTION']) > 0 else 0
    trip_var_direction = temp['DIRECTION'].max() - temp['DIRECTION'].min()

    # 时间特征
    # temp['weekday'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).weekday())
    temp['hour'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    # print(temp['hour'])
    hour_state = np.zeros([24,1])
    # print("hour_state",hour_state)
    for i in range(24):
        hour_state[i] = temp.loc[temp['hour']==i].shape[0]/float(nsh)
        # print("hour_state", hour_state)

    hour_state_2_3 = hour_state[2] + hour_state[3]
    hour_state_5_10 = hour_state[5] + hour_state[6] + hour_state[7] + hour_state[8] + hour_state[9] + hour_state[10]
    hour_state_14_15 = hour_state[14] + hour_state[15]
    hour_state_16_21 = hour_state[16] + hour_state[17] + hour_state[18] + hour_state[19] + hour_state[20] + hour_state[21]
    hour_state_22_23 = hour_state[22] + hour_state[23]
    # 驾驶行为特征
    mean_speed = temp['SPEED'].mean()
    var_speed = temp['SPEED'].var()
    mean_height = temp['HEIGHT'].mean()

    # 单次行程时长
    trip_sum_interval = temp['TIME'].max() - temp['TIME'].min() + 60

    # 单次行程高度
    trip_avg_height = temp['HEIGHT'].mean()
    trip_max_height = temp['HEIGHT'].max()
    trip_var_height =temp['HEIGHT'].max() - temp['HEIGHT'].min()

    # 单次行程速度
    trip_avg_speed = temp['SPEED'].mean()
    trip_max_speed = temp['SPEED'].max()
    trip_var_speed = temp['SPEED'].max() - temp['SPEED'].min()

    # 单次加速度
    trip_acceleration_speed = trip_var_speed / trip_sum_interval
    trip_acceleration_height = trip_var_height / trip_sum_interval
    trip_acceleration_direction = trip_var_direction / trip_sum_interval

    #海拔除以时间
    trip_height=temp['HEIGHT'].max() - temp['HEIGHT'].min()
    trip_heightime=trip_height/num_of_records

    multi_speed_height = mean_speed * mean_height
    speed_state_negative = temp.loc[temp['SPEED'] < 0].shape[0] / float(nsh)
    direction_state_negative = temp.loc[temp['DIRECTION'] < 0].shape[0] / float(nsh)

    temp['month'] = temp['TIME'].apply(lambda x: datetime.date.fromtimestamp(x).month)
    month_state = np.zeros([12, 1])
    for i in range(12):
        month_state[i] = temp.loc[temp['month'] == i + 1].shape[0] / float(nsh)
    # month_state_3_4_5 = month_state[2] + month_state[3] + month_state[4]
    # month_state_6_7_8 = month_state[5] + month_state[6] + month_state[7]
    # month_state_9_10_11 = month_state[8] + month_state[9] + month_state[10]
    # month_state_12_1_2 = month_state[11] + month_state[0] + month_state[1]
    month_state_front = month_state[2:7].sum()
    month_state_behind = month_state[8:].sum() + month_state[0] + month_state[1]

    # 添加label
    target = temp.loc[0, 'Y']
    # print("target",target)

    # 所有特征
    feature = [item, num_of_trips, num_of_records,num_of_state_0,num_of_state_1,num_of_state_2,num_of_state_3,num_of_state_4,\
               mean_speed,var_speed,mean_height\
        ,float(hour_state[0]),float(hour_state[1]),float(hour_state[4])
        # ,float(hour_state[5])
        # ,float(hour_state[6]),float(hour_state[7]),float(hour_state[8]),float(hour_state[9]),float(hour_state[10])
        ,float(hour_state[11])
        ,float(hour_state[12]),float(hour_state[13])
        , float(hour_state_2_3)
        , float(hour_state_5_10)
               # ,float(hour_state_11_13)
        , float(hour_state_14_15)
        , float(hour_state_16_21)
        , float(hour_state_22_23)
        , float(month_state_front)
        , float(month_state_behind)
        # ,float(hour_state[16]),float(hour_state[17])
        # ,float(hour_state[18]),float(hour_state[19]),float(hour_state[20]),float(hour_state[21]),float(hour_state[22]),float(hour_state[23])
        ,hdis1,trip_var_direction,trip_sum_interval,trip_acceleration_direction,trip_acceleration_speed,multi_speed_height,speed_state_negative,direction_state_negative
        ,target]

    train1.append(feature)
    # print("train1",train1)
train1 = pd.DataFrame(train1)
# print("train1",train1)
# train1.to_csv('result1.csv',header=True,index=False)

# 特征命名
featurename = ['item', 'num_of_trips', 'num_of_records','num_of_state_0','num_of_state_1','num_of_state_2','num_of_state_3','num_of_state_4',\
              'mean_speed','var_speed','mean_height'
    ,'h0','h1','h4'
    # ,'h5','h6','h7','h8','h9','h10'
    ,'h11','h12','h13'
    # ,'h16','h17','h18','h19','h20','h21','h22','h23'
    , 'hour_state_2_3'
    , 'hour_state_5_10'
    , 'hour_state_14_15'
    , 'hour_state_16_21'
    , 'hour_state_22_23'
    ,'month_state_front'
    ,'month_state_behind'
    ,'dis','trip_var_direction','trip_sum_interval','trip_acceleration_direction','trip_acceleration_speed','multi_speed_height','speed_state_negative','direction_state_negative'
    ,'target']
train1.columns = featurename

print("train data process time:",(datetime.datetime.now()-start_all).seconds)

# Train model
feature_use = [ 'num_of_trips', 'num_of_records','num_of_state_0','num_of_state_1','num_of_state_2','num_of_state_3','num_of_state_4',\
               'mean_speed','var_speed','mean_height'
    ,'h0','h1','h4'
    # ,'h5','h6','h7','h8','h9','h10'
    ,'h11' ,'h12','h13'
    # ,'h16','h17','h18','h19','h20','h21','h22','h23'
    , 'hour_state_2_3'
    , 'hour_state_5_10'
    , 'hour_state_14_15'
    , 'hour_state_16_21'
    , 'hour_state_22_23'
     ,'month_state_front'
    ,'month_state_behind'
    ,'dis','trip_var_direction','trip_sum_interval','trip_acceleration_direction','trip_acceleration_speed','multi_speed_height','speed_state_negative','direction_state_negative']

params = {
    "objective": 'reg:linear',
    "eval_metric":'rmse',
    "seed":1123,
    "booster": "gbtree",
    "min_child_weight":5,
    "gamma":0.04,
    "max_depth": 3,
    "eta": 0.038,
    "silent": 1,
    "subsample":0.75102,
    "colsample_bytree":0.4,
    "scale_pos_weight":0.91
    # "nthread":16
}
# train1[feature_use].fillna(-1)
# print(train1[feature_use])
df_train = xgb.DMatrix(train1[feature_use].fillna(-1), train1['target'])
# print(train1[feature_use])
gbm = xgb.train(params,df_train,num_boost_round=1100)

# save model to file
pickle.dump(gbm, open("pima.pickle.dat", "wb"))
print("training end:",(datetime.datetime.now()-start_all).seconds)
# clf=GridSearchCV()

# The same process for the test set
data = pd.read_csv(path_test)
test1 = []

for item in data['TERMINALNO'].unique():

    print('user NO:',item)

    temp = data.loc[data['TERMINALNO'] == item,:]
    temp.index = range(len(temp))

    # trip 特征
    num_of_trips = temp['TRIP_ID'].nunique()

    # record 特征
    num_of_records = temp.shape[0]
    num_of_state = temp[['TERMINALNO','CALLSTATE']]
    nsh = num_of_state.shape[0]
    num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE']==0].shape[0]/float(nsh)
    num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE']==1].shape[0]/float(nsh)
    num_of_state_2 = num_of_state.loc[num_of_state['CALLSTATE']==2].shape[0]/float(nsh)
    num_of_state_3 = num_of_state.loc[num_of_state['CALLSTATE']==3].shape[0]/float(nsh)
    num_of_state_4 = num_of_state.loc[num_of_state['CALLSTATE']==4].shape[0]/float(nsh)
    del num_of_state


    ### 地点特征
    startlong = temp.loc[0, 'LONGITUDE']
    startlat  = temp.loc[0, 'LATITUDE']

    hdis1 = haversine1(startlong, startlat, 113.9177317,22.54334333)



    # 时间特征
    # temp['weekday'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).weekday())
    temp['hour'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    hour_state = np.zeros([24,1])
    for i in range(24):
        hour_state[i] = temp.loc[temp['hour']==i].shape[0]/float(nsh)
    hour_state_2_3 = hour_state[2] + hour_state[3]
    hour_state_5_10 = hour_state[5] + hour_state[6] + hour_state[7] + hour_state[8] + hour_state[9] + hour_state[10]
    hour_state_14_15 = hour_state[14] + hour_state[15]
    hour_state_16_21 = hour_state[16] + hour_state[17] + hour_state[18] + hour_state[19] + hour_state[20] + hour_state[21]
    hour_state_22_23 = hour_state[22] + hour_state[23]
    # 驾驶行为特征
    mean_speed = temp['SPEED'].mean()
    var_speed = temp['SPEED'].var()
    mean_height = temp['HEIGHT'].mean()

    # 单次行程时长
    trip_sum_interval = temp['TIME'].max() - temp['TIME'].min() + 60

    # 单次行程高度
    trip_avg_height = temp['HEIGHT'].mean()
    trip_max_height = temp['HEIGHT'].max()
    trip_var_height = temp['HEIGHT'].max() - temp['HEIGHT'].min()

    # 单次行程速度
    trip_avg_speed = temp['SPEED'].mean()
    trip_max_speed = temp['SPEED'].max()
    trip_var_speed = temp['SPEED'].max() - temp['SPEED'].min()

    # 单次加速度
    trip_acceleration_speed = trip_var_speed / trip_sum_interval
    trip_acceleration_height = trip_var_height / trip_sum_interval
    trip_acceleration_direction = trip_var_direction / trip_sum_interval

    multi_speed_height = mean_speed * mean_height
    speed_state_negative = temp.loc[temp['SPEED'] < 0].shape[0] / float(nsh)
    direction_state_negative = temp.loc[temp['DIRECTION'] < 0].shape[0] / float(nsh)

    temp['month'] = temp['TIME'].apply(lambda x: datetime.date.fromtimestamp(x).month)
    month_state = np.zeros([12, 1])
    for i in range(12):
        month_state[i] = temp.loc[temp['month'] == i + 1].shape[0] / float(nsh)
    # month_state_3_4_5 = month_state[2] + month_state[3] + month_state[4]
    # month_state_6_7_8 = month_state[5] + month_state[6] + month_state[7]
    # month_state_9_10_11 = month_state[8] + month_state[9] + month_state[10]
    # month_state_12_1_2 = month_state[11] + month_state[0] + month_state[1]
    month_state_front = month_state[2:7].sum()
    month_state_behind = month_state[8:].sum() + month_state[0] + month_state[1]

    # test标签设为-1
    target = -1.0

    feature = [item, num_of_trips, num_of_records,num_of_state_0,num_of_state_1,num_of_state_2,num_of_state_3,num_of_state_4,\
               mean_speed,var_speed,mean_height\
        ,float(hour_state[0]),float(hour_state[1]),float(hour_state[4])
        # ,float(hour_state[5]),float(hour_state[6]),float(hour_state[7]),float(hour_state[8]),float(hour_state[9]),float(hour_state[10])
        ,float(hour_state[11]),float(hour_state[12]),float(hour_state[13])
        # ,float(hour_state[16]),float(hour_state[17]),float(hour_state[18]),float(hour_state[19]),float(hour_state[20]),float(hour_state[21]),float(hour_state[22]),float(hour_state[23])
        , float(hour_state_2_3)
        , float(hour_state_5_10)
               # ,float(hour_state_11_13)
        , float(hour_state_14_15)
        , float(hour_state_16_21)
        , float(hour_state_22_23)
        , float(month_state_front)
        , float(month_state_behind)
        ,hdis1,trip_var_direction,trip_sum_interval,trip_acceleration_direction,trip_acceleration_speed,multi_speed_height,speed_state_negative,direction_state_negative
        ,target]

    test1.append(feature)

# make predictions for test data
test1 = pd.DataFrame(test1)
test1.columns = featurename
df_test = xgb.DMatrix(test1[feature_use].fillna(-1))
# print(test1[feature_use])
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
y_pred = loaded_model.predict(df_test)

# output result
result = pd.DataFrame(test1['item'])
result['pre'] = y_pred
result = result.rename(columns={'item':'Id','pre':'Pred'})
# result.to_csv('./model/result_.csv',header=True,index=False)
result.to_csv('result.csv',header=True,index=False)

print("Time used:",(datetime.datetime.now()-start_all).seconds)

# '''

