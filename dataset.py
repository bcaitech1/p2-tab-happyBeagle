import pandas as pd 
import numpy as np
from collections import defaultdict

TOTAL_THRES = 300# 구매액 임계값
SEED = 42 # 랜덤 시드


def getDatasetbyCustomerandMonth(raw_train_data):
    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')
    raw_train_data['year'] = raw_train_data['order_date'].dt.strftime('%Y')

    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')
    raw_train_data['year'] = raw_train_data['order_date'].dt.strftime('%Y')

    month_data = pd.unique(raw_train_data.year_month)
    user_list = pd.unique(raw_train_data.customer_id)

    new_data_list = {user_id : {month : 0 for month in month_data} for user_id in user_list}

    for i, temp in raw_train_data.iterrows():
        month = temp.year_month
        customer_id = temp.customer_id
        total = temp.total
        new_data_list[customer_id][month] += total
    
    customer_id = []
    total_by_month = {month : [] for month in month_data}

    for key, data in new_data_list.items():
        customer_id.append(key)
        for month, total in data.items():
            total_by_month[month].append(total)

    df_data = total_by_month
    df_data["customer_id"] = customer_id
    feature = ["customer_id", "prev", '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
    train_raw_data = {}
    train_raw_data[feature[0]] = df_data[feature[0]]
    train_raw_data[feature[1]] = df_data["2009-12"]
    for month in feature[2:]:
        train_raw_data[month] = df_data["2010-"+month]
    label = []
    for price in df_data["2010-12"]:
        label.append(int(price >= TOTAL_THRES)) 
    train_raw_data["label"] = label  
    train_data = pd.DataFrame(train_raw_data)

    test_raw_data = {}
    test_raw_data[feature[0]] = df_data[feature[0]]
    test_raw_data[feature[1]] = df_data["2010-12"]
    for month in feature[2:]:
        test_raw_data[month] = df_data["2011-"+month]
    test_data = pd.DataFrame(test_raw_data)

    return train_data, test_data, train_data["label"], feature[1:]

def getDatasetbyCustomerandMonthConsiderFirstBuy(raw_train_data):
    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')
    raw_train_data['year'] = raw_train_data['order_date'].dt.strftime('%Y')

    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')
    raw_train_data['year'] = raw_train_data['order_date'].dt.strftime('%Y')

    month_data = pd.unique(raw_train_data.year_month)
    user_list = pd.unique(raw_train_data.customer_id)

    new_data_list = {user_id : {month : -50000 for month in month_data} for user_id in user_list}

    for i, temp in raw_train_data.iterrows():
        month = temp.year_month
        customer_id = temp.customer_id
        total = temp.total
        if new_data_list[customer_id][month] == -50000:
            new_data_list[customer_id][month] = 0
        new_data_list[customer_id][month] += total
    
    for customer_id, dates in new_data_list.items():
        flag = False
        for month, total in dates.items():
            if total >= 0:
                flag = True
            if (flag) & (total ==-50000 ):
                new_data_list[customer_id][month] = 0

    customer_id = []
    total_by_month = {month : [] for month in month_data}

    for key, data in new_data_list.items():
        customer_id.append(key)
        for month, total in data.items():
            total_by_month[month].append(total)

    df_data = total_by_month
    df_data["customer_id"] = customer_id
    feature = ["customer_id", "prev", '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
    train_raw_data = {}
    train_raw_data[feature[0]] = df_data[feature[0]]
    train_raw_data[feature[1]] = df_data["2009-12"]
    for month in feature[2:]:
        train_raw_data[month] = df_data["2010-"+month]
    label = []
    for price in df_data["2010-12"]:
        label.append(int(price >= TOTAL_THRES)) 
    train_raw_data["label"] = label  
    train_data = pd.DataFrame(train_raw_data)

    test_raw_data = {}
    test_raw_data[feature[0]] = df_data[feature[0]]
    test_raw_data[feature[1]] = df_data["2010-12"]
    for month in feature[2:]:
        test_raw_data[month] = df_data["2011-"+month]
    test_data = pd.DataFrame(test_raw_data)

    return train_data, test_data, train_data["label"], feature[1:]

def getDatasetbyCustomerandMonthConsiderFirstBuyWithAverageConsume(raw_train_data):

    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')
    raw_train_data['year'] = raw_train_data['order_date'].dt.strftime('%Y')

    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')
    raw_train_data['year'] = raw_train_data['order_date'].dt.strftime('%Y')

    month_data = pd.unique(raw_train_data.year_month)
    user_list = pd.unique(raw_train_data.customer_id)

    new_data_list = {user_id : {month : -50000 for month in month_data} for user_id in user_list}

    for i, temp in raw_train_data.iterrows():
        month = temp.year_month
        customer_id = temp.customer_id
        total = temp.total
        if new_data_list[customer_id][month] == -50000:
            new_data_list[customer_id][month] = 0
        new_data_list[customer_id][month] += total
    
    customers_average_consumption = []
    for customer_id, dates in new_data_list.items():
        flag = False
        consumption = 0
        count = 0 
        for month, total in dates.items():
            if (not flag) & (total != -50000):
                flag = True
            if (flag) & (total ==-50000 ):
                new_data_list[customer_id][month] = 0
            else:
                consumption += total
                count += 1
        customers_average_consumption.append(consumption/count)

    customer_id = []
    total_by_month = {month : [] for month in month_data}

    for key, data in new_data_list.items():
        customer_id.append(key)
        for month, total in data.items():
            total_by_month[month].append(total)

    df_data = total_by_month
    df_data["customer_id"] = customer_id
    feature = ["customer_id", "prev", "average_consumption", '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
    train_raw_data = {}
    train_raw_data[feature[0]] = df_data[feature[0]]
    train_raw_data[feature[1]] = df_data["2009-12"]
    train_raw_data[feature[2]] = customers_average_consumption
    for month in feature[3:]:
        train_raw_data[month] = df_data["2010-"+month]
    label = []
    for price in df_data["2010-12"]:
        label.append(int(price >= TOTAL_THRES)) 
    train_raw_data["label"] = label  
    train_data = pd.DataFrame(train_raw_data)

    test_raw_data = {}
    test_raw_data[feature[0]] = df_data[feature[0]]
    test_raw_data[feature[1]] = df_data["2010-12"]
    test_raw_data[feature[2]] = customers_average_consumption
    for month in feature[3:]:
        test_raw_data[month] = df_data["2011-"+month]
    test_data = pd.DataFrame(test_raw_data)

    return train_data, test_data, train_data["label"], feature[1:]

def getDatasetVer1(raw_train_data):

    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')


    month_data = pd.unique(raw_train_data.year_month)
    user_list = pd.unique(raw_train_data.customer_id)

    new_data_list_total = {user_id : {month : -50000 for month in month_data} for user_id in user_list}
    new_data_list_mean = {user_id : {month : -50000 for month in month_data} for user_id in user_list}
    for i, temp in raw_train_data.iterrows():
        month = temp.year_month
        customer_id = temp.customer_id
        total = temp.total
        if new_data_list_total[customer_id][month] == -50000:
            new_data_list_total[customer_id][month] = 0
        new_data_list_total[customer_id][month] = total
    
        if new_data_list_mean[customer_id][month] == -50000:
            new_data_list_mean[customer_id][month] = 0.
        new_data_list_mean[customer_id][month] += 1.

    for user_id, personal_data in new_data_list_total.items():
        first = False
        for month, total in personal_data.items():
            if not first and total != -50000:
                first = True
            if first and total == -50000:
                new_data_list_total[user_id][month] = 0
            count = new_data_list_mean[user_id][month]
            new_data_list_mean[user_id][month] = new_data_list_total[user_id][month] / count

    raw_train_data = defaultdict(list)
    feature_date = ["12", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    year = 2010
    for user_id in user_list:
        raw_train_data["customer_id"].append(user_id)
        raw_train_data["label"].append(int(new_data_list_total[user_id][str(year)+'-12'] >= 300))
        for month in feature_date:
            month_label = ""
            if month == "12":
                month_label = str(year - 1) + "-" + month
            else:
                month_label = str(year) + "-" + month
            
            raw_train_data[month + "_total"].append(new_data_list_total[user_id][month_label])
         #   raw_train_data[month + "_mean"].append(new_data_list_mean[user_id][month_label])
        
    train_data = pd.DataFrame(raw_train_data)
    feature = train_data.columns[2:]

    year = 2011
    raw_test_data = defaultdict(list)
    for user_id in user_list:
        raw_test_data["customer_id"].append(user_id)
        for month in feature_date:
            model_label = ""
            if month == "12":
                month_label = str(year - 1) + "-" + month
            else:
                month_label = str(year) + "-" + month 

            raw_test_data[month+"_total"].append(new_data_list_total[user_id][month_label]) 
         #   raw_test_data[month+"_mean"].append(new_data_list_mean[user_id][month_label])

    test_data = pd.DataFrame(raw_test_data)

    return train_data, test_data, train_data["label"], feature


def getDatasetVer2(raw_train_data):
    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')


    month_data = pd.unique(raw_train_data.year_month)
    user_list = pd.unique(raw_train_data.customer_id)

    new_data_list = {user_id : {month : [] for month in month_data} for user_id in user_list}
    
    for i, temp in raw_train_data.iterrows():
        month = temp.year_month
        customer_id = temp.customer_id
        total = temp.total
        new_data_list[customer_id][month].append(total)
    

    raw_data = defaultdict(list)
    for customer_id, date_list in new_data_list.items():
        raw_data["user_id"].append(customer_id)
        idx = len(raw_data["first_buy"])
        for date, prices in date_list.items():
            if len(prices) > 0:
                price_list = np.array(prices)
            else:
                price_list = np.array([0])
            raw_data[date+":total"].append(np.sum(price_list))
            raw_data[date+":mean"].append(np.mean(price_list))
            raw_data[date+":std"].append(np.std(price_list))
            if len(prices) != 0 and idx == len(raw_data["first_buy"]):
                raw_data["first_buy"].append(date)
                raw_data["first_buy_price"].append(price_list[0])
                raw_data["first_buy_total"].append(np.sum(price_list))
    train_year = 2010

    train_raw_data = {}
    train_raw_data["customer_id"] = raw_data["user_id"]
    train_raw_data["label"] = list((np.array(raw_data[str(train_year) + "-12:total"]) >= 300).astype(int))     
    
    #train_raw_data["first_buy_month"] = raw_data["first_buy"]
    train_raw_data["first_buy_price"] = raw_data["first_buy_price"]
    train_raw_data["first_buy_total"] = raw_data["first_buy_total"]
    months = []
    for i in range(1, 12):
        month = str(i).zfill(2)
        months.append(month)

    statics = ["total", "mean", "std"]

    for month in months:
        if month == '12':
            label = str(train_year - 1)
        else:
            label = str(train_year)
        label += '-'+month
        for s in statics:
            train_raw_data[month + "_" + s] = raw_data[label + ":" + s]
    train_data = pd.DataFrame(train_raw_data)

    test_raw_data = {}
    test_raw_data["customer_id"] = raw_data["user_id"]
    test_raw_data["label"] = [0] * len(raw_data["user_id"])
    
    #test_raw_data["first_buy_month"] = raw_data["first_buy"]
    test_raw_data["first_buy_price"] = raw_data["first_buy_price"]
    test_raw_data["first_buy_total"] = raw_data["first_buy_total"]

    test_year = 2011

    for month in months:
        if month == '12':
            label = str(train_year - 1)
        else:
            label = str(train_year)
        label += '-'+month
        for s in statics:
            test_raw_data[month + "_" + s] = raw_data[label + ":" + s]
    
    test_data = pd.DataFrame(test_raw_data)
    
    return train_data, test_data, train_data["label"], test_data.columns[2:]


def getDatasetVer3(raw_train_data):
    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')

    month_data = pd.unique(raw_train_data.year_month)
    user_list = pd.unique(raw_train_data.customer_id)

    new_data_list = {user_id : {month : 0 for month in month_data} for user_id in user_list}
    
    for i, temp in raw_train_data.iterrows():
        month = temp.year_month
        customer_id = temp.customer_id
        total = temp.total
        new_data_list[customer_id][month] += total

    raw_data = defaultdict(list)

    for customer_id, date_list in new_data_list.items():
        raw_data["user_id"].append(customer_id)
        more_than_300_cnt_2010 = 0
        more_than_300_2010 = 0
        more_than_300_cnt_2011 = 0
        more_than_300_2011 = 0
    
        total_2010 = 0
        total_2011 = 0
        for date, price in date_list.items():
            if int(date[:4]) <= 2010:
                more_than_300_cnt_2010 += 1 if price >= 300 else 0
                more_than_300_2010 += price if price >= 300 else 0
                total_2010 += price
            more_than_300_cnt_2011 += 1 if price >= 300 else 0
            more_than_300_2011 += price if price >= 300 else 0
            total_2011 += price
           
            raw_data[date].append(price)
        raw_data["more_than_300_cnt_2010"].append(more_than_300_cnt_2010)
        raw_data["more_than_300_avg_2010"].append(more_than_300_2010 / more_than_300_cnt_2010 if more_than_300_cnt_2010 > 0 else 0)
        raw_data["total_used_2010"].append(total_2010)
        raw_data["more_than_300_cnt_2011"].append(more_than_300_cnt_2011)
        raw_data["more_than_300_avg_2011"].append(more_than_300_2011 / more_than_300_cnt_2011 if more_than_300_cnt_2011 > 0 else 0)
        raw_data["total_used_2011"].append(total_2011)

    months = ['12']
    for i in range(1, 12):
        month = str(i).zfill(2)
        months.append(month)

    train_year = 2010

    train_raw_data = {}
    train_raw_data["customer_id"] = raw_data["user_id"]
    train_raw_data["label"] = list((np.array(raw_data[str(train_year) + "-12"]) >= 300).astype(int))     

    train_raw_data["more_than_300_cnt"] = raw_data["more_than_300_cnt_2010"]
    train_raw_data["more_than_300_avg"] = raw_data["more_than_300_avg_2010"]
    train_raw_data["total_used"] = raw_data["total_used_2010"]


    for month in months:
        if month == '12':
            label = str(train_year - 1)
        else:
            label = str(train_year)
        label += '-'+month
        train_raw_data[month] = raw_data[label]
    train_data = pd.DataFrame(train_raw_data)

    test_raw_data = {}
    test_raw_data["customer_id"] = raw_data["user_id"]
    test_raw_data["label"] = [0] * len(raw_data["user_id"])

    test_raw_data["more_than_300_cnt"] = raw_data["more_than_300_cnt_2011"]
    test_raw_data["more_than_300_avg"] = raw_data["more_than_300_avg_2011"]
    test_raw_data["total_used"] = raw_data["total_used_2011"]

    test_year = 2011

    for month in months:
        if month == '12':
            label = str(test_year - 1)
        else:
            label = str(test_year)
        label += '-'+month
        test_raw_data[month] = raw_data[label]
    test_data = pd.DataFrame(test_raw_data)   

    return train_data, test_data, train_data["label"], train_data.columns[2:]



def myDataset(raw_train_data):
    raw_train_data['year_month'] = raw_train_data['order_date'].dt.strftime('%Y-%m')

    customer_id_list = pd.unique(raw_train_data.customer_id)
    month_list = pd.unique(raw_train_data.year_month)

    raw_data = {customer_id : {month : 0 for month in month_list} for customer_id in customer_id_list}

    for i, data in raw_train_data.iterrows():
        raw_data[data.customer_id][data.year_month] += data.total

    months = ['12']
    for i in range(1, 12):
        months.append(str(i).zfill(2))

    raw_data_all = defaultdict(list)

    for customer_id, consume_list in raw_data.items():
        raw_data_all["customer_id"].append(customer_id)
        more_than_300_count_all, more_than_300_count_2010 = 0, 0
        more_than_300_total_all, more_than_300_total_2010 = 0, 0
        total_used_all, total_used_2010 = 0, 0
        flag = False
        for month, total in consume_list.items():
            if (not flag) and (total != 0):
                flag = True
            elif (not flag) and (total == 0) :
                total = -500000
            if total >= 300:
                now_year = int(month[:4])
                if now_year == 2010 and int(month[5:]) != 12:
                    more_than_300_count_2010 += 1
                    more_than_300_total_2010 += total
                elif now_year == 2011:
                    more_than_300_count_all += 1
                    more_than_300_total_all += total
            if int(month[:4]) <= 2010:
                total_used_2010 += total if total > -500000 else 0
            raw_data_all[month].append(total)
            total_used_all += total if total > -500000 else 0
        raw_data_all["more_than_300_count_all"].append(more_than_300_count_all)
        raw_data_all["more_than_300_total_all"].append(more_than_300_total_all)
        raw_data_all["more_than_300_avg_all"].append((more_than_300_total_all / more_than_300_count_all) if more_than_300_count_all > 0 else 0)
        raw_data_all["more_than_300_count_2010"].append(more_than_300_count_2010)
        raw_data_all["more_than_300_total_2010"].append(more_than_300_total_2010)
        raw_data_all["more_than_300_avg_2010"].append((more_than_300_total_2010 / more_than_300_count_2010) if more_than_300_count_2010 > 0 else 0)

        raw_data_all["consum_total_all"].append(total_used_all)
        raw_data_all["consum_total_2010"].append(total_used_2010) 
    
    train_year = 2010

    train_raw_data = {}
    train_raw_data["customer_id"] = raw_data_all["customer_id"]
    train_raw_data["label"] = list((np.array(raw_data_all[str(train_year) + "-12"]) >= 300).astype(int))
    train_raw_data["more_than_300_count"] = raw_data_all["more_than_300_count_2010"]
    train_raw_data["more_than_300_avg"] = raw_data_all["more_than_300_avg_2010"]
    train_raw_data["more_than_300_total"] = raw_data_all["more_than_300_total_2010"]
    train_raw_data["consum_total"] = raw_data_all["consum_total_2010"]

    for month in months[1:]:
        if month == '12':
            label = str(train_year - 1)
        else:
            label = str(train_year)
        label +=  "-" + month
        train_raw_data[month] = raw_data_all[label]
    train_data = pd.DataFrame(train_raw_data)


    test_year = 2011

    test_raw_data = {}
    test_raw_data["customer_id"] = raw_data_all["customer_id"]
    test_raw_data["more_than_300_count"] = raw_data_all["more_than_300_count_all"]
    test_raw_data["more_than_300_avg"] = raw_data_all["more_than_300_avg_all"]
    test_raw_data["more_than_300_total"] = raw_data_all["more_than_300_total_all"]
    test_raw_data["consum_total"] = raw_data_all["consum_total_all"]

    for month in months[1:]:
        if month == '12':
            label = str(test_year - 1)
        else:
            label = str(test_year)
        label +=  "-" + month
        test_raw_data[month] = raw_data_all[label]
    test_data = pd.DataFrame(test_raw_data)

    return train_data, test_data, train_data["label"], train_data.columns[2:]


def myDatasetVer1():
    train_data = pd.read_csv("/opt/ml/code/my_src/data/train_data_thres_rate_3_6_12.csv")
    test_data = pd.read_csv("/opt/ml/code/my_src/data/test_data_thres_rate_3_6_12.csv")

    feature = list(train_data.columns[2:7])
    feature.extend(list(train_data.columns[-3:]))
    return train_data, test_data, train_data["label"], train_data.columns[2:]