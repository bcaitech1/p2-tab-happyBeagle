import pandas as pd 
import numpy as np
import datetime
from tqdm import tqdm

def generate_label(df, user_list, thres = 300, print_log=False):
    df = df.copy()
    
    # year_month에 해당하는 label 데이터 생성
    df = df[["customer_id", "total"]].groupby("customer_id").sum()
    
    label_list = {user_id: 0 for user_id in user_list}
    for key, total in df.iterrows():
        label_list[key] = float(total)
    
    label = []
    for user in user_list:
        label.append(1 if label_list[user] >= 300 else 0)
    return pd.DataFrame({"label": label})

def get_final_data(df, agg_func, data_type="training dataset", plus_columns=['total-mean', 'total-max', 'total-min', 'total-sum', 'total-count', 'total-std', 'total-skew']):
    print("[Info] get_final_data........")
    ret_data = pd.DataFrame()
    for i, tr_ym in tqdm(enumerate(df['year_month'].unique()), desc=data_type):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = df.loc[df['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_func)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in train_agg.columns.levels[0]:
            for stat in train_agg.columns.levels[1]:
                new_cols.append(f'{col}-{stat}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        ret_data = ret_data.append(train_agg)
    
    for new_col in plus_columns:
        ret_data[new_col+"-1"] = ret_data[new_col]
    return ret_data


def generate_pixel(df, customer_list, data_type="training dataset", img_size=28):
    print("[Info] generate_pixel........")
    monthes = pd.unique(df["year_month"])
    period = img_size // len(monthes)
    plus_num = img_size % len(monthes)
    month_list = [(date + "_" + str(num)) for num in range(0, period) for date in monthes]

    for date in monthes[-1 * plus_num:]:
        month_list.append(date + "_" + str(period))
    
    feature_list = df.drop(["customer_id", "year_month"], axis=1).columns

    feature_dict = {f: idx for idx, f in enumerate(feature_list)}
    month_dict = {month: idx for idx, month in enumerate(month_list)}
    max_list = df.max()
    min_list = df.min()

    ret_data = {i: [] for i in range(0,img_size * img_size)}
    ret_data["id"] = []

    for customer_id in tqdm(customer_list, desc=data_type):
        temp = df[df.customer_id == customer_id]
        ret_data["id"].append(customer_id)
        for date in month_list:
            for feature in feature_list:
                label = date.split("_")[0]
                max_num = float(max_list[feature])
                min_num = float(min_list[feature])

                try:
                    num = float(temp[temp.year_month == label][feature])
                    num = int(((num - min_num) / (max_num - min_num)) * 255.) if max_num != min_num else 0
                except:
                    num = 0. 

                key = feature_dict[feature] * 28 + month_dict[date]
                ret_data[key].append(num)
    return pd.DataFrame(ret_data)

def generate_image(raw_data, train_year, valid_year, test_year, agg_func=['mean','max','min','sum','count','std','skew']):
    raw_data["year_month"] = raw_data["order_date"].dt.strftime('%Y-%m')

    train = raw_data[(train_year["start"] <= raw_data["order_date"]) & (raw_data["order_date"] < train_year["end"])]
    validation = raw_data[(valid_year["start"] <= raw_data["order_date"]) & (raw_data["order_date"] < valid_year["end"])]
    test = raw_data[(test_year["start"] <= raw_data["order_date"]) & (raw_data["order_date"] < test_year["end"])]
    
    customer_list = pd.unique(raw_data.customer_id)

    train_label = generate_label(raw_data[raw_data.year_month == train_year["end"]], customer_list)
    valid_label = generate_label(raw_data[raw_data.year_month == valid_year["end"]], customer_list)
    test_label = generate_label(raw_data[raw_data.year_month == test_year["end"]], customer_list)
    
    train_data_final = get_final_data(train, agg_func, data_type="training dataset")
    train_data_final = train_data_final.fillna(0)

    valid_data_final = get_final_data(validation, agg_func, data_type="validation dataset")
    valid_data_final = valid_data_final.fillna(0)

    test_data_final = get_final_data(test, agg_func, data_type="test dataset")
    test_data_final = test_data_final.fillna(0)

    train_img_df = generate_pixel(train_data_final, customer_list, data_type="training dataset")
    valid_img_df = generate_pixel(valid_data_final, customer_list, data_type="validation dataset")
    test_img_df = generate_pixel(test_data_final, customer_list, data_type="test dataset")

    return train_img_df, train_label, valid_img_df, valid_label, test_img_df, test_label

    
if __name__=="__main__":
    train_data = pd.read_csv("/content/drive/MyDrive/boostcamp/stage2_tabular/my_src_1/input/train.csv", parse_dates=["order_date"])
    train_data["year_month"] = train_data["order_date"].dt.strftime('%Y-%m')

    train_year = {"start": "2009-12", "end": "2010-12"}
    valid_year = {"start": "2010-11", "end": "2011-11"}
    test_year = {"start": "2010-12", "end": "2011-12"}

    train, train_y, valid, valid_y, test, test_y = generate_image(train_data, train_year, valid_year, test_year)