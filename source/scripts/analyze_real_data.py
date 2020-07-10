import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
current_real_data_dir = "../../data/testing/27-06-2020-sns/"


def merge_metadata(current_real_data_dir, n_files=5):
#this function is for reading metadata files downloaded from tns server and merging
#them together into one for later easier analysis. it receives the initial data
#directory and the number of files in it.

    data_file = "tns_search.csv"
    df=pd.read_csv(current_real_data_dir+data_file)

    for i in np.arange(1,n_files+1):
        data_file = current_real_data_dir+"tns_search({}).csv".format(i)
        print(data_file)
        df2 = pd.read_csv(data_file)
        df = pd.concat([df,df2])
        # print(df.head())
        print(df.shape)

    print(df.keys())
    df=df.drop_duplicates(keep="first")

    df.to_csv(current_real_data_dir+"tns_search_sn_metadata.csv",index=False)

# merge_metadata(current_real_data_dir)

def autolabel(ax,rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# data_file="tns_search_sn_metadata.csv"
# real_sns = pd.read_csv(current_real_data_dir+data_file)
# print(real_sns)

# by_type = real_sns.groupby(["Obj. Type"]).count()

# parent_types = ["Ib", "Ic","SLSN", "Ia", "II"]
# parent_types_2 = ["Ib/c","SLSN","Ia","II"]
# by_type["Type"] = by_type.index.values

# by_type.loc[by_type["Type"].str.contains("SN Ib"),"Type"] = "Ib/c"
# by_type.loc[by_type["Type"].str.contains("SN Ic"),"Type"] = "Ib/c"
# by_type.loc[by_type["Type"].str.contains("SN II"),"Type"] = "II"
# by_type.loc[by_type["Type"].str.contains("SN Ia"),"Type"] = "Ia"
# by_type.loc[by_type["Type"].str.contains("SLSN"),"Type"] = "SLSN"

# by_parent_type = by_type.groupby(["Type"]).sum()
# print(by_parent_type)


# # print(by_type)
# # print(by_type.index.values)
# # print(by_type.ID.values)

# fig, ax = plt.subplots()
# rects = ax.bar(by_parent_type.index.values,by_parent_type.ID.values,color="#9c5b33")
# plt.xticks(rotation=45)
# autolabel(rects)
# # ax.legend()
# plt.show()

#by date

# print(real_sns["Discovery Date (UT)"])

# real_sns["by_month"]=real_sns["Discovery Date (UT)"]

# real_sns["by_month"] = real_sns["by_month"].str.split(" ").apply(lambda x : x[0])
# real_sns["by_month"] = real_sns["by_month"].str.split("-").apply(lambda x : x[0]+"-"+x[1])
# by_month = real_sns.groupby(["by_month"]).count()
# print(by_month.sum())

# fig, ax = plt.subplots()
# rects = ax.bar(by_month.index.values,by_month.ID.values,color="#6a3718")
# plt.xticks(rotation=45)
# autolabel(rects)
# # ax.legend()
# plt.show()

# ids = real_sns["Disc. Internal Name"].drop_duplicates()
# ids = ids.dropna()
# ids = ids[ids.str.contains("ZTF")]
# ids = ids.str.split(",").apply(lambda x: x[0].strip() if "ZTF" in x[0] else x[1].strip()).values

# id_str = ''
# for i in ids:
#     id_str= id_str+'"'+i+'", '

# print(id_str)

real_sns=None

for i in np.arange(65):
    data_file = "{}000-{}000.csv".format(i,i+1)
    print(data_file)
    real_sns_new = pd.read_csv(current_real_data_dir+data_file,sep="\t")
    if real_sns is not None:
        real_sns = pd.concat([real_sns, real_sns_new])
    else:
        real_sns = real_sns_new
    
real_sns.to_csv("tns_search.csv",index=False)

real_sns.columns = ["objid","time","flux","flux_err","band"]
print(real_sns)