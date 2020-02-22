# %%
# 上海、重庆、北京和各小城市累计病例从丁香园 csv中获取，运行前要先pull一下:~/Documents/DXY-COVID-19-Data$ git pull
# 上海内部交通从百度
# 外城市到上海流量从百度

# %%
import json
import numpy as np
import pandas as pd
import requests as rq
import datetime
import time
# %%
#先取得今天日期建立一个shanghai的记录csv
# %%
start = '20200120'#上海这天才有2例
start_date = datetime.datetime.strptime(start,'%Y%m%d').date()#字符串格式化为日期
#只能有昨天的数据,yesterday指的是昨天
yesterday = datetime.datetime.now().date()-datetime.timedelta(days=1)#.strftime('%Y%m%d') 
# print(yesterday) #20200217, 但是丁香园数据只到20200216
days=yesterday - start_date

Date_l=[]
Date_l =pd.date_range(start=start_date, end=yesterday) #日期序列

zeros=np.zeros(51*len(Date_l)).reshape(len(Date_l),51)
col_name=['Date', 'sh_acc_p1d', 'sh-internal_traffic', 'cq', 'wz', 'bj', 'xy',
       'hf', 'hz', 'nb', 'bb', 'fy', 'tz', 'sr', 'bz', 'nj', 'sq', 'sz', 'aq',
       'xz', 'zk', 'la', 'ha', 'wx', 'ly', 'lyg', 'jx', 'cq_num', 'bj_num',
       'wz_num', 'xy_num', 'hf_num', 'hz_num', 'nb_num', 'bb_num', 'fy_num',
       'tz_num', 'sr_num', 'bz_num', 'nj_num', 'sq_num', 'sz_num', 'aq_num',
       'xz_num', 'zk_num', 'la_num', 'ha_num', 'wx_num', 'ly_num', 'lyg_num',
       'jx_num']
sh_record=pd.DataFrame(data=zeros,columns=col_name)
sh_record.Date=Date_l
sh_record.Date= sh_record.Date.apply(lambda x: datetime.datetime.strftime(x,'%Y%m%d'))#日期改字符串
sh_record.head()

#第一步成功,输入日期，建立空表
# %%第二步
end_date=yesterday.strftime('%Y%m%d')
url='https://huiyan.baidu.com/migration/internalflowhistory.jsonp?dt=city&id=310000&date=%s&callback=jsonp_1581684251505_7585429' % (end_date)
headers = {'User-Agent':'Mozilla/5.0(Macintosh;IntelMacOSX10_7_0)AppleWebKit/535.11(KHTML,likeGecko)Chrome/79.0.3945.88Safari/535.11'}
t = rq.get(url, headers=headers).content.decode('unicode_escape','ignore')#.encode('unicode_escape')

n=t.find('(',0,30) #找到‘（
t1=t.lstrip(t[0:n+1]).rstrip(t[-1])#去掉jsonp_1581733157953_6359285(，去掉最后的）
t1
j=json.loads(t1) #字符串变json
#上午百度尚未更新昨天数据，就会这里出错
for date in sh_record.Date: #df中日期字符串对应的traffic cell是json 日期对应的值
    sh_record.loc[sh_record[sh_record.Date==date].index[0],'sh-internal_traffic']=j['data']['list'][date]

#第二步取得上海室内流量成功
# %%
#第三步 pull dxy 数据到本地
# %% 第四步 下载大城市数据-上海有点数据不对的。
d_num=pd.read_csv('/home/will/Documents/DXY-COVID-19-Data/csv/DXYArea.csv')
d_num['date']=pd.to_datetime(d_num.updateTime) #将时间戳字符串改成时间戳
# d_num['Date']=datetime.datetime.strptime(d_num.date,'%Y%m%d')#只对字符串有用
d_num['Date']=d_num['date'].apply(lambda x: x.strftime('%Y%m%d'))#时间戳改为字符串，格式与上面相同

cq=d_num[d_num['provinceName']=='重庆市'].groupby(['Date'])['province_confirmedCount'].min()
cq

for date in sh_record.Date: #df中日期字符串对应的traffic cell是json 日期对应的值
    if (date in cq.index):
        sh_record.loc[(sh_record[sh_record.Date==date].index[0]-1),'cq_num']=cq[date]
sh_record.loc[sh_record[sh_record.Date==yesterday.strftime('%Y%m%d')].index[0],'cq_num']=cq[datetime.datetime.now().date().strftime('%Y%m%d')]

bj=d_num[d_num['provinceName']=='北京市'].groupby(['Date'])['province_confirmedCount'].min()
for date in sh_record.Date: #df中日期字符串对应的traffic cell是json 日期对应的值
    if (date in bj.index):
        sh_record.loc[(sh_record[sh_record.Date==date].index[0]-1),'bj_num']=bj[date]
sh_record.loc[sh_record[sh_record.Date==yesterday.strftime('%Y%m%d')].index[0],'bj_num']=bj[datetime.datetime.now().date().strftime('%Y%m%d')]

sh=d_num[d_num['provinceName']=='上海市'].groupby(['Date'])['province_confirmedCount'].min()
for date in sh_record.Date: #df中日期字符串对应的traffic cell是json 日期对应的值, 这里没有今天0217
    if (date in sh.index):
        sh_record.loc[(sh_record[sh_record.Date==date].index[0]-2),'sh_acc_p1d']=sh[date]
#当天因为17号而数据sh_record中对应最多16号，会17号少一个对应
sh_record.loc[sh_record[sh_record.Date==(yesterday-datetime.timedelta(days=1)).strftime('%Y%m%d')].index[0],'sh_acc_p1d']=sh[datetime.datetime.now().date().strftime('%Y%m%d')]
print('yesterday included')
# %% 处理上海数据
# sh_record=pd.read_csv('data/shanghai-data-20200216.csv')
zeros=np.zeros(sh_record.shape[0])
sh_record['sh_add']=zeros

i=1
while i < sh_record.shape[0]-1:
    sh_record.loc[i,'sh_add']=sh_record.loc[i,'sh_acc_p1d']-sh_record.loc[i-1,'sh_acc_p1d']
    i+=1
sh_record.loc[0,'sh_add']=5 #20日为首日，增加了2个，也共2个。21日为7个
sh_record.loc[1,'sh_add']=2
sh_record.loc[2,'sh_add']=7
sh_record.loc[3,'sh_add']=17
sh_record.loc[4,'sh_add']=7
sh_record.loc[5,'sh_add']=13
#下载大城市的数据成功
# %% 第五步 下载小城市数据
city_cn1=['温州','信阳','合肥','杭州','宁波','蚌埠','阜阳','台州','上饶',
    '亳州','南京','商丘','苏州','安庆','徐州','周口','六安','淮安','无锡','临沂','连云港','嘉兴','武汉']
city_list1=['wz_num','xy_num','hf_num','hz_num','nb_num','bb_num','fy_num','tz_num','sr_num','bz_num','nj_num',
    'sq_num','sz_num','aq_num','xz_num','zk_num','la_num','ha_num','wx_num','ly_num','lyg_num','jx_num','wh_num']
k=0
while k< len(city_cn1):
    temp=d_num[d_num['cityName']==city_cn1[k]].groupby(['Date'])['city_confirmedCount'].min()
    print(temp) #这个城市名的代表
    print(city_list1[k])
    # shanghai_data[city_list1[k]]=zeros
    #商丘从27号开始
    # shanghai_data[city_list1[k]][5:]=shanghai_data['Date'][5:].apply(lambda x: temp[str(x)] )
    # sh_record[city_list1[k]][5+23-len(temp):]=sh_record['Date'][5+23-len(temp):].apply(lambda x: temp[str(x)] )
    for date in sh_record.Date: #df中日期字符串对应的traffic cell是json 日期对应的值, 这里没有今天0217
        if (date in temp.index):
            sh_record.loc[(sh_record[sh_record.Date==date].index[0]-1),city_list1[k]]=temp[date]
    #当天因为17号而数据sh_record中对应最多16号，会17号少一个对应
    sh_record.loc[sh_record[sh_record.Date==yesterday.strftime('%Y%m%d')].index[0],city_list1[k]]=temp[datetime.datetime.now().date().strftime('%Y%m%d')]
    k+=1
# 下载小城市的数据成功

# %% 第六步 成功将百度迁徙到上海流量下载！！
import datetime
import time
start = '2020-01-20'
start_date = datetime.datetime.strptime(start,'%Y-%m-%d').date() #字符串转日期
# end = '2020-02-14' #end = '2020-02-14'
end_date = datetime.datetime.today().date()
zeros=np.zeros(9000).reshape(3000,3)
record=pd.DataFrame(data=zeros,columns=['date','city','traffic'])

k =0

while (datetime.timedelta(days=k)+start_date)<end_date:
    # print(datetime.timedelta(days=k)+start_date) #日期相加
    # print((datetime.timedelta(days=k)+start_date).strftime('%Y%m%d')) #日期格式转字符串 #https://segmentfault.com/a/1190000014934953
    date=(datetime.timedelta(days=k)+start_date).strftime('%Y%m%d') #日期相加
    url='https://huiyan.baidu.com/migration/cityrank.jsonp?dt=province&id=310000&type=move_in&date=%s&callback=jsonp_1581733157953_6359285' % (date)
    headers = {'User-Agent':'Mozilla/5.0(Macintosh;IntelMacOSX10_7_0)AppleWebKit/535.11(KHTML,likeGecko)Chrome/79.0.3945.88Safari/535.11'}
    t = rq.get(url, headers=headers).content.decode('unicode_escape','ignore')#.encode('unicode_escape')
    n=t.find('(',0,30) #找到‘（
    t1=t.lstrip(t[0:n+1]).rstrip(t[-1])#去掉jsonp_1581733157953_6359285(，去掉最后的）
    t1
    j=json.loads(t1)
    print(j)
    i=0
    while i < 100:
        record.loc[k*100+i,'date']= date
        record.loc[k*100+i,'city']= j['data']['list'][i]['city_name']
        record.loc[k*100+i,'traffic']= j['data']['list'][i]['value']
        i+=1
    # record
  
    k+=1

record.to_csv('data/shanghai_in.csv',index=False)
# %% 第七步 下载流入上海数据
city_cn=['重庆市','温州市','北京市','信阳市','合肥市','杭州市','宁波市','蚌埠市','阜阳市','台州市','上饶市',
    '亳州市','南京市','商丘市','苏州市','安庆市','徐州市','周口市','六安市','淮安市','无锡市','临沂市','连云港市','嘉兴市','武汉市']
city_list=['cq','wz','bj','xy','hf','hz','nb','bb','fy','tz','sr','bz','nj','sq','sz','aq','xz','zk','la','ha','wx','ly','lyg','jx','wh']

i=0
while i < len(city_cn):
    temp_city=record[record['city']==city_cn[i]]

    sh_record[city_list[i]]=temp_city.reset_index(drop=True).traffic
    i+=1

# %% 第八步 去除NaN值，保存csv
# 按列显示nan的比例
all_data_na = (sh_record.isnull().sum() / len(sh_record)) * 100
all_data_na
#只有温州有nan，流量跌出前100
sh_record=sh_record.fillna(0.1)
sh_record.to_csv('data/shanghai-data-%s.csv' % ((datetime.datetime.now().date()-datetime.timedelta(days=1)).strftime('%Y%m%d') ),index=False)
# %% end==============================

# 	traffic
# city	
# 苏州市	9.386923
# 南通市	5.073462
# 盐城市	3.193077
# 嘉兴市	2.396923
# 阜阳市	2.299231
# 杭州市	2.103462
# 无锡市	1.924615
# 淮南市	1.892308
# 六安市	1.810000
# 合肥市	1.802692
# 泰州市	1.708462
# 宁波市	1.579615
# 芜湖市	1.435385
# 徐州市	1.434615
# 安庆市	1.406538
# 亳州市	1.380385
# 南京市	1.331538
# 扬州市	1.314231
# 重庆市	1.283077
# 周口市	1.256923
# 宿迁市	1.208846
# 淮安市	1.207308
# 宿州市	1.198846
# 常州市	1.154615
# 信阳市	1.144615
# 北京市	1.118462
# 蚌埠市	1.113077
# 宣城市	1.093077
# 商丘市	1.068077
# 绍兴市	1.037308

# =======对上海有影响城市，病例多，在前40个进入上海流量城市名单中
# 地区	确诊	进入上海流量排名（+1）
# 重庆	537	20
# 温州	499	36
# 北京	375	27
# 信阳	252	26
# 合肥	168	11
# 杭州	166	7
# 宁波	154	13
# 蚌埠	153	28
# 阜阳	150	6
# 台州	145	40
# 上饶	123	37
# 亳州	107	17
# 南京	91	18
# 商丘	90	30
# 苏州	86	2
# 安庆	82	16
# 徐州	77	15
# 周口	70	21
# 六安	63	10
# 淮安	60	23
# 无锡	52	8
# 临沂	47	33
# 连云港	46	39
# 嘉兴	43	5
