该数据集来自个人收集，计划预测各城市未来一天新型肺炎确认人数。目前主要在上海和北京做尝试。希望大家集思广益，有一个好的模型，帮助资源提前规划。

说明：
1. 数据来自公开网络：
阜阳：http://wjw.fy.gov.cn/content/channel/5c35be0048787ad14d009195/
苏州：http://wsjkw.suzhou.gov.cn/szswjw/bdxw/nav_list.shtml
湖北：http://wjw.hubei.gov.cn/fbjd/dtyw/
广东：http://wsjkw.gd.gov.cn/zwyw_yqxx/index.html
温州：http://wjw.wenzhou.gov.cn/col/col1209919/index.html?uid=3978136&pageNum=3
北京：http://wjw.beijing.gov.cn/xwzx_20031/wnxw/
苏州： http://wjw.sz.gov.cn/yqxx/
上海： http://wsjkw.sh.gov.cn/index.html
人口迁出迁入数据：https://qianxi.baidu.com/?from=shoubai#city=310000

2. 数据说明：
totalPlus1d: 官方0-24小时段公布累计数量（第二天上午数据），如日期2月6日，281个累计确诊病例（是2月7日24时的数据），该行其它为2月6日数据。
from_hubei:患者与湖北来客接触或去过湖北（后面不在更新，也没有使用）
shangahi_in: 外地来上海流量，来自百度地图疫情地图
wuhan_out:武汉外出流量
wuhan_shanghai:武汉来上海流量
等等。

第一个动态：https://maimai.cn/article/detail?fid=1408838232&efid=EouU5Plywrcxb3F7yvHYig&from=single_feed

3. 尝试了h2o的AutoML和LSTM。

4. 问题：
预测值有时候和官方报道接近，有时候相差较大。
同样的数据、相同时间内、相同代码出来的模型重现性不高（设置了seed)。
北京的一个模型，比较吻合官方数据；上海两个模型，鲁棒性不高。
从百度和各卫健委爬数据未成功，这是完全的‘人工’智能。

5.版权：
自由使用。

