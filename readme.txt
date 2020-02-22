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
人口迁出迁入和各城市流入上海数据：https://qianxi.baidu.com/?from=shoubai#city=310000
现在病例数据直接来自：https://github.com/BlankerL/DXY-COVID-19-Data， 每天生成最新数据集前先git pull 一下将最新csv下载。

2. 数据说明：
totalPlus1d: 官方0-24小时段公布累计数量（第二天上午数据），如日期2月6日，281个累计确诊病例（是2月7日24时的数据），该行其它为2月6日数据。
#from_hubei:患者与湖北来客接触或去过湖北（后面不在更新，也没有使用）
sh_add:每日行政病例
#shangahi_in: 外地来上海流量，来自百度地图疫情地图
#wuhan_out:武汉外出流量
#wuhan_shanghai:武汉来上海流量
wh:武汉进入上海流量
wh_num:武汉累计病例
等等。

第一个动态：https://maimai.cn/article/detail?fid=1408838232&efid=EouU5Plywrcxb3F7yvHYig&from=single_feed

3. 尝试了h2o的AutoML和LSTM，发现h2o所健模型数据偏大(已删除)。
用sklearn automl比较好，近期上海每天增加2例左右。

4.组织各目标城市数据集前，可以在百度迁徙地图上查看前100流入目标城市的其它城市名称，及他们目前的病例数，相应地选择需要数据的城市，直接在代码中修改，如增加石家庄市就增加‘石家庄市’，'sjz','sjz-num',运行一下baidu&dxy_data-auto-add-only.py就可以形成所需的数据集。如果昨天的数据还没有在百度上更新，会报错。

5.版权：
自由使用。

