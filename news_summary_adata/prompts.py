news_cn_tech = """
# 指令
作为科技情报分析师，请搜索过去24小时，热点科技与数字化新闻，并提取完整信源信息

# 输出规范，请不要输出除JSON以外的其他内容！！！
```json
{
  "event_type": ["AI与计算", "芯片与半导体", "通信与网络", "数字应用", "硬件设备", "互联网平台", "企业信息化", "数字娱乐" (请从上述类别中选择一或多个类别)],
  "title": "新闻原标题",
  "source": "媒体名称",
  "publish_date": "YYYY-MM-DD",
  "event_abstract": "150字技术突破摘要，包含核心技术参数、产业链位置、商用时间表及市场空间"
}
"""

news_analysis = """
# 输出规范，请不要输出除JSON以外的其他内容！！！
```json
{
  "index_name": 板块名称，
  "prediction": 利好/利空
  "reason": 预测原因
}
"""

data_analysis_system = """
你是一个股票分析专家, 你需要根据一A股概念的近(1,5,10)日的资金流向, 分析该概念的未来走势, 输出预测结果。
以下为输入数据字段的注释：
index_code: 概念代码	
index_name: 概念名称	
days_type: 天数类型(1,5,10)
change_pct: 概念最近N日涨跌幅(%)
main_net_inflow: 主力资金净流入(元)
main_net_inflow_rate: 主力资金净流入占比(%)
max_net_inflow: 特大单净流入(元)
max_net_inflow_rate: 特大单净流入占比(%)
lg_net_inflow: 大单净流入(元)
lg_net_inflow_rate: 特大单净流入占比(%)
mid_net_inflow: 中单净流入(元)
mid_net_inflow_rate: 中单净流入占比(%)
sm_net_inflow: 小单净流入(元)
sm_net_inflow_rate: 小单净流入占比(%)
stock_code: 流入最大股代码
stock_name: 流入最大股名称
你将会得到三份以上的资金流向数据, 分别为1,5,10日的数据, 你需要根据这三份数据来分析该概念的未来走势。

# 输出规范, 请不要输出除JSON以外的其他内容！！！
```json
{
  "index_code": 板块代码,
  "index_name": 板块名称,
  "prediction": 数字1到10之间的整数值(不要加引号), 1表示预测未来走势不佳, 10表示预测未来走势极好, 
  "reason": 请在此处详细说明原因
}
"""
