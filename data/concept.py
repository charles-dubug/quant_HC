def filter_high_potential_sectors(data):
    """
    基于多周期资金流数据的板块筛选函数
    新增功能：大额资金流三重维度风险分析
    """
    qualified_sectors = []
    
    for sector in data:
        # 获取周期数据；如果缺失，则跳过该板块
        d1 = sector.get('days_type_1')
        d5 = sector.get('days_type_5')
        d10 = sector.get('days_type_10')
        if not (d1 and d5 and d10):
            continue

        # 检查必备的字段是否存在，若缺失则跳过
        main_net_inflow_d10 = d10.get('main_net_inflow')
        main_net_inflow_d5 = d5.get('main_net_inflow')
        flow_1 = d1.get('main_net_inflow')
        if main_net_inflow_d10 is None or main_net_inflow_d5 is None or flow_1 is None:
            continue

        # 计算关键指标
        daily_10 = main_net_inflow_d10 / 10 if main_net_inflow_d10 else 0
        daily_5 = main_net_inflow_d5 / 5 if main_net_inflow_d5 else 0
        
        # 大额资金分析预处理（检查数据是否存在）
        lg_net_inflow_d10 = d10.get('lg_net_inflow')
        lg_net_inflow_d5 = d5.get('lg_net_inflow')
        lg_flow_1 = d1.get('lg_net_inflow')
        if lg_net_inflow_d10 is None or lg_net_inflow_d5 is None or lg_flow_1 is None:
            continue
        lg_daily_10 = lg_net_inflow_d10 / 10 if lg_net_inflow_d10 else 0
        lg_daily_5 = lg_net_inflow_d5 / 5 if lg_net_inflow_d5 else 0

        # 核心条件判断
        condition_continuity = (daily_10 <= daily_5 * 1.1) and (daily_5 <= flow_1 * 1.1)
        # 避免除零错误，若daily_10为0，则用1e-6代替
        strength_ratio = (flow_1 / (daily_10 + 1e-6)) * 100  
        condition_strength = strength_ratio > 150

        # 获取涨跌幅数据
        d1_change_pct = d1.get('change_pct')
        d5_change_pct = d5.get('change_pct')
        d10_change_pct = d10.get('change_pct')
        if d1_change_pct is None or d5_change_pct is None or d10_change_pct is None:
            continue

        # 量价健康度验证（注意：这里调整了顺序以匹配示例数据）
        price_condition = (d1_change_pct > 0 and d10_change_pct > d5_change_pct and d1_change_pct < 7)

        # 新增大额资金风险指标
        lg_strength_ratio = round((lg_flow_1 / (lg_daily_10 + 1e-6) * 100), 1)
        lg_acceleration = lg_daily_5 > lg_daily_10
        lg_price_divergence = (lg_flow_1 > 0) and (d1_change_pct < 1)
        lg_risk_indicators = {
            "lg_strength_ratio": lg_strength_ratio,
            "lg_acceleration": lg_acceleration,
            "lg_price_divergence": lg_price_divergence
        }

        # 主力资金验证（检查必要的流入率字段）
        d1_main_net_inflow_rate = d1.get('main_net_inflow_rate')
        d5_main_net_inflow_rate = d5.get('main_net_inflow_rate')
        d10_main_net_inflow_rate = d10.get('main_net_inflow_rate')
        if d1_main_net_inflow_rate is None or d5_main_net_inflow_rate is None or d10_main_net_inflow_rate is None:
            continue
        capital_condition = (d1_main_net_inflow_rate > 0.8 and
                             d5_main_net_inflow_rate > 0.3 and 
                             d10_main_net_inflow_rate > -0.1)
        
        if all([condition_continuity, condition_strength, price_condition, capital_condition]):
            sector_data = {
                "index_code": sector.get('index_code'),
                "index_name": sector.get('index_name'),
                "capital_flow": {
                    "strength_ratio": round(strength_ratio, 1),
                    "trend_type": "阶梯式流入" if strength_ratio < 300 else "短期过热",
                    "daily_sequence": [
                        {"days": 10, "net_inflow": d10.get('main_net_inflow'), "change_pct": d10.get('change_pct')},
                        {"days": 5, "net_inflow": d5.get('main_net_inflow'), "change_pct": d5.get('change_pct')},
                        {"days": 1, "net_inflow": d1.get('main_net_inflow'), "change_pct": d1.get('change_pct')}
                    ]
                },
                "risk_indicators": {
                    "small_capital_outflow": d1.get('sm_net_inflow', 0) < 0,
                    "mid_capital_behavior": "跟风" if d1.get('mid_net_inflow_rate', 0) > 0 else "撤离",
                    "large_capital_analysis": lg_risk_indicators
                }
            }
            qualified_sectors.append(sector_data)
    
    return qualified_sectors


# 假设您的原始数据格式如下
sample_data = [
    {
        "index_code": "BK1015",
        "index_name": "能源金属",
        "days_type_1": {
            "change_pct": 3.6,
            "main_net_inflow": 930234736.0,
            "main_net_inflow_rate": 5.61,
            "max_net_inflow": 529605232.0,
            "max_net_inflow_rate": 3.19,
            "lg_net_inflow": 400629504.0,
            "lg_net_inflow_rate": 2.42,
            "mid_net_inflow": -245510912.0,
            "mid_net_inflow_rate": -1.48,
            "sm_net_inflow": -684723760.0,
            "sm_net_inflow_rate": -4.13,
            "stock_code": "华友钴业",
            "stock_name": "603799"
        },
        "days_type_5": {
            "change_pct": 5.96,
            "main_net_inflow": 1550043536.0,
            "main_net_inflow_rate": 2.59,
            "max_net_inflow": 967757040.0,
            "max_net_inflow_rate": 1.62,
            "lg_net_inflow": 582286496.0,
            "lg_net_inflow_rate": 0.97,
            "mid_net_inflow": -31947936.0,
            "mid_net_inflow_rate": -0.05,
            "sm_net_inflow": -1518095536.0,
            "sm_net_inflow_rate": -2.54,
            "stock_code": "华友钴业",
            "stock_name": "603799"
        },
        "days_type_10": {
            "change_pct": 6.33,
            "main_net_inflow": 839000816.0,
            "main_net_inflow_rate": 0.93,
            "max_net_inflow": 513060016.0,
            "max_net_inflow_rate": 0.57,
            "lg_net_inflow": 325940800.0,
            "lg_net_inflow_rate": 0.36,
            "mid_net_inflow": 40740320.0,
            "mid_net_inflow_rate": 0.05,
            "sm_net_inflow": -879741072.0,
            "sm_net_inflow_rate": -0.97,
            "stock_code": "华友钴业",
            "stock_name": "603799"
        }
    },
]

# 执行筛选
result = filter_high_potential_sectors(sample_data)
print(result)
