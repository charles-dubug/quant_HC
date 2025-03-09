# 导入原始概念板块列表
from index_names_list import index_names

# 将概念板块分类
concept_categories = {
    "科技与数字化": {
        "AI与计算": [
            "人工智能", "云计算", "大数据", "算力概念", "边缘计算", "数字经济", "数字孪生", "东数西算",
            "ChatGPT概念", "AIGC概念", "AI芯片", "AI智能体", "AI语料", "AI制药", "AI手机", "AI眼镜",
            "智谱AI", "Kimi概念", "DeepSeek概念", "Sora概念", "多模态AI", "MLOps概念", "空间计算",
            "时空大数据", "商汤概念", "人脑工程", "AIPC", "机器视觉", "国资云概念",
        ],
        
        "芯片与半导体": [
            "半导体概念", "国产芯片", "存储芯片", "汽车芯片", "第三代半导体", "第四代半导体", 
            "IGBT概念", "碳化硅", "氮化镓", "中芯概念", "华为海思", "英伟达概念", "EDA概念", "Chiplet概念",
            "高带宽内存",
        ],
        
        "通信与网络": [
            "5G概念", "6G概念", "光通信模块", "F5G概念", "WiFi", "VPN", "天基互联", "量子通信",
            "毫米波概念", "UWB概念", "RCS概念", "eSIM", "物联网", "IPv6", "铜缆高速连接", 
            "星闪概念",
        ],
        
        "数字应用": [
            "区块链", "数字货币", "元宇宙概念", "NFT概念", "Web3.0", "虚拟数字人", "虚拟现实", "增强现实",
            "混合现实", "数据安全", "网络安全", "数据确权", "数据要素", "数字水印", "数字哨兵", "全息技术",
            "知识产权", "数据中心", "EDR概念", "超清视频"
        ],

        "硬件设备": [
            "LED", "OLED", "MiniLED", "MicroLED", "光刻机(胶)", "柔性屏(折叠屏)", 
            "玻璃基板", "PCB", "MLCC", "被动元件", "传感器", "激光雷达", "无线耳机",
            "智能穿戴", "电子车牌", "3D摄像头", "屏下摄像", "裸眼3D", "3D玻璃",
            "电子纸概念", "无线充电", "液冷概念", "SPD概念", 
        ],
        
        "互联网平台": [
            "华为概念", "华为欧拉", "华为昇腾", "鸿蒙概念", "阿里概念", "百度概念", 
            "腾讯云", "抖音概念(字节概念)", "快手概念", "拼多多概念", "小红书概念",
            "小米概念", "苹果概念", "富士康", "荣耀概念",
        ],
        
        "企业信息化": [
            "ERP概念", "智能家居", "智能电视", "智慧城市", "智慧政务", "智慧灯杆",
            "智能机器", "财税数字化", "DRG/DIP", "PLC概念", "信创", "国产软件",
            "电子身份证",
        ],

        "数字娱乐": [
            "网络游戏", "电子竞技", "云游戏", "短剧互动游戏", "手游概念"
        ]
    },
    
    "新能源与环保": {
        "清洁能源": [
            "新能源", "风能", "太阳能", "氢能源", "核能核电", "生物质能发电", 
            "地热能", "绿色电力", "可控核聚变",
        ],
        
        "能源设备": [
            "光伏设备", "光伏建筑一体化", "光伏高速公路", "特高压", "TOPCon电池",
            "超导概念", "虚拟电厂", "空气能热泵", "超超临界发电", "智能电网"
        ],
        
        "储能技术": [
            "储能", "熔盐储能", "抽水蓄能", "锂电池", "钠离子电池", "固态电池", 
            "钒电池", "燃料电池", "超级电容", "钙钛矿电池", "BC电池",
        ],

        "新能源汽车": [
            "新能源车", "动力电池回收", "麒麟电池", "刀片电池", "HIT电池",
        ],
        
        "节能减排": [
            "节能环保", "碳交易", "碳基材料", "低碳冶金", "尾气治理",
        ],
        
        "环境保护": [
            "垃圾分类", "土壤修复", "海绵城市", "核污染防治", "降解塑料",
        ],
    },
    
    "医药与健康": {
        "医疗器械": [
            "医疗器械概念", "体外诊断", "血氧仪", "注射器概念", "抗菌面料"
        ],
        
        "医疗服务": [
            "医疗服务", "医疗美容", "互联医疗", "CRO", "医废处理",
            "养老概念", "辅助生殖", "毛发医疗",
        ],
        
        "创新医药": [
            "创新药", "精准医疗", "基因测序", "单抗概念", "CAR-T细胞疗法",
            "免疫治疗", "生物识别", "合成生物", "重组蛋白", "纳米银",
        ],
        
        "传统医药": [
            "中药概念", "化学制药", "生物疫苗", "生物制品",
            "熊去氧胆酸", "地塞米松", "蒙脱石散", "青蒿素", 
            "千金藤素", "维生素", "肝素概念", "独家药品"
        ],

        "公共卫生": [
            "健康中国", "新冠药物", "疫苗冷链", "口罩", "消毒剂",
            "病毒防治", "流感", "气溶胶检测", "痘病毒防治",
            "超级真菌", "幽门螺杆菌概念", "阿兹海默", "肝炎概念",
        ],
        
        "健康管理": [
            "减肥药", "长寿药",
        ],
    },
    
    "消费与服务": {
        "食品饮料": [
            "白酒", "啤酒概念", "调味品概念", "食品安全", "乳业", "预制菜概念", 
            "代糖概念", "娃哈哈概念",
        ],
        
        "生活消费": [
            "化妆品概念", "美容护理", "宠物经济", "婴童概念", "托育服务", 
            "电子烟", "超级品牌", "谷子经济",
        ],
        
        "零售渠道": [
            "新零售", "电商概念", "跨境电商", "社区团购", "免税概念", 
            "抖音小店", "快递概念", "冷链物流", "进口博览", "万达概念", "内贸流通",
            "地摊经济",
        ],
        
        "服务业态": [
            "在线旅游", "在线教育", "网红直播", "盲盒经济", "职业教育",
            "共享经济", "远程办公", "彩票概念", "赛马概念",
        ],
        
        "商业模式": [
            "C2M概念", "新型工业化", "数字阅读",
        ],

        "文化体育": [
            "体育产业", "世界杯", "中超概念", "冰雪经济", "户外露营", "影视概念",
        ],
    },
    
    "制造与工业": {
        "先进制造": [
            "工业4.0", "工业互联", "智能制造", "专精特新", "工业母机", 
            "3D打印", "新型工业化",
        ],
        
        "机器人": [
            "机器人概念", "人形机器人", "机器人执行器", "减速器", "同步磁阻电机",
        ],
        
        "汽车工业": [
            "汽车整车", "汽车零部件", "汽车服务", "特斯拉", "华为汽车", "小米汽车", 
            "汽车热管理", "汽车一体化压铸", "无人驾驶", "车联网(车路云)", 
            "电子后视镜", "胎压监测", "轮毂电机", "充电桩", "换电概念", 
            "高压快充", "飞行汽车(eVTOL)", "汽车拆解",
        ],
        
        "工业材料": [
            "新材料", "稀土永磁", "石墨烯", "碳纤维", "稀缺资源", "工业大麻",
            "小金属概念", "锂矿概念", "有机硅", "PEEK材料概念", "培育钻石", "民爆概念",
        ],

        "工业设备": [
            "工程机械概念", "工业气体", "氦气概念", "装配建筑", "通用航空",
            "磁悬浮概念", "海工装备", "无人机", "复合集流体", "植物照明",
        ],
        
        "建筑工业": [
            "建筑节能", "铁路基建", "装配建筑", 
        ],
    },
    
    "金融与投资": {
        "金融服务": [
            "银行", "证券", "保险", "多元金融", "互联金融", "券商概念", "移动支付",
            "京东金融", "蚂蚁概念", "参股银行", "参股券商", "跨境支付", "ETC",
            "参股保险", "参股期货",
        ],
        
        "投资主题": [
            "融资融券", "创投", "REITs概念", "独角兽", "股权激励", "并购重组概念",
            "股权转让", "举牌", "基金重仓", "社保重仓", "机构重仓", "高送转",
            "QFII重仓", "证金持股", "转债标的", "贬值受益",
        ],

        "指数成分": [
            "沪股通", "深股通", "富时罗素", "MSCI中国", "标准普尔",
        ],

        "交易市场": [
            "B股", "AB股", "AH股", "GDR", "北交所概念", 
            "科创板做市商", "科创板做市股", "参股新三板", "IPO受益",
        ],
        
        "金融创新": [
            "纾困概念", "养老金", "租售同权", "首发经济", "化债(AMC)概念",
        ],
    },
    
    "国家战略与区域发展": {
        "区域发展战略": [
            "一带一路", "京津冀", "雄安新区", "长江三角", "粤港自贸", "东北振兴", "西部大开发",
            "成渝特区", "深圳特区", "上海自贸", "湖北自贸", "滨海新区", "乡村振兴", "新型城镇化",
            "土地流转", "沪企改革",
        ],
        
        "国企与军工": [
            "国企改革", "央企改革", "中字头", "军工", "航天航空", "航母概念", "大飞机", "军民融合",
            "国家安防", "商业航天", "北斗导航", "空间站概念",
        ],

        "国际合作": [
            "东盟自贸区概念", "RCEP概念", "中特估", "中俄贸易概念",
        ],
        
        "发展规划": [
            "2025规划", "统一大市场", "供销社概念", "PPP模式",
        ],
    },
    
    "资源与能源": {
        "传统能源": [
            "石油行业", "天然气", "煤炭行业", "煤化工", "页岩气", "油气设服", "可燃冰",
            "发电机概念",
        ],
        
        "农业资源": [
            "农业种植", "农牧饲渔", "水产养殖", "粮食概念", "生态农业", "转基因", "草甘膦",
            "猪肉概念", "鸡肉概念", "人造肉",
        ],
        
        "矿产资源": [
            "有色金属", "贵金属", "黄金概念", "钛白粉", "磷化工", "氟化工",
        ],

        "化工材料": [
            "化工原料", "环氧丙烷", "PVDF概念", "蓝宝石",
        ],
        
        "水资源": [
            "水利建设", "净水概念", "海洋经济",
        ],
    },
    

    "基建与公共服务": {
        "基础设施": [
            "地下管网", "低空经济", "广电",
        ],
        
        "公共服务": [
            "噪声防治", "房屋检测", "退税商店", "CPO概念",
        ],
    },

    "其他特色主题": {
        "特色指数": [
            "上证50_", "上证180_", "上证380", "中证500", "HS300_", "深证100R", "深成500",
            "创业成份", "创业板综", "央视50_", "茅指数", "宁组合",
        ],
        
        "特殊状态": [
            "ST股", "次新股", "注册制次新股", "低价股", "百元股", "微盘股", "破净股",
            "昨日涨停", "昨日涨停_含一字", "昨日连板", "昨日连板_含一字", "昨日触板",
            "预盈预增", "预亏预减", "壳资源",
        ],
    },
}

# 创建反向映射，用于检查分类情况
all_categorized = []
category_mapping = {}  # 记录每个概念属于哪个大类和子类
duplicates = []  # 记录重复分类的概念

def get_concept_boards(category=None, subcategory=None, keep_structure=False):
    """
    提取概念板块作为字符串
    
    参数:
        category (str, optional): 大类名称，如"科技与数字化"。默认为None，表示所有大类。
        subcategory (str or list, optional): 小类名称，如"AI与计算"或["AI与计算", "芯片与半导体"]。默认为None，表示所有小类。
        keep_structure (bool, optional): 是否在输出中保留分类结构。默认为False。
        
    返回:
        str: 概念板块字符串，根据keep_structure决定是否包含分类信息
    """
    result = []
    
    # 将subcategory转换为列表格式
    subcategory_list = []
    if subcategory is not None:
        if isinstance(subcategory, list):
            subcategory_list = subcategory
        else:
            subcategory_list = [subcategory]
    
    # 如果指定了大类但未找到
    if category is not None and category not in concept_categories:
        return ''
    
    # 获取概念并组织输出
    if category is None:
        categories = concept_categories
    else:
        categories = {category: concept_categories[category]}
    
    structured_results = []
    for cat, subcats in categories.items():
        if isinstance(subcats, dict):
            for subcat, concepts in subcats.items():
                if not subcategory_list or subcat in subcategory_list:
                    if keep_structure:
                        structured_results.append(f"{subcat}：{','.join(concepts)}")
                    else:
                        result.extend(concepts)
        else:
            if keep_structure:
                structured_results.append(f"{cat}：{','.join(subcats)}")
            else:
                result.extend(subcats)
    
    # 返回结果
    if keep_structure:
        return '\n'.join(structured_results)
    else:
        return ','.join(result)

# 遍历所有分类
for category, subcategories in concept_categories.items():
    if isinstance(subcategories, dict):
        # 处理有子类的大类
        for subcategory, concepts in subcategories.items():
            for concept in concepts:
                all_categorized.append(concept)
                # 检查是否重复分类
                if concept in category_mapping:
                    duplicates.append({
                        "concept": concept,
                        "original": category_mapping[concept],
                        "duplicate": {"category": category, "subcategory": subcategory}
                    })
                else:
                    category_mapping[concept] = {
                        "category": category,
                        "subcategory": subcategory
                    }
    else:
        # 处理没有子类的大类
        for concept in subcategories:
            all_categorized.append(concept)
            if concept in category_mapping:
                duplicates.append({
                    "concept": concept,
                    "original": category_mapping[concept],
                    "duplicate": {"category": category, "subcategory": None}
                })
            else:
                category_mapping[concept] = {
                    "category": category,
                    "subcategory": None
                }

# 检查未分类的概念
uncategorized = [concept for concept in index_names if concept not in all_categorized]

# 打印分类结果
if __name__ == "__main__":
    # 打印分类统计
    print("\n=== 分类统计 ===")
    category_stats = {}
    for mapping in category_mapping.values():
        cat = mapping["category"]
        subcat = mapping["subcategory"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "subcategories": {}}
        category_stats[cat]["total"] += 1
        if subcat:
            if subcat not in category_stats[cat]["subcategories"]:
                category_stats[cat]["subcategories"][subcat] = 0
            category_stats[cat]["subcategories"][subcat] += 1

    for cat, stats in category_stats.items():
        print(f"\n{cat}（共{stats['total']}个概念）:")
        if stats["subcategories"]:
            for subcat, count in stats["subcategories"].items():
                print(f"  - {subcat}：{count}个")

    # 打印重复分类
    if duplicates:
        print("\n=== 重复分类 ===")
        for dup in duplicates:
            print(f"\n概念「{dup['concept']}」出现在多个分类中：")
            print(f"  - 原始分类：{dup['original']['category']}", end="")
            if dup['original']['subcategory']:
                print(f" / {dup['original']['subcategory']}")
            print(f"  - 重复分类：{dup['duplicate']['category']}", end="")
            if dup['duplicate']['subcategory']:
                print(f" / {dup['duplicate']['subcategory']}")

    # 打印未分类概念
    if uncategorized:
        print(f"\n=== 未分类概念（{len(uncategorized)}个）===")
        print(", ".join(uncategorized))

    # 打印总体统计
    print(f"\n=== 总体统计 ===")
    print(f"大类数量：{len(category_stats)}个")
    print(f"已分类概念：{len(category_mapping)}个")
    print(f"重复分类概念：{len(duplicates)}个")
    print(f"未分类概念：{len(uncategorized)}个")
    print(f"原始概念总数：{len(index_names)}个")