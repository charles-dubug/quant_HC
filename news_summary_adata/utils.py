from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from prompts import news_cn_tech, news_analysis, data_analysis_system
from datetime import datetime, timedelta
import pytz
import json
import os
import adata
from datetime import datetime, timedelta

gemini_client = genai.Client(api_key="AIzaSyCLrO8jQVBo_c6XarrmVSBhiZr0oLPjM94")
deepseek_client = OpenAI(api_key="sk-08d04f54a7a6416c927546befd18ce51", base_url="https://api.deepseek.com")
 
model_id = "gemini-2.0-flash"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

def extract_json(text):
    """
    从API响应中提取JSON数据
    """
    # 提取JSON部分
    if "```json" in text and "```" in text.split("```json", 1)[1]:
        json_content = text.split("```json", 1)[1].split("```", 1)[0].strip()
        try:
            data = json.loads(json_content)
            return data
        except json.JSONDecodeError:
            raise ValueError("JSON解析错误")
    # 如果没有json标记但看起来是json内容
    elif text.strip().startswith("[") and text.strip().endswith("]"):
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError:
            raise ValueError("JSON解析错误")
    
    # 如果没有找到有效的JSON内容
    raise ValueError("未找到有效的JSON内容")

def clean_response(response):
    for each in response.candidates[0].content.parts:
        text = each.text
        # 提取JSON部分
        if "```json" in text and "```" in text.split("```json", 1)[1]:
            json_content = text.split("```json", 1)[1].split("```", 1)[0].strip()
            try:
                news_data = json.loads(json_content)
                # 获取当前日期
                today = datetime.now()
                yesterday = today - timedelta(days=1)
                today_str = today.strftime("%Y-%m-%d")
                yesterday_str = yesterday.strftime("%Y-%m-%d")
                
                # 过滤近两日的新闻
                recent_news = [news for news in news_data if news.get("publish_date") in [today_str, yesterday_str]]
                
                # 输出过滤后的新闻
                print(json.dumps(recent_news, ensure_ascii=False, indent=2))
                return(json.dumps(recent_news, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                raise ValueError("JSON解析错误")
        # 如果没有json标记但看起来是json内容
        elif text.strip().startswith("[") and text.strip().endswith("]"):
            try:
                news_data = json.loads(text)
                # 获取当前日期
                today = datetime.now()
                yesterday = today - timedelta(days=1)
                today_str = today.strftime("%Y-%m-%d")
                yesterday_str = yesterday.strftime("%Y-%m-%d")
                
                # 过滤近两日的新闻
                recent_news = [news for news in news_data if news.get("publish_date") in [today_str, yesterday_str]]
                
                # 输出过滤后的新闻
                print(json.dumps(recent_news, ensure_ascii=False, indent=2))
                return(json.dumps(recent_news, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                raise ValueError("JSON解析错误")

def single_search(prompt):
    today = datetime.now().strftime("%Y年%m月%d日")
    response = gemini_client.models.generate_content(
        model=model_id,
        contents=f"今天是{today}\n{prompt}",
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    )
    return clean_response(response)

def web_search(query):
    match query:
        case "科技与数字化":
            prompt = news_cn_tech
        case "新能源与环保":
            prompt = news_cn_energy
        case "医药与健康":
            prompt = news_cn_health
        case "消费与服务":
            prompt = news_cn_consumption
        case "制造与工业":
            prompt = news_cn_manufacturing
        case "金融与投资":
            prompt = news_cn_finance
        case "国家战略与区域发展":
            prompt = news_cn_strategy
        case "资源与能源":
            prompt = news_cn_resource
        case "基建与公共服务":
            prompt = news_cn_infrastructure
        case _:
            raise ValueError("无效的查询类别")
    
    news = ""

    run_times = 0
    while run_times < 3:
        news += single_search(prompt)
        news += ("\n")
        run_times += 1
    
    response = gemini_client.models.generate_content(
        model=model_id,
        contents=f"请过滤并删除重复新闻，保留原有JSON格式\n{news}",
    )
    
    return clean_response(response)

def extract_event_types(news_json_str):
    """
    从新闻JSON字符串中提取所有不重复的event_type
    """
    news_data = json.loads(news_json_str)
    event_types = set()
    for item in news_data:
        if "event_type" in item and isinstance(item["event_type"], list):
            event_types.update(item["event_type"])
    return sorted(list(event_types))

def analysis_with_deepseek(prompt):
    """
    使用DeepSeek API分析数据
    """
    response = deepseek_client.chat.completions.create(
        model="deepseek-reasoner",
        messages=prompt,
        stream=False
    )
    return extract_json(response.choices[0].message.content)

def analysis_with_deepseek_robust(prompt):
    try:
        response = analysis_with_deepseek(prompt)
        return response
    except Exception as e:
        print(f"DeepSeek API调用失败: {e}")
        print(f"正在重试...")
        return analysis_with_deepseek_robust(prompt)

def get_data_analysis_prompt(data):
    prompt = [
          {"role": "system", "content": data_analysis_system},
          {"role": "user", "content": json.dumps(data, ensure_ascii=False, indent=2)}
        ]
    return prompt

def filter_high_potential_sectors(data):
    """
    基于多周期资金流数据的板块筛选函数
    新增功能：大额资金流三重维度风险分析
    """
    qualified_sectors = []
    
    for sector in data:
        print(sector['index_name'])
        # 获取周期数据；如果缺失，则跳过该板块
        d1 = sector.get('days_type_1')
        d5 = sector.get('days_type_5')
        d10 = sector.get('days_type_10')
        if not (d1 and d5 and d10):
            print("数据缺失，跳过该板块")
            continue

        # 检查必备的字段是否存在，若缺失则跳过
        main_net_inflow_d10 = d10.get('main_net_inflow')
        main_net_inflow_d5 = d5.get('main_net_inflow')
        flow_1 = d1.get('main_net_inflow')
        if main_net_inflow_d10 is None or main_net_inflow_d5 is None or flow_1 is None:
            print("数据缺失，跳过该板块")
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
            print("涨跌幅数据缺失，跳过该板块")
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
            print("主力流入率数据缺失，跳过该板块")
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
            print("符合条件")
            qualified_sectors.append(sector_data)
        else:
            print("不符合条件")
    
    return qualified_sectors

def sort_predictions_by_score(input_file, output_file):
    """
    Sort predictions.json by prediction score from highest to lowest
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the sorted JSON file
    """
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Convert prediction values to numeric if they are strings
    for item in predictions:
        if isinstance(item['prediction'], str) and item['prediction'].isdigit():
            item['prediction'] = int(item['prediction'])
    
    # Sort predictions by prediction score (highest to lowest)
    sorted_predictions = sorted(predictions, key=lambda x: x['prediction'], reverse=True)
    
    # Save the sorted predictions to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_predictions, f, ensure_ascii=False, indent=2)
    
    print(f"Sorted {len(sorted_predictions)} predictions by score (highest to lowest)")
    print(f"Saved to: {output_file}")

def get_date_ten_trading_days_ago():
    """
    Returns the date of ten trading days (excluding weekends) before the current date
    in 'YYYY-MM-DD' format, using China Standard Time (CST).
    
    Returns:
        str: Date string in 'YYYY-MM-DD' format
    """
    # Get current time in China Standard Time
    china_tz = pytz.timezone('Asia/Shanghai')
    current_time_china = datetime.now(china_tz)
    
    # Initialize counter for trading days
    trading_days_count = 0
    days_to_go_back = 0
    
    # Go back until we find 10 trading days
    while trading_days_count < 10:
        days_to_go_back += 1
        check_date = current_time_china - timedelta(days=days_to_go_back)
        
        # Check if it's a weekday (0 = Monday, 6 = Sunday)
        if check_date.weekday() < 5:  # 0-4 are weekdays
            trading_days_count += 1
    
    # Calculate the target date
    target_date = current_time_china - timedelta(days=days_to_go_back)
    
    # Format the date as YYYY-MM-DD
    formatted_date = target_date.strftime('%Y-%m-%d')
    
    return formatted_date

def get_today_date():
    """
    Returns today's date in 'YYYY-MM-DD' format using China Standard Time (CST).
    
    Returns:
        str: Today's date string in 'YYYY-MM-DD' format
    """
    # Get current time in China Standard Time
    china_tz = pytz.timezone('Asia/Shanghai')
    current_time_china = datetime.now(china_tz)
    
    # Format the date as YYYY-MM-DD
    formatted_date = current_time_china.strftime('%Y-%m-%d')
    
    return formatted_date

# Convert DataFrame to the desired JSON structure
def stock_capital_flow_df_to_json(dataframe):
    """
    Convert DataFrame to a specific JSON structure
    """
    if dataframe.empty:
        return {"stock_code": "", "capital_flow": []}
    
    # Get the stock code from the first row
    stock_code = dataframe['stock_code'].iloc[0]
    
    # Convert each row to a dictionary and build the data list
    data_list = []
    for _, row in dataframe.iterrows():
        data_item = {
            "trade_date": row['trade_date'],
            "main_net_inflow": float(row['main_net_inflow']),
            "sm_net_inflow": float(row['sm_net_inflow']),
            "mid_net_inflow": float(row['mid_net_inflow']),
            "lg_net_inflow": float(row['lg_net_inflow']),
            "max_net_inflow": float(row['max_net_inflow'])
        }
        data_list.append(data_item)
    
    # Create the final JSON structure
    result = {
        "stock_code": stock_code,
        "capital_flow": data_list
    }
    
    return result

def compute_stock_ratings(predictions_file, stock_concept_map_file, top_n=500):
    """
    Compute stock ratings based on the average rating of their related concepts.
    
    Args:
        predictions_file (str): Path to the predictions.json file
        stock_concept_map_file (str): Path to the stock_concept_map.json file
        top_n (int): Number of top stocks to return
        
    Returns:
        list: List of tuples (stock_code, rating) for the top N stocks
    """
    # Load the predictions data
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    
    # Create a dictionary of concept ratings
    concept_ratings = {}
    for item in predictions_data:
        concept_code = item['index_code']
        rating = item['prediction']
        concept_ratings[concept_code] = rating
    
    # Load the stock concept map
    with open(stock_concept_map_file, 'r', encoding='utf-8') as f:
        stock_concept_map = json.load(f)
    
    # Compute the rating for each stock
    stock_ratings = {}
    for stock_code, concepts in stock_concept_map.items():
        if not concepts:  # Skip stocks with no concepts
            continue
        
        # Calculate the average rating of the stock's concepts
        total_rating = 0
        valid_concepts = 0
        
        for concept in concepts:
            if concept in concept_ratings:
                total_rating += concept_ratings[concept]
                valid_concepts += 1
        
        # Only calculate average if there are valid concepts
        if valid_concepts > 0:
            avg_rating = total_rating / valid_concepts
            stock_ratings[stock_code] = avg_rating
    
    # Sort stocks by rating in descending order and get the top N
    top_stocks = sorted(stock_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return top_stocks