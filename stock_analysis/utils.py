from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from prompts import news_cn_tech, news_analysis, concept_analysis_system, stock_analysis_system, order_analysis_system
from config import GEMINI_API_KEY, DEEPSEEK_API_KEY
from datetime import datetime, timedelta
import pytz
import json
import os
import adata
import concurrent.futures
import time
import signal
import pandas as pd
from datetime import datetime, timedelta
from pprint import pprint 

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# deepseek_client = OpenAI(api_key="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", base_url="https://openrouter.ai/api/v1")
 
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
        # temperature=0.6,
        # top_p=0.95,
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

def get_analysis_prompt(data, system_prompt):
    prompt = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": json.dumps(data, ensure_ascii=False, indent=2)}
        ]
    return prompt

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


def compute_stock_ratings(top_n=None):
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
    today = get_today_date()
    predictions_file = f'data/{today}/concept_predictions.json'
    stock_concept_map_file = r'data/stock_concept_map.json'
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
    if top_n is None:
        top_n = len(stock_ratings)
    top_stocks = sorted(stock_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    stock_codes = [stock[0] for stock in top_stocks]
    return stock_codes


def get_selected_stock_capital_flow(stock_codes, filter=None):
    date = get_date_ten_trading_days_ago()
    today = get_today_date()

    output_file = f'./data/{today}/stock_capital_flow.json'

    stocks_capital_flow = []
    filtered_stocks = []

    if filter is not None:
        total_stocks = filter
    else:
        total_stocks = len(stock_codes)

    completed = 0
    pbar = tqdm(total=total_stocks, desc="Processing stocks", unit="stock")

    for stock_code in stock_codes:
        df = adata.stock.market.get_capital_flow(stock_code=stock_code, start_date=date, end_date=today) 
        
        stock_capital_flow = {
                        "stock_code": stock_code,
                        "days_type_1": {
                            "main_net_inflow": 0,
                            "sm_net_inflow": 0,
                            "mid_net_inflow": 0,
                            "lg_net_inflow": 0,
                            "max_net_inflow": 0
                        },
                        "days_type_5": {
                            "main_net_inflow": 0,
                            "sm_net_inflow": 0,
                            "mid_net_inflow": 0,
                            "lg_net_inflow": 0,
                            "max_net_inflow": 0
                        },
                        "days_type_10": {
                            "main_net_inflow": 0,
                            "sm_net_inflow": 0,
                            "mid_net_inflow": 0,
                            "lg_net_inflow": 0,
                            "max_net_inflow": 0
                        }
                    }
        
        for days_type in [1, 5, 10]:
            period_data = df.head(days_type)

            # Calculate sum of each metric
            main_net_inflow = period_data['main_net_inflow'].astype(float).sum().round(1)
            sm_net_inflow = period_data['sm_net_inflow'].astype(float).sum().round(1)
            mid_net_inflow = period_data['mid_net_inflow'].astype(float).sum().round(1)
            lg_net_inflow = period_data['lg_net_inflow'].astype(float).sum().round(1)
            max_net_inflow = period_data['max_net_inflow'].astype(float).sum().round(1)
            
            # Update the dictionary
            stock_capital_flow[f"days_type_{days_type}"] = {
                "main_net_inflow": main_net_inflow,
                "sm_net_inflow": sm_net_inflow,
                "mid_net_inflow": mid_net_inflow,
                "lg_net_inflow": lg_net_inflow,
                "max_net_inflow": max_net_inflow
            }

        if filter is not None:
            if stock_capital_flow['days_type_10']['main_net_inflow'] < 10000000:
                continue
        
        filtered_stocks.append(stock_code)
        stocks_capital_flow.append(stock_capital_flow)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stocks_capital_flow, f, ensure_ascii=False, indent=2)
        
        completed += 1
        pbar.update(1)
        pbar.set_postfix({"completed": f"{completed}/{total_stocks}", "percent": f"{completed/total_stocks*100:.1f}%"})
        
        if completed >= total_stocks:
            break

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stocks_capital_flow, f, ensure_ascii=False, indent=2)
    
    pbar.close()

    return filtered_stocks
 

def process_concept(data):
    try:
        prompt = get_analysis_prompt(data, concept_analysis_system)

        # print(f"Analyzing {data['index_name']} (code: {data['index_code']})...")
        prediction = analysis_with_deepseek_robust(prompt)
        prediction['timestamp'] = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        return prediction
    except Exception as e:
        print(f"Error processing {data['index_name']}: {str(e)}")
        return {
            "index_code": data.get('index_code', ''),
            "index_name": data.get('index_name', ''),
            "prediction": "",
            "reason": f"处理失败: {str(e)}",
            "timestamp": datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        }

def concept_analysis(max_workers=3):
    today = get_today_date()
    file_path =  f'data/{today}/concept_capital_flow.json'

    with open(file_path, 'r', encoding='utf-8') as file:
        capital_flow_data = json.load(file)

    output_file = f'data/{today}/concept_predictions.json'

    predictions = []

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            print(f"Loaded {len(predictions)} existing predictions.")
        except json.JSONDecodeError:
            print("Error loading existing predictions file. Starting fresh.")

    data_to_process = []

    for data in capital_flow_data:
        if any(existing['index_code'] == data['index_code'] for existing in predictions):
            print(f"Skipping {data['index_name']} (code: {data['index_code']}) - already processed")
        else:
            data_to_process.append(data)

    print(f"Processing {len(data_to_process)} items in parallel...")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_data = {executor.submit(process_concept, data): data for data in data_to_process}
            
            # Create progress bar
            total_items = len(data_to_process)
            completed = 0
            pbar = tqdm(total=total_items, desc="Processing boards", unit="board")
            
            for future in concurrent.futures.as_completed(future_to_data):
                data = future_to_data[future]
                try:
                    prediction = future.result()
                    if prediction:
                        predictions.append(prediction)
                        completed += 1
                        pbar.update(1)
                        pbar.set_postfix({"completed": f"{completed}/{total_items}", "percent": f"{completed/total_items*100:.1f}%"})
                        
                        # Save after each successful prediction to avoid data loss
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(predictions, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Exception occurred while processing {data['index_name']}: {str(e)}")
            
            pbar.close()
    except KeyboardInterrupt:
        print("\nGracefully shutting down... Saving current progress...")
        # Final save of all predictions
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print("Progress saved. Program stopped.")
        exit(0)

    # Final save of all predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print("All processing complete!")


def process_stock(data):
    try:
        prompt = get_analysis_prompt(data, stock_analysis_system)
        # print(f"Analyzing {data['stock_code']}...")
        prediction = analysis_with_deepseek_robust(prompt)
        prediction['timestamp'] = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        return prediction
    except Exception as e:
        print(f"Error processing {data['stock_code']}: {str(e)}")
        return {
            "stock_code": data.get('stock_code', ''),
            "prediction": "",
            "reason": f"处理失败: {str(e)}",
            "timestamp": datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        }

def process_order(data):
    try:
        prompt = get_analysis_prompt(data, order_analysis_system)
        # print(f"Analyzing {data['stock_code']}...")
        prediction = analysis_with_deepseek_robust(prompt)
        prediction['timestamp'] = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        return prediction
    except Exception as e:
        print(f"Error processing {data['stock_code']}: {str(e)}")
        return {
            "stock_code": data.get('stock_code', ''),
            "prediction": "",
            "reason": f"处理失败: {str(e)}",
            "stop_loss": "",
            "take_profit": "",
            "timestamp": datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        }

def stock_analysis(max_workers=3, sorted=True):
    today = get_today_date()
    capital_flow_path = f'data/{today}/stock_capital_flow.json'
    k_line_path = f'data/{today}/stock_k_line.json'


    # Load both data sources
    with open(capital_flow_path, 'r', encoding='utf-8') as file:
        capital_flow_data = json.load(file)
    
    with open(k_line_path, 'r', encoding='utf-8') as file:
        k_line_data = json.load(file)
    
    # Create a dictionary for faster k-line data lookup
    k_line_dict = {item['stock_code']: item['k-line'] for item in k_line_data}

    output_file = f'data/{today}/stock_predictions.json'
    output_file_sorted = f'data/{today}/stock_predictions_sorted.json'
    predictions = []

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            print(f"Loaded {len(predictions)} existing predictions.")
        except json.JSONDecodeError:
            print("Error loading existing predictions file. Starting fresh.")

    # Merge data and prepare for processing
    data_to_process = []
    for flow_data in capital_flow_data:
        stock_code = flow_data['stock_code']
        
        # Skip if already processed
        if any(existing['stock_code'] == stock_code for existing in predictions):
            print(f"Skipping {stock_code} - already processed")
            continue
        
        # Get k-line data for this stock
        k_line = k_line_dict.get(stock_code)
        if k_line is None:
            print(f"Warning: No k-line data found for {stock_code}")
            continue
        
        # Merge the data
        merged_data = {
            "stock_code": stock_code,
            "capital_flow": flow_data,
            "k_line": k_line
        }
        data_to_process.append(merged_data)

    print(f"Processing {len(data_to_process)} items in parallel...")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_data = {executor.submit(process_stock, data): data for data in data_to_process}
            total_items = len(data_to_process)
            completed = 0
            pbar = tqdm(total=total_items, desc="Processing stocks", unit="stock")
            
            for future in concurrent.futures.as_completed(future_to_data):
                data = future_to_data[future]
                try:
                    prediction = future.result()
                    if prediction:
                        predictions.append(prediction)
                        completed += 1
                        pbar.update(1)
                        pbar.set_postfix({"completed": f"{completed}/{total_items}", 
                                        "percent": f"{completed/total_items*100:.1f}%"})
                        
                        # Save after each successful prediction
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(predictions, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Exception occurred while processing {data['stock_code']}: {str(e)}")
            
            pbar.close()
    except KeyboardInterrupt:
        print("\nGracefully shutting down... Saving current progress...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print("Progress saved. Program stopped.")
        exit(0)

    # Final save of all predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    if sorted:
        sort_predictions_by_score(output_file, output_file_sorted)

    print("All processing complete!")

def get_selected_stock_k(stock_codes):
    date = get_date_ten_trading_days_ago()
    today = get_today_date()

    output_file = f'./data/{today}/stock_k_line.json'

    stocks_k = []

    total_stocks = len(stock_codes)
    completed = 0
    pbar = tqdm(total=total_stocks, desc="Processing stocks", unit="stock")

    for stock_code in stock_codes:
        df = adata.stock.market.get_market(stock_code=stock_code, start_date=date, end_date=today, k_type=1, adjust_type=1)

        stock_k = {
            "stock_code": stock_code,
            "k-line": []
        }

        for _, row in df.iloc[::-1].iterrows():
            k_line_item = {
                "trade_date": row['trade_date'],
                "open": float(row['open']),
                "close": float(row['close']),
                "high": float(row['high']),
                "low": float(row['low']),
                "volume": float(row['volume']),
                "amount": float(row['amount']),
                "change": float(row['change']),
                "change_pct": float(row['change_pct']),
                "turnover_ratio": float(row['turnover_ratio']),
                "pre_close": float(row['pre_close'])
            }

            stock_k["k-line"].append(k_line_item)

        stocks_k.append(stock_k)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stocks_k, f, ensure_ascii=False, indent=2)
        
        completed += 1
        pbar.update(1)
        pbar.set_postfix({"completed": f"{completed}/{total_stocks}", "percent": f"{completed/total_stocks*100:.1f}%"})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stocks_k, f, ensure_ascii=False, indent=2)

    pbar.close()

def get_concept_capital_flow(): 
    today = get_today_date()
    os.makedirs(f'./data/{today}', exist_ok=True)
    output_file = f'./data/{today}/concept_capital_flow.json'

    df_1day = adata.stock.market.all_capital_flow_east(days_type=1)
    print("1-day data loaded.")
    df_5day = adata.stock.market.all_capital_flow_east(days_type=5)
    print("5-day data loaded.")
    df_10day = adata.stock.market.all_capital_flow_east(days_type=10)
    print("10-day data loaded.")

    # Create a dictionary to store the restructured data
    restructured_data = {}

    # Process 1-day data
    for _, row in df_1day.iterrows():
        index_code = row['index_code']
        if index_code not in restructured_data:
            restructured_data[index_code] = {
                'index_code': index_code,
                'index_name': row['index_name'],
                'days_type_1': {},
                'days_type_5': {},
                'days_type_10': {}
            }
        
        # Add all columns except index_code and index_name to days_type_1
        days_type_1_data = {col: row[col] for col in row.index if col not in ['index_code', 'index_name']}
        restructured_data[index_code]['days_type_1'] = days_type_1_data

    # Process 5-day data
    for _, row in df_5day.iterrows():
        index_code = row['index_code']
        if index_code not in restructured_data:
            restructured_data[index_code] = {
                'index_code': index_code,
                'index_name': row['index_name'],
                'days_type_1': {},
                'days_type_5': {},
                'days_type_10': {}
            }
        
        # Add all columns except index_code and index_name to days_type_5
        days_type_5_data = {col: row[col] for col in row.index if col not in ['index_code', 'index_name']}
        restructured_data[index_code]['days_type_5'] = days_type_5_data

    # Process 10-day data
    for _, row in df_10day.iterrows():
        index_code = row['index_code']
        if index_code not in restructured_data:
            restructured_data[index_code] = {
                'index_code': index_code,
                'index_name': row['index_name'],
                'days_type_1': {},
                'days_type_5': {},
                'days_type_10': {}
            }
        
        # Add all columns except index_code and index_name to days_type_10
        days_type_10_data = {col: row[col] for col in row.index if col not in ['index_code', 'index_name']}
        restructured_data[index_code]['days_type_10'] = days_type_10_data

    # Convert the dictionary to a list for JSON serialization
    restructured_list = list(restructured_data.values())

    # Save to JSON file with proper encoding for Chinese characters
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(restructured_list, f, ensure_ascii=False, indent=2)

    print(f"Restructured data saved to ./data/{today}/concept_capital_flow.json with {len(restructured_list)} records")

def order_analysis(max_workers=3, top_n=50):
    today = get_today_date()
    stock_sorted = f'./data/{today}/stock_predictions_sorted.json' 
    with open(stock_sorted, 'r', encoding='utf-8') as f:
        stock_analysis = json.load(f)
    
    # Get the top N stocks from the sorted predictions
    top_stocks = stock_analysis[:top_n]
    top_stock_codes = [stock['stock_code'] for stock in top_stocks]
    
    # Load capital flow data
    capital_flow_path = f'./data/{today}/stock_capital_flow.json'
    with open(capital_flow_path, 'r', encoding='utf-8') as f:
        capital_flow_data = json.load(f)
    
    # Load k-line data
    k_line_path = f'./data/{today}/stock_k_line.json'
    with open(k_line_path, 'r', encoding='utf-8') as f:
        k_line_data = json.load(f)
    
    # Create dictionaries for faster lookup
    capital_flow_dict = {item['stock_code']: item for item in capital_flow_data}
    k_line_dict = {item['stock_code']: item['k-line'] for item in k_line_data}
    stock_info_dict = {item['stock_code']: item for item in stock_analysis}
    
    # Prepare data for further analysis
    data_to_process = []
    for stock_code in top_stock_codes:
        # Get stock info from original analysis
        stock_info = stock_info_dict.get(stock_code)
        
        # Get capital flow and k-line data
        capital_flow = capital_flow_dict.get(stock_code)
        k_line = k_line_dict.get(stock_code)

        capital_flow_data = {
            "days_type_1": capital_flow.get('days_type_1'),
            "days_type_5": capital_flow.get('days_type_5'),
            "days_type_10": capital_flow.get('days_type_10')
        }
        if not capital_flow or not k_line or not stock_info:
            print(f"Warning: Missing data for stock {stock_code}")
            continue
            
        # Create combined data with all information
        combined_data = {
            "stock_code": stock_code,
            "prediction": stock_info.get('prediction'),
            "reason": stock_info.get('reason', ''),
            "capital_flow": capital_flow_data,
            "k_line": k_line
        }
        data_to_process.append(combined_data)
    
    print(f"Processing {len(data_to_process)} top stocks for further analysis...")
    
    # Process the data in parallel, similar to stock_analysis function
    output_file = f'data/{today}/top_{top_n}_stocks_analysis.json'
    predictions = []
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_data = {executor.submit(process_order, data): data for data in data_to_process}
            total_items = len(data_to_process)
            completed = 0
            pbar = tqdm(total=total_items, desc="Processing top stocks", unit="stock")
            
            for future in concurrent.futures.as_completed(future_to_data):
                data = future_to_data[future]
                try:
                    prediction = future.result()
                    if prediction:
                        
                        predictions.append(prediction)
                        completed += 1
                        pbar.update(1)
                        pbar.set_postfix({"completed": f"{completed}/{total_items}", 
                                        "percent": f"{completed/total_items*100:.1f}%"})
                        
                        # Save after each successful prediction
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(predictions, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Exception occurred while processing {data['stock_code']}: {str(e)}")
            
            pbar.close()
    except KeyboardInterrupt:
        print("\nGracefully shutting down... Saving current progress...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print("Progress saved. Program stopped.")
        exit(0)

    # Final save of all predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print("Top stocks analysis complete!")

def combined_analysis(stocks):
    date = get_date_ten_trading_days_ago()
    today = get_today_date()
    for stock in stocks:
        capital_flow_df = adata.stock.market.get_capital_flow(stock_code=stock, start_date=date, end_date=today)
        stock_capital_flow = {
                        "days_type_1": {
                            "main_net_inflow": 0,
                            "sm_net_inflow": 0,
                            "mid_net_inflow": 0,
                            "lg_net_inflow": 0,
                            "max_net_inflow": 0
                        },
                        "days_type_5": {
                            "main_net_inflow": 0,
                            "sm_net_inflow": 0,
                            "mid_net_inflow": 0,
                            "lg_net_inflow": 0,
                            "max_net_inflow": 0
                        },
                        "days_type_10": {
                            "main_net_inflow": 0,
                            "sm_net_inflow": 0,
                            "mid_net_inflow": 0,
                            "lg_net_inflow": 0,
                            "max_net_inflow": 0
                        }
                    }
        
        for days_type in [1, 5, 10]:
            period_data = capital_flow_df.head(days_type)

            # Calculate sum of each metric
            main_net_inflow = float(period_data['main_net_inflow'].astype(float).sum().round(1))
            sm_net_inflow = float(period_data['sm_net_inflow'].astype(float).sum().round(1))
            mid_net_inflow = float(period_data['mid_net_inflow'].astype(float).sum().round(1))
            lg_net_inflow = float(period_data['lg_net_inflow'].astype(float).sum().round(1))
            max_net_inflow = float(period_data['max_net_inflow'].astype(float).sum().round(1))
            
            # Update the dictionary
            stock_capital_flow[f"days_type_{days_type}"] = {
                "main_net_inflow": main_net_inflow,
                "sm_net_inflow": sm_net_inflow,
                "mid_net_inflow": mid_net_inflow,
                "lg_net_inflow": lg_net_inflow,
                "max_net_inflow": max_net_inflow
            }

        k_line_df = adata.stock.market.get_market(stock_code=stock, start_date=date, end_date=today, k_type=1, adjust_type=1)
        stock_k = []

        for _, row in k_line_df.iloc[::-1].iterrows():
            k_line_item = {
                "trade_date": row['trade_date'],
                "open": float(row['open']),
                "close": float(row['close']),
                "high": float(row['high']),
                "low": float(row['low']),
                "volume": float(row['volume']),
                "amount": float(row['amount']),
                "change": float(row['change']),
                "change_pct": float(row['change_pct']),
                "turnover_ratio": float(row['turnover_ratio']),
                "pre_close": float(row['pre_close'])
            }

            stock_k.append(k_line_item)
        
        merged_data = {
            "stock_code": stock,
            "capital_flow": stock_capital_flow,
            "k-line": stock_k,
        }

        prompt_1 = get_analysis_prompt(merged_data, stock_analysis_system)

        response = analysis_with_deepseek_robust(prompt_1)

        merged_data['prediction']= response.get('prediction')
        merged_data['reason']= response.get('reason')

        prompt_2 = get_analysis_prompt(merged_data, order_analysis_system)

        response = analysis_with_deepseek_robust(prompt_2)

        pprint(response)
        print("\n-------------------------------------------------------\n")
             
        