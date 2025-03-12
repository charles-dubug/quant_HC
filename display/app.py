import adata
import pandas as pd
import time
import os
from datetime import datetime

# 配置股票列表
STOCK_LIST = [
    '600961', '601020', '600367', '002155', '301368',
    '603198', '605208', '603993', '000596', '600711'
]

HTML_FILENAME = 'capital_flow_monitor.html'

def get_latest_data(stock_code):
    """获取单个股票的最新分钟级资金流数据"""
    try:
        df = adata.stock.market.get_capital_flow_min(stock_code=stock_code)
        return df.tail(1) if not df.empty else None
    except Exception as e:
        print(f"Error getting data for {stock_code}: {e}")
        return None

def generate_html(df):
    """生成带自动刷新和样式的HTML文件"""
    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>实时资金流监控</title>
        <meta http-equiv="refresh" content="60">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; }}
            table {{ border-collapse: collapse; width: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .timestamp {{ color: #666; margin-bottom: 1rem; }}
        </style>
    </head>
    <body>
        <h2>股票资金流实时监控</h2>
        <div class="timestamp">最后更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        {df.to_html(index=False)}
    </body>
    </html>
    '''
    with open(HTML_FILENAME, 'w') as f:
        f.write(html_template)

def main():
    while True:
        start_time = time.time()
        
        # 获取所有股票的最新数据
        all_data = []
        for stock in STOCK_LIST:
            if df := get_latest_data(stock):
                all_data.append(df)
        
        # 合并数据
        combined_df = pd.concat(all_data) if all_data else pd.DataFrame()
        
        # 清屏并显示终端输出
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"最新数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(combined_df.to_string(index=False) if not combined_df.empty else print("暂无数据"))
        
        # 生成HTML文件
        if not combined_df.empty:
            generate_html(combined_df)
            print(f"\n已更新网页文件：{HTML_FILENAME}")
        
        # 计算等待时间（精确到下一分钟）
        sleep_time = 60 - (time.time() - start_time) % 60
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()