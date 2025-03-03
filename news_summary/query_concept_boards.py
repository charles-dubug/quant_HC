import sqlite3
import os
import pandas as pd

def connect_db():
    db_path = os.path.join(os.path.dirname(__file__), 'concept_boards.db')
    return sqlite3.connect(db_path)

def get_top_boards(limit=10, by_change=True):
    """Get top performing concept boards"""
    conn = connect_db()
    
    order_by = "change_percent DESC" if by_change else "market_value DESC"
    
    query = f"""
    SELECT 
        cb.board_name, 
        bd.rank, 
        bd.price, 
        bd.change_percent,
        bd.market_value / 100000000 as market_value_billion,
        c1.name as category,
        c2.name as subcategory
    FROM board_data bd
    JOIN concept_boards cb ON bd.board_code = cb.board_code
    LEFT JOIN categories c1 ON cb.category_id = c1.id
    LEFT JOIN categories c2 ON cb.subcategory_id = c2.id
    ORDER BY {order_by}
    LIMIT {limit}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_boards_by_category(category_name):
    """Get all boards in a specific category"""
    conn = connect_db()
    
    query = """
    SELECT 
        cb.board_name, 
        bd.rank, 
        bd.price, 
        bd.change_percent,
        bd.market_value / 100000000 as market_value_billion,
        c2.name as subcategory
    FROM board_data bd
    JOIN concept_boards cb ON bd.board_code = cb.board_code
    JOIN categories c1 ON cb.category_id = c1.id
    LEFT JOIN categories c2 ON cb.subcategory_id = c2.id
    WHERE c1.name = ?
    ORDER BY bd.change_percent DESC
    """
    
    df = pd.read_sql_query(query, conn, params=(category_name,))
    conn.close()
    return df

def get_board_details(board_name):
    """Get detailed information about a specific board"""
    conn = connect_db()
    
    query = """
    SELECT 
        cb.board_name,
        cb.board_code, 
        bd.rank, 
        bd.price, 
        bd.change_amount,
        bd.change_percent,
        bd.market_value / 100000000 as market_value_billion,
        bd.turnover_rate,
        bd.up_count,
        bd.down_count,
        bd.leading_stock,
        bd.leading_stock_change,
        c1.name as category,
        c2.name as subcategory
    FROM board_data bd
    JOIN concept_boards cb ON bd.board_code = cb.board_code
    LEFT JOIN categories c1 ON cb.category_id = c1.id
    LEFT JOIN categories c2 ON cb.subcategory_id = c2.id
    WHERE cb.board_name = ?
    """
    
    df = pd.read_sql_query(query, conn, params=(board_name,))
    conn.close()
    return df

if __name__ == "__main__":
    # Example usage
    print("Top 5 boards by change percent:")
    print(get_top_boards(5))
    
    print("\nTop 5 boards by market value:")
    print(get_top_boards(5, by_change=False))
    
    print("\nTechnology boards:")
    print(get_boards_by_category("科技与数字化"))
    
    print("\nDetails for '可燃冰' board:")
    print(get_board_details("可燃冰"))