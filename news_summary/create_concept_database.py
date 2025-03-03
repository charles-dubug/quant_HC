import json
import sqlite3
import os
from concept_boards_categorized import concept_categories, category_mapping

# Create database
db_path = os.path.join(os.path.dirname(__file__), 'concept_boards.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id INTEGER,
    FOREIGN KEY (parent_id) REFERENCES categories (id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS concept_boards (
    id INTEGER PRIMARY KEY,
    board_code TEXT NOT NULL,
    board_name TEXT NOT NULL,
    category_id INTEGER,
    subcategory_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories (id),
    FOREIGN KEY (subcategory_id) REFERENCES categories (id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS board_data (
    id INTEGER PRIMARY KEY,
    board_code TEXT NOT NULL,
    date TEXT NOT NULL,
    rank INTEGER,
    price REAL,
    change_amount REAL,
    change_percent REAL,
    market_value REAL,
    turnover_rate REAL,
    up_count INTEGER,
    down_count INTEGER,
    leading_stock TEXT,
    leading_stock_change REAL,
    FOREIGN KEY (board_code) REFERENCES concept_boards (board_code)
)
''')

# Insert categories
category_ids = {}
subcategory_ids = {}

# First, insert main categories
for category in concept_categories.keys():
    cursor.execute('INSERT INTO categories (name, parent_id) VALUES (?, NULL)', (category,))
    category_ids[category] = cursor.lastrowid

# Then, insert subcategories
for category, subcategories in concept_categories.items():
    if isinstance(subcategories, dict):
        for subcategory in subcategories.keys():
            cursor.execute('INSERT INTO categories (name, parent_id) VALUES (?, ?)', 
                          (subcategory, category_ids[category]))
            subcategory_ids[(category, subcategory)] = cursor.lastrowid

# Load board data from JSON
with open('concept_board_data.json', 'r', encoding='utf-8') as f:
    board_data = json.load(f)

# Process and insert board data
for board in board_data:
    board_name = board['板块名称']
    board_code = board['板块代码']
    
    # Find category and subcategory for this board
    category_id = None
    subcategory_id = None
    
    if board_name in category_mapping:
        mapping = category_mapping[board_name]
        category = mapping['category']
        subcategory = mapping['subcategory']
        
        category_id = category_ids.get(category)
        if subcategory:
            subcategory_id = subcategory_ids.get((category, subcategory))
    
    # Insert board info
    cursor.execute('''
    INSERT INTO concept_boards (board_code, board_name, category_id, subcategory_id)
    VALUES (?, ?, ?, ?)
    ''', (board_code, board_name, category_id, subcategory_id))
    
    # Insert current board data
    cursor.execute('''
    INSERT INTO board_data (
        board_code, date, rank, price, change_amount, change_percent,
        market_value, turnover_rate, up_count, down_count,
        leading_stock, leading_stock_change
    ) VALUES (?, date('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        board_code, 
        board['排名'],
        board['最新价'],
        board['涨跌额'],
        board['涨跌幅'],
        board['总市值'],
        board['换手率'],
        board['上涨家数'],
        board['下跌家数'],
        board['领涨股票'],
        board['领涨股票-涨跌幅']
    ))

# Commit changes and close connection
conn.commit()
conn.close()

print(f"Database created successfully at {db_path}")