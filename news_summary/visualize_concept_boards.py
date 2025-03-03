import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import os
import seaborn as sns
from query_concept_boards import get_top_boards, get_boards_by_category

def plot_top_boards(limit=10, by_change=True):
    """Plot top performing concept boards"""
    df = get_top_boards(limit, by_change)
    
    metric = "Change %" if by_change else "Market Value (Billion)"
    value_col = "change_percent" if by_change else "market_value_billion"
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(df['board_name'], df[value_col], color=sns.color_palette("viridis", limit))
    
    plt.xlabel(metric)
    plt.ylabel('Concept Board')
    plt.title(f'Top {limit} Concept Boards by {metric}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        plt.text(label_x_pos + 0.1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'top_boards_by_{"change" if by_change else "market_value"}.png')
    plt.show()

def plot_category_distribution():
    """Plot distribution of boards across categories"""
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'concept_boards.db'))
    
    query = """
    SELECT c.name as category, COUNT(cb.id) as board_count
    FROM categories c
    JOIN concept_boards cb ON c.id = cb.category_id
    WHERE c.parent_id IS NULL
    GROUP BY c.name
    ORDER BY board_count DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['category'], df['board_count'], color=sns.color_palette("muted", len(df)))
    
    plt.xlabel('Category')
    plt.ylabel('Number of Boards')
    plt.title('Distribution of Concept Boards Across Categories')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('category_distribution.png')
    plt.show()

def plot_category_performance():
    """Plot average performance by category"""
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'concept_boards.db'))
    
    query = """
    SELECT 
        c.name as category, 
        AVG(bd.change_percent) as avg_change,
        AVG(bd.market_value) / 100000000 as avg_market_value
    FROM categories c
    JOIN concept_boards cb ON c.id = cb.category_id
    JOIN board_data bd ON cb.board_code = bd.board_code
    WHERE c.parent_id IS NULL
    GROUP BY c.name
    ORDER BY avg_change DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot average change percent
    bars1 = ax1.bar(df['category'], df['avg_change'], color=sns.color_palette("coolwarm", len(df)))
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Average Change %')
    ax1.set_title('Average Performance by Category')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot average market value
    df_sorted = df.sort_values('avg_market_value', ascending=False)
    bars2 = ax2.bar(df_sorted['category'], df_sorted['avg_market_value'], 
                   color=sns.color_palette("viridis", len(df)))
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Average Market Value (Billion)')
    ax2.set_title('Average Market Value by Category')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Add values on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}B', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('category_performance.png')
    plt.show()

if __name__ == "__main__":
    # Create visualizations
    plot_top_boards(10, by_change=True)
    plot_top_boards(10, by_change=False)
    plot_category_distribution()
    plot_category_performance()