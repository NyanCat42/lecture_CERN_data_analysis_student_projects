import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import requests

# Import functions from other scripts
from ocr_the_menu import extract_text_from_pdf
from analyze_the_menu import parse_menu_text
from get_nutritional_info import scrape_nutritional_info, clean_nutritional_data
from match_price_to_nutrition import match_price_to_nutrition
from download_menu_pdf import download_menu_pdf


def run_complete_pipeline():
    """
    Run the complete data collection pipeline:
    1. Download menu PDF if not present
    2. OCR the menu PDF
    3. Parse menu items and prices
    4. Scrape nutritional information
    5. Match prices with nutrition data
    """
    print("=" * 60)
    print("RUNNING COMPLETE DATA COLLECTION PIPELINE")
    print("=" * 60)
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "output"
    
    # Create directories
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Download PDF if not present
    pdf_file = data_dir / "lit-hb-price-list-11.2025.pdf"
    print("\n" + "=" * 60)
    print("STEP 0: Checking/Downloading Menu PDF")
    print("=" * 60)
    download_menu_pdf(pdf_file)
    
    # Step 1: OCR the menu
    print("\n" + "=" * 60)
    print("STEP 1: Extracting text from PDF using OCR")
    print("=" * 60)
    menu_text_file = output_dir / "menu_text.txt"
    
    if not menu_text_file.exists():
        print(f"Extracting text from {pdf_file}...")
        extract_text_from_pdf(pdf_file, menu_text_file)
    else:
        print(f"Menu text already exists at {menu_text_file}")
    
    # Step 2: Parse menu items
    print("\n" + "=" * 60)
    print("STEP 2: Parsing menu items and prices")
    print("=" * 60)
    menu_items_file = output_dir / "menu_items.csv"
    
    if not menu_items_file.exists():
        print("Parsing menu text...")
        parse_menu_text(menu_text_file, menu_items_file)
    else:
        print(f"Menu items already exists at {menu_items_file}")
    
    # Step 3: Scrape nutritional info
    print("\n" + "=" * 60)
    print("STEP 3: Scraping nutritional information from website")
    print("=" * 60)
    nutrition_file = output_dir / "nutritional_info.csv"
    
    if not nutrition_file.exists():
        print("Scraping nutritional data...")
        url = "https://www.hesburger.lt/maistin---vert---ir-alergenai"
        df = scrape_nutritional_info(url)
        df = clean_nutritional_data(df)
        df.to_csv(nutrition_file, index=False)
    else:
        print(f"Nutritional info already exists at {nutrition_file}")
    
    # Step 4: Match prices with nutrition
    print("\n" + "=" * 60)
    print("STEP 4: Matching prices with nutritional information")
    print("=" * 60)
    combined_file = output_dir / "combined_menu_nutrition.csv"
    
    print("Combining data...")
    match_price_to_nutrition(
        str(menu_items_file),
        str(nutrition_file),
        str(combined_file)
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return combined_file


def create_calories_vs_price_plot(data_file, output_dir):
    """
    Create a scatter plot of calories vs price with €/kcal reference lines
    and labels for best deals.
    """
    df = pd.read_csv(data_file)
    
    # Filter out items with zero calories or very low prices
    df = df[(df['energy_kcal'] > 0) & (df['price'] > 0)]
    
    # Calculate kcal per euro
    df['kcal_per_euro'] = df['energy_kcal'] / df['price']
    
    # Create figure with 16:9 aspect ratio
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Color by item type
    types = df['item_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    type_colors = dict(zip(types, colors))
    
    for item_type in types:
        mask = df['item_type'] == item_type
        ax.scatter(df[mask]['price'], df[mask]['energy_kcal'], 
                  c=[type_colors[item_type]], label=item_type, 
                  s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add reference lines for €/kcal
    price_range = np.linspace(0, df['price'].max(), 100)
    
    # Add reference lines: 100, 200, 300 kcal per euro
    kcal_per_euro_values = [100, 200, 300]
    for kcal_per_euro in kcal_per_euro_values:
        calories = price_range * kcal_per_euro
        ax.plot(price_range, calories, '--', alpha=0.3, linewidth=2, 
               label=f'{kcal_per_euro} kcal/€')
    
    # Find and label the best deals (top 10 by kcal per euro)
    best_deals = df.nlargest(10, 'kcal_per_euro')
    
    for idx, row in best_deals.iterrows():
        ax.annotate(row['item_name'], 
                   xy=(row['price'], row['energy_kcal']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.5))
    
    ax.set_xlabel('Price (€)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Calories (kcal)', fontsize=14, fontweight='bold')
    ax.set_title('Calories vs Price - Hesburger Menu Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1600)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    output_file = output_dir / 'calories_vs_price.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    return output_file


def create_protein_vs_price_plot(data_file, output_dir):
    """
    Create a scatter plot of protein vs price with g/€ reference lines
    and labels for best deals.
    """
    df = pd.read_csv(data_file)
    
    # Filter out items with zero protein or very low prices
    df = df[(df['proteins_g'] > 0) & (df['price'] > 0)]
    
    # Calculate protein per euro
    df['protein_per_euro'] = df['proteins_g'] / df['price']
    
    # Create figure with 16:9 aspect ratio
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Color by item type
    types = df['item_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    type_colors = dict(zip(types, colors))
    
    for item_type in types:
        mask = df['item_type'] == item_type
        ax.scatter(df[mask]['price'], df[mask]['proteins_g'], 
                  c=[type_colors[item_type]], label=item_type, 
                  s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add reference lines for protein g/€
    price_range = np.linspace(0, df['price'].max(), 100)
    
    # Add reference lines: 5, 10, 15, 20 g protein per euro
    protein_per_euro_values = [5, 10, 15]
    for protein_per_euro in protein_per_euro_values:
        protein = price_range * protein_per_euro
        ax.plot(price_range, protein, '--', alpha=0.3, linewidth=2, 
               label=f'{protein_per_euro} g protein/€')
    
    # Find and label the best deals (top 10 by protein per euro)
    best_deals = df.nlargest(10, 'protein_per_euro')
    
    for idx, row in best_deals.iterrows():
        ax.annotate(row['item_name'], 
                   xy=(row['price'], row['proteins_g']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.5))
    
    ax.set_xlabel('Price (€)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Protein (g)', fontsize=14, fontweight='bold')
    ax.set_title('Protein vs Price - Hesburger Menu Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    output_file = output_dir / 'protein_vs_price.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    return output_file


if __name__ == "__main__":
    # Run the pipeline
    combined_file = run_complete_pipeline()
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / "output"
    
    print("\nGenerating calories vs price plot...")
    create_calories_vs_price_plot(combined_file, output_dir)
    
    print("\nGenerating protein vs price plot...")
    create_protein_vs_price_plot(combined_file, output_dir)
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)

