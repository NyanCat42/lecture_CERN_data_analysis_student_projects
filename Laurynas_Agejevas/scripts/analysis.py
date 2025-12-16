#!/usr/bin/env python3
"""
Comprehensive analysis script for Hesburger menu data.
Runs all analysis scripts and creates interactive visualizations.
"""

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


def download_menu_pdf(pdf_path):
    """
    Download the menu PDF from Hesburger website if not present.
    
    Args:
        pdf_path: Path where the PDF should be saved
    """
    if pdf_path.exists():
        print(f"PDF already exists at {pdf_path}")
        return
    
    print(f"PDF not found. Downloading from Hesburger website...")
    
    url = "https://www.hesburger.lt/mellow/output/getfile.php?id=2770"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create data directory if it doesn't exist
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the PDF
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Successfully downloaded PDF to {pdf_path}")
        print(f"File size: {len(response.content) / 1024:.2f} KB")
        
    except requests.RequestException as e:
        print(f"Error downloading PDF: {e}")
        raise


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


def load_data(file_path=None):
    """Load the combined menu and nutrition data."""
    if file_path is None:
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        output_dir = project_dir / "output"
        file_path = output_dir / "combined_menu_nutrition.csv"
    
    df = pd.read_csv(file_path)
    print(f"\nLoaded {len(df)} items from {file_path}")
    return df


def create_price_vs_calories_plot(df, output_dir):
    """Create scatter plot of price vs calories."""
    print("\nCreating price vs calories scatter plot...")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(24, 13.5))
    
    # Define colors for each item type
    colors = {
        'base': '#FF6B6B',
        'small_combo': '#4ECDC4',
        'large_combo': '#95E1D3'
    }
    
    # Plot each item type separately for legend
    for item_type in ['base', 'small_combo', 'large_combo']:
        mask = df['item_type'] == item_type
        ax.scatter(
            df[mask]['price'],
            df[mask]['energy_kcal'],
            c=colors[item_type],
            label=item_type,
            alpha=0.6,
            s=100,
            edgecolors='white',
            linewidth=1.5
        )
    
    # Add text labels for each point
    for idx, row in df.iterrows():
        # Create label with item name and type
        label = f"{row['item_name']}"
        if row['item_type'] == 'small_combo':
            label += " (S)"
        elif row['item_type'] == 'large_combo':
            label += " (L)"
        
        ax.annotate(
            label,
            (row['price'], row['energy_kcal']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=7,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
        )
    
    # Add trend line
    z = np.polyfit(df['price'], df['energy_kcal'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['price'].min(), df['price'].max(), 100)
    ax.plot(x_trend, p(x_trend), 'k--', alpha=0.5, linewidth=2, label='Trend Line')
    
    # Labels and title
    ax.set_xlabel('Price (€)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Calories (kcal)', fontsize=12, fontweight='bold')
    ax.set_title('Calories vs Price for Hesburger Menu Items', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(fontsize=10, framealpha=0.9, loc='upper left')
    
    # Tight layout
    plt.tight_layout()
    
    # Save as PNG
    output_file = output_dir / "price_vs_calories.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    plt.close()
    
    return fig


def create_price_vs_protein_plot(df, output_dir):
    """Create scatter plot of price vs protein."""
    print("\nCreating price vs protein scatter plot...")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(24, 13.5))
    
    # Define colors for each item type
    colors = {
        'base': '#FF6B6B',
        'small_combo': '#4ECDC4',
        'large_combo': '#95E1D3'
    }
    
    # Plot each item type separately for legend
    for item_type in ['base', 'small_combo', 'large_combo']:
        mask = df['item_type'] == item_type
        ax.scatter(
            df[mask]['price'],
            df[mask]['proteins_g'],
            c=colors[item_type],
            label=item_type,
            alpha=0.6,
            s=100,
            edgecolors='white',
            linewidth=1.5
        )
    
    # Add text labels for each point
    for idx, row in df.iterrows():
        # Create label with item name and type
        label = f"{row['item_name']}"
        if row['item_type'] == 'small_combo':
            label += " (S)"
        elif row['item_type'] == 'large_combo':
            label += " (L)"
        
        ax.annotate(
            label,
            (row['price'], row['proteins_g']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=7,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
        )
    
    # Add trend line
    z = np.polyfit(df['price'], df['proteins_g'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['price'].min(), df['price'].max(), 100)
    ax.plot(x_trend, p(x_trend), 'k--', alpha=0.5, linewidth=2, label='Trend Line')
    
    # Labels and title
    ax.set_xlabel('Price (€)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Protein (g)', fontsize=12, fontweight='bold')
    ax.set_title('Protein Content vs Price for Hesburger Menu Items', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(fontsize=10, framealpha=0.9, loc='upper left')
    
    # Tight layout
    plt.tight_layout()
    
    # Save as PNG
    output_file = output_dir / "price_vs_protein.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    plt.close()
    
    return fig


def calculate_statistics(df):
    """Calculate comprehensive statistics from the data."""
    print("\nCalculating statistics...")
    
    stats = {
        'total_items': len(df),
        'unique_items': df['item_name'].nunique(),
        'item_types': df['item_type'].value_counts().to_dict(),
        
        # Price statistics
        'avg_price': df['price'].mean(),
        'median_price': df['price'].median(),
        'min_price': df['price'].min(),
        'max_price': df['price'].max(),
        'std_price': df['price'].std(),
        
        # Calorie statistics
        'avg_calories': df['energy_kcal'].mean(),
        'median_calories': df['energy_kcal'].median(),
        'min_calories': df['energy_kcal'].min(),
        'max_calories': df['energy_kcal'].max(),
        'std_calories': df['energy_kcal'].std(),
        
        # Protein statistics
        'avg_protein': df['proteins_g'].mean(),
        'median_protein': df['proteins_g'].median(),
        'min_protein': df['proteins_g'].min(),
        'max_protein': df['proteins_g'].max(),
        'std_protein': df['proteins_g'].std(),
        
        # Fat statistics
        'avg_fat': df['fats_g'].mean(),
        'median_fat': df['fats_g'].median(),
        
        # Carbohydrate statistics
        'avg_carbs': df['carbohydrates_g'].mean(),
        'median_carbs': df['carbohydrates_g'].median(),
        
        # Price efficiency
        'avg_price_per_100kcal': df['price_per_100kcal'].mean(),
        'best_value_item': df.loc[df['price_per_100kcal'].idxmin(), 'item_name'],
        'best_value_price_per_100kcal': df['price_per_100kcal'].min(),
        
        # Correlations
        'price_calories_corr': df['price'].corr(df['energy_kcal']),
        'price_protein_corr': df['price'].corr(df['proteins_g']),
        'price_fat_corr': df['price'].corr(df['fats_g']),
        'price_carbs_corr': df['price'].corr(df['carbohydrates_g']),
    }
    
    # Most expensive items
    stats['most_expensive'] = df.nlargest(5, 'price')[['item_name', 'item_type', 'price']].to_dict('records')
    
    # Cheapest items
    stats['cheapest'] = df.nsmallest(5, 'price')[['item_name', 'item_type', 'price']].to_dict('records')
    
    # Highest calorie items
    stats['highest_calories'] = df.nlargest(5, 'energy_kcal')[['item_name', 'item_type', 'energy_kcal']].to_dict('records')
    
    # Highest protein items
    stats['highest_protein'] = df.nlargest(5, 'proteins_g')[['item_name', 'item_type', 'proteins_g']].to_dict('records')
    
    # Best value items (lowest price per 100 kcal)
    stats['best_value'] = df.nsmallest(5, 'price_per_100kcal')[['item_name', 'item_type', 'price', 'energy_kcal', 'price_per_100kcal']].to_dict('records')
    
    return stats


def generate_markdown_summary(df, stats, output_dir):
    """Generate a comprehensive markdown summary report."""
    print("\nGenerating markdown summary...")
    
    # Create markdown content
    md = []
    md.append("# Hesburger Menu Analysis Report")
    md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("\n---\n")
    
    # Overview
    md.append("## Overview")
    md.append(f"- **Total items analyzed:** {stats['total_items']}")
    md.append(f"- **Unique menu items:** {stats['unique_items']}")
    md.append(f"- **Item types breakdown:**")
    for item_type, count in stats['item_types'].items():
        md.append(f"  - {item_type}: {count}")
    md.append("")
    
    # Price Statistics
    md.append("## Price Statistics")
    md.append(f"- **Average price:** €{stats['avg_price']:.2f}")
    md.append(f"- **Median price:** €{stats['median_price']:.2f}")
    md.append(f"- **Price range:** €{stats['min_price']:.2f} - €{stats['max_price']:.2f}")
    md.append(f"- **Standard deviation:** €{stats['std_price']:.2f}")
    md.append("")
    
    # Nutritional Statistics
    md.append("## Nutritional Statistics")
    md.append("\n### Calories")
    md.append(f"- **Average:** {stats['avg_calories']:.0f} kcal")
    md.append(f"- **Median:** {stats['median_calories']:.0f} kcal")
    md.append(f"- **Range:** {stats['min_calories']:.0f} - {stats['max_calories']:.0f} kcal")
    md.append("")
    
    md.append("### Protein")
    md.append(f"- **Average:** {stats['avg_protein']:.1f} g")
    md.append(f"- **Median:** {stats['median_protein']:.1f} g")
    md.append(f"- **Range:** {stats['min_protein']:.1f} - {stats['max_protein']:.1f} g")
    md.append("")
    
    md.append("### Fats")
    md.append(f"- **Average:** {stats['avg_fat']:.1f} g")
    md.append(f"- **Median:** {stats['median_fat']:.1f} g")
    md.append("")
    
    md.append("### Carbohydrates")
    md.append(f"- **Average:** {stats['avg_carbs']:.1f} g")
    md.append(f"- **Median:** {stats['median_carbs']:.1f} g")
    md.append("")
    
    # Price Efficiency
    md.append("## Price Efficiency")
    md.append(f"- **Average price per 100 kcal:** €{stats['avg_price_per_100kcal']:.2f}")
    md.append(f"- **Best value item:** {stats['best_value_item']} (€{stats['best_value_price_per_100kcal']:.2f} per 100 kcal)")
    md.append("")
    
    # Correlations
    md.append("## Correlations with Price")
    md.append(f"- **Calories:** {stats['price_calories_corr']:.3f}")
    md.append(f"- **Protein:** {stats['price_protein_corr']:.3f}")
    md.append(f"- **Fat:** {stats['price_fat_corr']:.3f}")
    md.append(f"- **Carbohydrates:** {stats['price_carbs_corr']:.3f}")
    md.append("")
    md.append("*Note: Values range from -1 (negative correlation) to +1 (positive correlation)*")
    md.append("")
    
    # Most Expensive Items
    md.append("## Most Expensive Items")
    md.append("| Rank | Item Name | Type | Price |")
    md.append("|------|-----------|------|-------|")
    for i, item in enumerate(stats['most_expensive'], 1):
        md.append(f"| {i} | {item['item_name']} | {item['item_type']} | €{item['price']:.2f} |")
    md.append("")
    
    # Cheapest Items
    md.append("## Cheapest Items")
    md.append("| Rank | Item Name | Type | Price |")
    md.append("|------|-----------|------|-------|")
    for i, item in enumerate(stats['cheapest'], 1):
        md.append(f"| {i} | {item['item_name']} | {item['item_type']} | €{item['price']:.2f} |")
    md.append("")
    
    # Highest Calorie Items
    md.append("## Highest Calorie Items")
    md.append("| Rank | Item Name | Type | Calories |")
    md.append("|------|-----------|------|----------|")
    for i, item in enumerate(stats['highest_calories'], 1):
        md.append(f"| {i} | {item['item_name']} | {item['item_type']} | {item['energy_kcal']:.0f} kcal |")
    md.append("")
    
    # Highest Protein Items
    md.append("## Highest Protein Items")
    md.append("| Rank | Item Name | Type | Protein |")
    md.append("|------|-----------|------|---------|")
    for i, item in enumerate(stats['highest_protein'], 1):
        md.append(f"| {i} | {item['item_name']} | {item['item_type']} | {item['proteins_g']:.1f} g |")
    md.append("")
    
    # Best Value Items
    md.append("## Best Value Items (Price per 100 kcal)")
    md.append("| Rank | Item Name | Type | Price | Calories | €/100kcal |")
    md.append("|------|-----------|------|-------|----------|-----------|")
    for i, item in enumerate(stats['best_value'], 1):
        md.append(f"| {i} | {item['item_name']} | {item['item_type']} | €{item['price']:.2f} | {item['energy_kcal']:.0f} kcal | €{item['price_per_100kcal']:.2f} |")
    md.append("")
    
    # Key Findings
    md.append("## Key Findings")
    md.append("")
    md.append("### Price vs Calories")
    md.append(f"- There is a **{'strong' if abs(stats['price_calories_corr']) > 0.7 else 'moderate' if abs(stats['price_calories_corr']) > 0.4 else 'weak'} positive correlation** (r={stats['price_calories_corr']:.3f}) between price and calorie content.")
    md.append(f"- Higher-priced items generally contain more calories, as expected.")
    md.append(f"- See the scatter plot: ![Price vs Calories](price_vs_calories.png)")
    md.append("")
    
    md.append("### Price vs Protein")
    md.append(f"- There is a **{'strong' if abs(stats['price_protein_corr']) > 0.7 else 'moderate' if abs(stats['price_protein_corr']) > 0.4 else 'weak'} positive correlation** (r={stats['price_protein_corr']:.3f}) between price and protein content.")
    md.append(f"- Protein-rich items tend to be more expensive.")
    md.append(f"- See the scatter plot: ![Price vs Protein](price_vs_protein.png)")
    md.append("")
    
    md.append("### Value for Money")
    md.append(f"- The best value item is **{stats['best_value_item']}** at €{stats['best_value_price_per_100kcal']:.2f} per 100 kcal.")
    md.append(f"- Base items generally offer better value than combo meals in terms of price per calorie.")
    md.append(f"- However, combo meals provide a complete meal experience with sides and drinks.")
    md.append("")
    
    # Methodology
    md.append("## Methodology")
    md.append("")
    md.append("### Data Collection")
    md.append("1. **Menu extraction:** OCR (EasyOCR) was used to extract text from the Hesburger PDF menu")
    md.append("2. **Price parsing:** Regular expressions parsed item names and prices from the extracted text")
    md.append("3. **Nutritional data:** Web scraping from the official Hesburger nutritional information page")
    md.append("4. **Data matching:** Item names were normalized and matched between price and nutrition datasets")
    md.append("5. **Combo calculations:** Combo meal nutrition was calculated by adding base item + fries + drink")
    md.append("")
    
    md.append("### Analysis Tools")
    md.append("- **Python 3** with pandas for data processing")
    md.append("- **Matplotlib** for data visualizations")
    md.append("- **EasyOCR** for PDF text extraction")
    md.append("- **BeautifulSoup** for web scraping")
    md.append("")
    
    md.append("---")
    md.append(f"\n*Report generated by analysis.py on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*")
    
    # Write to file
    output_file = output_dir / "analysis_summary.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    
    print(f"Saved markdown summary to {output_file}")
    return output_file


def main():
    """Main function to run complete analysis."""
    print("=" * 60)
    print("HESBURGER MENU COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_dir = project_dir / "output"
    
    # Check if combined data exists, if not run the pipeline
    combined_file = output_dir / "combined_menu_nutrition.csv"
    
    if not combined_file.exists():
        print("\nCombined data not found. Running complete pipeline...")
        combined_file = run_complete_pipeline()
    else:
        print(f"\nUsing existing data from {combined_file}")
        print("To regenerate data, delete the output files and run again.")
    
    # Load data
    df = load_data(combined_file)
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING INTERACTIVE VISUALIZATIONS")
    print("=" * 60)
    
    create_price_vs_calories_plot(df, output_dir)
    create_price_vs_protein_plot(df, output_dir)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    summary_file = generate_markdown_summary(df, stats, output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. Data: {combined_file}")
    print(f"  2. Price vs Calories Plot: {output_dir / 'price_vs_calories.png'}")
    print(f"  3. Price vs Protein Plot: {output_dir / 'price_vs_protein.png'}")
    print(f"  4. Summary Report: {summary_file}")
    print("\nOpen the PNG files to view the plots!")
    print("=" * 60)


if __name__ == "__main__":
    main()
