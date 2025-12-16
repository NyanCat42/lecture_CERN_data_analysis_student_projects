#!/usr/bin/env python3
"""
Script to scrape nutritional information from Hesburger website.
Extracts kcal, fats, carbohydrates, proteins, etc. for each menu item.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os


def scrape_nutritional_info(url):
    """
    Scrape nutritional information from Hesburger nutrition page.
    
    Args:
        url (str): URL of the Hesburger nutritional information page
        
    Returns:
        pd.DataFrame: DataFrame containing nutritional information for all items
    """
    print(f"Fetching data from {url}...")
    
    # Send GET request with headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all tables in the page
    tables = soup.find_all('table')
    
    print(f"Found {len(tables)} tables on the page")
    
    # List to store all nutritional data
    all_data = []
    
    # Define expected column names (in Lithuanian)
    expected_columns = [
        'Gaminiai',  # Product name
        'energinė vertė (kJ)',  # Energy in kJ
        'energinė vertė (kcal)',  # Energy in kcal
        'riebalai (g)',  # Fats
        'angliavandenių ( g)',  # Carbohydrates (note the space)
        'Iš jų cukraus (g)',  # Of which sugars
        'baltymai (g)',  # Proteins
        'druska ( g)'  # Salt (note the space)
    ]
    
    # Process each table
    for table_idx, table in enumerate(tables):
        rows = table.find_all('tr')
        
        # Skip if table has no rows
        if not rows:
            continue
        
        # Extract header (first row)
        header_row = rows[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        
        # Check if this looks like a nutritional table (should have many columns)
        if len(headers) < 8:
            continue
        
        # Process data rows (skip header)
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            
            # Skip empty rows or rows with less than 8 cells
            if len(cells) < 8:
                continue
            
            # Extract cell values
            row_data = [cell.get_text(strip=True) for cell in cells]
            
            # Skip if first cell (item name) is empty
            if not row_data[0]:
                continue
            
            # Create dictionary for this item - take first 8 columns
            item_dict = {
                'Gaminiai': row_data[0],
                'energinė vertė (kJ)': row_data[1] if len(row_data) > 1 else '',
                'energinė vertė (kcal)': row_data[2] if len(row_data) > 2 else '',
                'riebalai (g)': row_data[3] if len(row_data) > 3 else '',
                'angliavandenių ( g)': row_data[4] if len(row_data) > 4 else '',
                'Iš jų cukraus (g)': row_data[5] if len(row_data) > 5 else '',
                'baltymai (g)': row_data[6] if len(row_data) > 6 else '',
                'druska ( g)': row_data[7] if len(row_data) > 7 else ''
            }
            
            all_data.append(item_dict)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"Scraped {len(df)} items")
    
    return df


def clean_nutritional_data(df):
    """
    Clean and standardize the nutritional data.
    
    Args:
        df (pd.DataFrame): Raw nutritional data
        
    Returns:
        pd.DataFrame: Cleaned nutritional data
    """
    # Common column name mappings (Lithuanian to English)
    column_mapping = {
        'Gaminiai': 'item_name',
        'energinė vertė (kJ)': 'energy_kj',
        'energinė vertė (kcal)': 'energy_kcal',
        'riebalai (g)': 'fats_g',
        'angliavandenių ( g)': 'carbohydrates_g',
        'Iš jų cukraus (g)': 'sugar_g',
        'baltymai (g)': 'proteins_g',
        'druska ( g)': 'salt_g'
    }
    
    # Rename columns if they exist
    df = df.rename(columns=column_mapping)
    
    # Select only the nutritional columns we need
    nutritional_columns = [
        'item_name', 'energy_kj', 'energy_kcal', 'fats_g', 
        'carbohydrates_g', 'sugar_g', 'proteins_g', 'salt_g'
    ]
    
    # Keep only existing columns
    existing_columns = [col for col in nutritional_columns if col in df.columns]
    df = df[existing_columns]
    
    # Convert numeric columns to float, handling empty strings
    numeric_columns = [col for col in existing_columns if col != 'item_name']
    
    for col in numeric_columns:
        # Replace empty strings and handle commas as decimal separators
        df[col] = df[col].replace('', '0')
        # Remove spaces (for numbers like "1 971" -> "1971")
        df[col] = df[col].str.replace(' ', '', regex=False)
        # Replace commas with dots (for decimals like "26,7" -> "26.7")
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Remove duplicate items (keep first occurrence)
    df = df.drop_duplicates(subset=['item_name'], keep='first')
    
    # Remove items with all zero nutritional values (likely invalid entries)
    mask = df[numeric_columns].sum(axis=1) > 0
    df = df[mask]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Cleaned data: {len(df)} unique items")
    
    return df


def save_nutritional_data(df, output_path):
    """
    Save nutritional data to CSV file.
    
    Args:
        df (pd.DataFrame): Nutritional data to save
        output_path (str): Path to output CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved nutritional data to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total items: {len(df)}")
    
    if 'energy_kcal' in df.columns:
        print(f"Average calories: {df['energy_kcal'].mean():.1f} kcal")
        print(f"Min calories: {df['energy_kcal'].min():.1f} kcal")
        print(f"Max calories: {df['energy_kcal'].max():.1f} kcal")
    
    if 'fats_g' in df.columns:
        print(f"Average fats: {df['fats_g'].mean():.1f} g")
    
    if 'carbohydrates_g' in df.columns:
        print(f"Average carbohydrates: {df['carbohydrates_g'].mean():.1f} g")
    
    if 'proteins_g' in df.columns:
        print(f"Average proteins: {df['proteins_g'].mean():.1f} g")
    
    # Show first few items
    print("\n=== Sample Data (first 5 items) ===")
    print(df.head().to_string(index=False))


def main():
    """Main function to scrape and save nutritional information."""
    # URL of the Hesburger nutritional information page
    url = "https://www.hesburger.lt/maistin---vert---ir-alergenai"
    
    # Output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'output', 'nutritional_info.csv')
    
    try:
        # Scrape the data
        df = scrape_nutritional_info(url)
        
        # Clean the data
        df = clean_nutritional_data(df)
        
        # Save the data
        save_nutritional_data(df, output_path)
        
        print("\n✓ Nutritional information successfully scraped and saved!")
        
    except requests.RequestException as e:
        print(f"Error fetching data from website: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()
