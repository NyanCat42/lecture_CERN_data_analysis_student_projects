import pandas as pd
import numpy as np
import os
import re


def normalize_name(name):
    """
    Normalize item names by removing OCR artifacts and standardizing format.
    
    Args:
        name (str): Original item name
        
    Returns:
        str: Normalized item name
    """
    if pd.isna(name):
        return ""
    
    # Convert to string and lowercase
    name = str(name).lower().strip()
    
    # Remove leading/trailing quotes, asterisks, commas
    name = name.strip('"\'*,')
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Remove ** markers (used for special items)
    name = name.replace('**', '')
    
    # Replace VEKE@ with VEKE®
    name = name.replace('veke@', 'veke®')
    
    # Remove stray opening quotes/brackets at the start
    name = re.sub(r'^["\',]+', '', name)
    
    return name.strip()


def clean_name_for_display(name):
    """
    Clean item name for display by removing OCR artifacts, numbers, and special characters.
    
    Args:
        name (str): Original item name
        
    Returns:
        str: Cleaned item name
    """
    if pd.isna(name):
        return ""
    
    # Convert to string
    name = str(name).strip()
    
    # Remove leading/trailing quotes, asterisks, commas
    name = name.strip('"\'*,')
    
    # Remove all asterisks
    name = name.replace('*', '')
    
    # Remove all quotes (both single and double)
    name = name.replace('"', '')
    name = name.replace("'", '')
    
    # Remove all commas
    name = name.replace(',', '')
    
    # Remove numbers (including decimals like 2.30, 0.3, etc.)
    name = re.sub(r'\d+[\.,]?\d*', '', name)
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Remove leading/trailing spaces
    name = name.strip()
    
    return name


def find_matching_nutrition(item_name, nutrition_df):
    """
    Find nutritional information for an item by matching normalized names.
    
    Args:
        item_name (str): Item name from menu
        nutrition_df (pd.DataFrame): Nutritional information dataframe
        
    Returns:
        pd.Series or None: Nutritional data if found, None otherwise
    """
    normalized_search = normalize_name(item_name)
    
    # Try exact match first
    for idx, row in nutrition_df.iterrows():
        if normalize_name(row['item_name']) == normalized_search:
            return row
    
    # Try partial match (in case of slight variations)
    for idx, row in nutrition_df.iterrows():
        normalized_nutrition = normalize_name(row['item_name'])
        # Check if either name contains the other
        if (normalized_search in normalized_nutrition or 
            normalized_nutrition in normalized_search) and len(normalized_search) > 5:
            return row
    
    return None


def add_nutritional_values(base_nutrition, additional_nutritions):
    """
    Add nutritional values from multiple items.
    
    Args:
        base_nutrition (pd.Series): Base item nutrition
        additional_nutritions (list): List of additional nutrition Series
        
    Returns:
        pd.Series: Combined nutritional values
    """
    result = base_nutrition.copy()
    
    nutritional_fields = ['energy_kj', 'energy_kcal', 'fats_g', 
                         'carbohydrates_g', 'sugar_g', 'proteins_g', 'salt_g']
    
    for additional in additional_nutritions:
        if additional is not None:
            for field in nutritional_fields:
                if field in result and field in additional:
                    result[field] = result[field] + additional[field]
    
    return result


def match_price_to_nutrition(menu_path, nutrition_path, output_path):
    """
    Match menu prices with nutritional information and save combined dataset.
    
    Args:
        menu_path (str): Path to menu items CSV
        nutrition_path (str): Path to nutritional info CSV
        output_path (str): Path to save combined CSV
    """
    print("Loading data files...")
    
    # Load data
    menu_df = pd.read_csv(menu_path)
    nutrition_df = pd.read_csv(nutrition_path)
    
    print(f"Menu items: {len(menu_df)}")
    print(f"Nutrition items: {len(nutrition_df)}")
    
    # Get combo components
    fries_small = find_matching_nutrition("Skrudintos bulvytės", nutrition_df)
    fries_large = find_matching_nutrition("Skrudintos bulvytės, didelė porcija", nutrition_df)
    cola_small = find_matching_nutrition("Coca-Cola / 0.25l", nutrition_df)
    cola_large = find_matching_nutrition("Coca-Cola / 0,5l", nutrition_df)
    
    print("\nCombo components found:")
    print(f"  Small fries: {'✓' if fries_small is not None else '✗'}")
    print(f"  Large fries: {'✓' if fries_large is not None else '✗'}")
    print(f"  Small Coca-Cola: {'✓' if cola_small is not None else '✗'}")
    print(f"  Large Coca-Cola: {'✓' if cola_large is not None else '✗'}")
    
    # Prepare output data
    matched_items = []
    unmatched_items = []
    
    print("\nMatching items...")
    
    for idx, row in menu_df.iterrows():
        item_name = row['item_name']
        
        # Skip items that look like metadata or invalid entries
        if pd.isna(item_name) or len(str(item_name).strip()) < 3:
            continue
        
        # Skip Kečupas completely
        if 'kečupas' in item_name.lower():
            continue

        # Skip Kečupas completely
        if 'arbata' in item_name.lower():
            continue
        
        # Skip obvious non-food items
        skip_keywords = ['padažas', 'padažai', 'siekiant', 'hesburger']
        if any(keyword in item_name.lower() for keyword in skip_keywords):
            if 'mėsainis' not in item_name.lower():  # Keep Hesburger burger
                continue
        
        # Fix price parsing for VEKE Dvigubas mėsainis
        if 'veke@ dvigubas mėsainis su sūriu' in item_name.lower() and '2.30' in item_name:
            # Extract the correct prices
            row = row.copy()
            row['base_price'] = 2.30
            row['small_combo_price'] = 4.6
            row['large_combo_price'] = 5.7
        
        # Find base nutrition
        base_nutrition = find_matching_nutrition(item_name, nutrition_df)
        
        if base_nutrition is None:
            unmatched_items.append(item_name)
            continue
        
        # Create base item entry
        clean_item_name = clean_name_for_display(item_name)
        
        base_item = {
            'item_name': clean_item_name,
            'item_type': 'base',
            'price': row['base_price'],
            'energy_kj': base_nutrition['energy_kj'],
            'energy_kcal': base_nutrition['energy_kcal'],
            'fats_g': base_nutrition['fats_g'],
            'carbohydrates_g': base_nutrition['carbohydrates_g'],
            'sugar_g': base_nutrition['sugar_g'],
            'proteins_g': base_nutrition['proteins_g'],
            'salt_g': base_nutrition['salt_g']
        }
        matched_items.append(base_item)
        
        # Check if combos should be skipped for this item
        no_combo_items = ['arbata', 'graikiškos salotos']
        skip_combos = any(keyword in item_name.lower() for keyword in no_combo_items)
        
        # Add small combo if price exists
        if not skip_combos and pd.notna(row['small_combo_price']) and row['small_combo_price'] != '':
            if fries_small is not None and cola_small is not None:
                combo_nutrition = add_nutritional_values(
                    base_nutrition, 
                    [fries_small, cola_small]
                )
                
                small_combo = {
                    'item_name': clean_item_name,
                    'item_type': 'small_combo',
                    'price': row['small_combo_price'],
                    'energy_kj': combo_nutrition['energy_kj'],
                    'energy_kcal': combo_nutrition['energy_kcal'],
                    'fats_g': combo_nutrition['fats_g'],
                    'carbohydrates_g': combo_nutrition['carbohydrates_g'],
                    'sugar_g': combo_nutrition['sugar_g'],
                    'proteins_g': combo_nutrition['proteins_g'],
                    'salt_g': combo_nutrition['salt_g']
                }
                matched_items.append(small_combo)
        
        # Add large combo if price exists
        if not skip_combos and pd.notna(row['large_combo_price']) and row['large_combo_price'] != '':
            if fries_large is not None and cola_large is not None:
                combo_nutrition = add_nutritional_values(
                    base_nutrition, 
                    [fries_large, cola_large]
                )
                
                large_combo = {
                    'item_name': clean_item_name,
                    'item_type': 'large_combo',
                    'price': row['large_combo_price'],
                    'energy_kj': combo_nutrition['energy_kj'],
                    'energy_kcal': combo_nutrition['energy_kcal'],
                    'fats_g': combo_nutrition['fats_g'],
                    'carbohydrates_g': combo_nutrition['carbohydrates_g'],
                    'sugar_g': combo_nutrition['sugar_g'],
                    'proteins_g': combo_nutrition['proteins_g'],
                    'salt_g': combo_nutrition['salt_g']
                }
                matched_items.append(large_combo)
    
    # Create output dataframe
    output_df = pd.DataFrame(matched_items)
    
    # Convert price to float
    output_df['price'] = pd.to_numeric(output_df['price'], errors='coerce')
    
    # Remove rows with invalid prices
    output_df = output_df.dropna(subset=['price'])
    
    # Sort by item name and type
    type_order = {'base': 0, 'small_combo': 1, 'large_combo': 2}
    output_df['type_order'] = output_df['item_type'].map(type_order)
    output_df = output_df.sort_values(['item_name', 'type_order'])
    output_df = output_df.drop('type_order', axis=1)
    
    # Calculate price per 100 kcal
    output_df['price_per_100kcal'] = (output_df['price'] / output_df['energy_kcal'] * 100).round(2)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    print(f"\n=== Results ===")
    print(f"Matched items: {len(matched_items)}")
    print(f"Unmatched items: {len(unmatched_items)}")
    print(f"Unique items matched: {output_df['item_name'].nunique()}")
    print(f"Total entries (with combos): {len(output_df)}")
    
    if unmatched_items:
        print(f"\n=== Unmatched items ({len(unmatched_items)}) ===")
        for item in unmatched_items[:10]:  # Show first 10
            print(f"  - {item}")
        if len(unmatched_items) > 10:
            print(f"  ... and {len(unmatched_items) - 10} more")
    
    # Show statistics
    print(f"\n=== Statistics ===")
    print(f"Average price: €{output_df['price'].mean():.2f}")
    print(f"Average calories: {output_df['energy_kcal'].mean():.0f} kcal")
    print(f"Average price per 100 kcal: €{output_df['price_per_100kcal'].mean():.2f}")
    
    print(f"\n=== Most economical items (price per 100 kcal) ===")
    economical = output_df.nsmallest(5, 'price_per_100kcal')[['item_name', 'item_type', 'price', 'energy_kcal', 'price_per_100kcal']]
    print(economical.to_string(index=False))
    
    print(f"\n✓ Combined data saved to {output_path}")


def main():
    """Main function."""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    
    menu_path = os.path.join(output_dir, 'menu_items.csv')
    nutrition_path = os.path.join(output_dir, 'nutritional_info.csv')
    output_path = os.path.join(output_dir, 'combined_menu_nutrition.csv')
    
    # Check if input files exist
    if not os.path.exists(menu_path):
        print(f"Error: Menu file not found at {menu_path}")
        return
    
    if not os.path.exists(nutrition_path):
        print(f"Error: Nutritional info file not found at {nutrition_path}")
        return
    
    # Match and save
    match_price_to_nutrition(menu_path, nutrition_path, output_path)


# This block only runs when the script is executed directly
if __name__ == "__main__":
    main()
