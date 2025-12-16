#!/usr/bin/env python3
"""
Script to parse menu text and extract items with prices into CSV format
"""

import re
import csv
from pathlib import Path


def extract_prices_from_line(line):
    """
    Extract prices from a line in format x.xx or x,xx (< 10€)
    Returns list of floats
    """
    # Match prices like 1.55, 3.05*, 4,30 €*, etc.
    # Look for single digit, followed by . or , and two digits
    pattern = r'(\d)[.,](\d{2})\s*[€*]*'
    matches = re.findall(pattern, line)
    
    if not matches:
        return None
    
    # Convert to float
    prices = []
    for match in matches:
        price = float(f"{match[0]}.{match[1]}")
        if price < 10:  # Only items below 10€
            prices.append(price)
    
    return prices if prices else None


def parse_menu_text(input_file, output_file):
    """
    Parse menu text file and extract items with prices to CSV
    
    Format: 
    - Item name on one line
    - Base price on next line
    - Optional small combo price on following line
    - Optional large combo price on following line
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Clean and filter lines
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]  # Remove empty lines
    
    menu_items = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line contains a price
        prices = extract_prices_from_line(line)
        
        if prices and i > 0:
            # The previous line should be the item name
            item_name = lines[i - 1]
            
            # Skip if the name looks like a header or is too short
            if (len(item_name) > 2 and 
                not item_name.startswith('---') and
                not re.match(r'^[€\d\s,.*]+$', item_name)):  # Skip if it's all numbers/symbols
                
                # Collect all consecutive price lines
                all_prices = prices.copy()
                j = i + 1
                
                # Look ahead for more price lines (up to 2 more)
                while j < len(lines) and len(all_prices) < 3:
                    next_prices = extract_prices_from_line(lines[j])
                    if next_prices:
                        all_prices.extend(next_prices)
                        j += 1
                    else:
                        break
                
                base_price = all_prices[0] if len(all_prices) > 0 else None
                small_combo = all_prices[1] if len(all_prices) > 1 else None
                large_combo = all_prices[2] if len(all_prices) > 2 else None
                
                menu_items.append({
                    'item_name': item_name,
                    'base_price': base_price,
                    'small_combo_price': small_combo,
                    'large_combo_price': large_combo
                })
                
                # Skip the price lines we just processed
                i = j - 1
        
        i += 1
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['item_name', 'base_price', 'small_combo_price', 'large_combo_price'])
        writer.writeheader()
        writer.writerows(menu_items)
    
    print(f"Extracted {len(menu_items)} menu items")
    print(f"Saved to: {output_file}")
    
    return menu_items


if __name__ == "__main__":
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_dir = project_dir / "output"
    
    # Input and output files
    input_file = output_dir / "menu_text.txt"
    output_file = output_dir / "menu_items.csv"
    
    # Parse menu
    items = parse_menu_text(input_file, output_file)
    
    # Display first few items
    print("\n--- First 10 items ---")
    for i, item in enumerate(items[:10], 1):
        print(f"{i}. {item['item_name']}: €{item['base_price']}", end="")
        if item['small_combo_price']:
            print(f" (small combo: €{item['small_combo_price']}", end="")
            if item['large_combo_price']:
                print(f", large combo: €{item['large_combo_price']})", end="")
            else:
                print(")", end="")
        print()
