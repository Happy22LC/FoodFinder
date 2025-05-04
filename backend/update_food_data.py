import json
import torchvision
from pathlib import Path

def get_food101_classes():
    """Get Food101 class names"""
    dataset = torchvision.datasets.Food101(
        root='./data',
        split='train',
        download=True
    )
    return dataset.classes

def generate_food_descriptions():
    """Generate comprehensive food descriptions"""
    classes = get_food101_classes()
    
    # Base template for cuisine categories
    cuisine_categories = {
        'italian': {
            'description': 'Italian cuisine emphasizes fresh, simple ingredients and traditional cooking methods.',
            'dishes': ['pizza', 'lasagna', 'spaghetti_carbonara', 'risotto']
        },
        'asian': {
            'description': 'Asian cuisines are diverse, featuring rice, noodles, and various cooking techniques.',
            'dishes': ['sushi', 'ramen', 'pad_thai', 'spring_roll']
        },
        'american': {
            'description': 'American cuisine is diverse and multicultural, often featuring comfort foods and fusion dishes.',
            'dishes': ['hamburger', 'hot_dog', 'fried_chicken', 'apple_pie']
        },
        'mediterranean': {
            'description': 'Mediterranean cuisine emphasizes fresh vegetables, olive oil, and seafood.',
            'dishes': ['greek_salad', 'hummus', 'falafel']
        },
        'french': {
            'description': 'French cuisine is known for its refined techniques and rich flavors.',
            'dishes': ['french_onion_soup', 'macarons', 'croque_madame']
        }
    }

    food_data = {}
    
    # Generate descriptions for each class
    for food_class in classes:
        # Determine cuisine category
        cuisine_type = 'international'
        cuisine_desc = 'A delicious dish from international cuisine.'
        
        for cuisine, info in cuisine_categories.items():
            if food_class in info['dishes']:
                cuisine_type = cuisine
                cuisine_desc = info['description']
                break
        
        # Generate food description
        food_desc = f"A popular dish known as {food_class.replace('_', ' ')}. "
        food_desc += "This dish is prepared with traditional ingredients and methods."
        
        food_data[food_class] = {
            'description': food_desc,
            'cuisine': cuisine_type.capitalize(),
            'cuisine_description': cuisine_desc
        }
    
    return food_data

def main():
    """Generate and save food data"""
    print("Generating food data...")
    food_data = generate_food_descriptions()
    
    # Save to JSON file
    output_path = Path(__file__).parent / 'food_data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(food_data, f, indent=4)
    
    print(f"Food data saved to {output_path}")
    print(f"Generated data for {len(food_data)} food classes")

if __name__ == '__main__':
    main() 