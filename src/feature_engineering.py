import pandas as pd

def create_features(df):
    # Future price (8% growth)
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * (1.08 ** 5)

    # Amenity count
    df['Amenity_Count'] = df['Amenities'].apply(lambda x: len(str(x).split(',')))

    # Good Investment Rule
    median_price = df['Price_per_SqFt'].median()

    df['Good_Investment'] = (
        (df['Price_per_SqFt'] < median_price) &
        (df['BHK'] >= 2) &
        (df['Availability_Status'] == 'Ready_to_Move')
    ).astype(int)

    return df