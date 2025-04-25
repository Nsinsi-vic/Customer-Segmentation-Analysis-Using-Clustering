import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(raw_df):
    """
    Handles missing values, outliers, and feature scaling.
    
    Args:
        raw_df (pd.DataFrame): Raw input data
        
    Returns:
        pd.DataFrame: Scaled and cleaned data
    """
    # Handle missing values
    df = raw_df.dropna()
    
    # Remove outliers (e.g., spending > 99th percentile)
    df = df[df['Total_Spending'] < df['Total_Spending'].quantile(0.99)]
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include='number'))
    
    return pd.DataFrame(scaled_data, columns=df.select_dtypes(include='number').columns)

if __name__ == "__main__":
    raw_data = pd.read_csv("../data/raw/customers.csv")
    processed_data = clean_data(raw_data)
    processed_data.to_csv("../data/processed/cleaned_data.csv", index=False)

import logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO)
logging.info("PCA explained variance: %s", pca.explained_variance_ratio_)