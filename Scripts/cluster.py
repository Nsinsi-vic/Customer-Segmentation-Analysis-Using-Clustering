from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from preprocess import clean_data
import pandas as pd

def cluster_data(df, variance_threshold=0.9):
    """
    Dimensionality reduction + clustering pipeline.
    
    Args:
        df (pd.DataFrame): Preprocessed data
        variance_threshold (float): PCA variance to retain
        
    Returns:
        pd.DataFrame: Data with 'Cluster' labels
    """
    # PCA
    pca = PCA(n_components=variance_threshold)
    reduced_data = pca.fit_transform(df)
    
    # Optimal K via Elbow Method (simplified)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)
    
    return clusters

if __name__ == "__main__":
    cleaned_data = pd.read_csv("../data/processed/cleaned_data.csv")
    clusters = cluster_data(cleaned_data)
    
    # Save results
    final_df = pd.read_csv("../data/raw/customers.csv").iloc[:len(clusters)]
    final_df['Cluster'] = clusters
    final_df.to_csv("../data/results/clustered_customers.csv", index=False)

import logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO)
logging.info("PCA explained variance: %s", pca.explained_variance_ratio_)