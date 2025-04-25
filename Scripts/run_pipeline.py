import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

def main():
    print("🚀 Starting customer segmentation pipeline...")
    
    # 1. Load data
    data_path = os.path.join('data', 'raw', 'customers.csv')
    print(f"📂 Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Successfully loaded {len(df)} records")
    except Exception as e:
        print(f"❌ Failed to load data: {str(e)}")
        return

    # 2. Preprocessing
    print("🔧 Preprocessing data...")
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 3. Dimensionality Reduction
    print("🎛️ Reducing dimensions with PCA...")
    pca = PCA(n_components=0.9)
    reduced_data = pca.fit_transform(df[numeric_cols])
    print(f"📉 Reduced to {reduced_data.shape[1]} principal components")

    # 4. Clustering
    print("🔮 Running clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(reduced_data)

    # 5. Save results
    output_path = os.path.join('data', 'processed', 'segmented_customers.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Results saved to {output_path}")
    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()