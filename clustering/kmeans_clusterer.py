import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

class ClusteringEngine:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
        # We must scale features before K-Means
        self.scaler = StandardScaler()
        # Features required by the paper
        self.feature_cols = ['Word_Count', 'Sentence_Length', 'Complex_Word_Percentage', 'Average_Word_Length']
        
        # Ensure evaluation directory exists for saving plots
        os.makedirs("evaluation", exist_ok=True)

    def generate_elbow_plot(self, df: pd.DataFrame, max_k: int = 10):
        """
        Calculates WCSS (Within-Cluster Sum of Square) for K=1 to max_k
        and saves the Elbow Plot to disk.
        """
        # Ensure we have data
        if df.empty or len(df) < max_k:
            print("Dataset too small for a meaningful elbow plot.")
            return

        # 1. Extract and Scale features
        X = df[self.feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Calculate WCSS for different K values
        wcss = []
        for i in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
        # 3. Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
        plt.title('The Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('WCSS (Inertia)')
        plt.grid(True)
        plt.savefig("evaluation/elbow_plot.png")
        plt.close()
        print("Elbow plot saved to evaluation/elbow_plot.png")

    def apply_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies K-Means clustering and returns the dataframe with Cluster Labels.
        """
        if df.empty:
            return df
            
        X = df[self.feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        # Predict clusters
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Create a new column in our dataframe
        df['Cluster_Label'] = clusters
        return df

    def generate_cluster_plot(self, df: pd.DataFrame):
        """
        Generates a 2D scatter plot of the clusters using the first two features.
        """
        if 'Cluster_Label' not in df.columns:
            print("No cluster labels found. Run apply_clustering first.")
            return
            
        plt.figure(figsize=(10, 6))
        # Using Sentence Length and Complex Word Percentage as axes for visualization
        sns.scatterplot(
            data=df, 
            x='Sentence_Length', 
            y='Complex_Word_Percentage', 
            hue='Cluster_Label', 
            palette='viridis', 
            s=100
        )
        plt.title('K-Means Clusters of Software Requirements (K=3)')
        plt.xlabel('Sentence Length')
        plt.ylabel('Complex Word Percentage')
        plt.grid(True)
        plt.savefig("evaluation/cluster_plot.png")
        plt.close()
        print("Cluster plot saved to evaluation/cluster_plot.png")

# Simple test if run directly
if __name__ == "__main__":
    # Create mock data mimicking Step 3 output to test the script
    mock_data = {
        'Word_Count': [5, 25, 8, 30, 4],
        'Sentence_Length': [35, 150, 45, 180, 20],
        'Complex_Word_Percentage': [0.0, 30.0, 10.0, 40.0, 0.0],
        'Average_Word_Length': [5.0, 7.2, 5.5, 8.1, 4.5]
    }
    df_mock = pd.DataFrame(mock_data)
    
    engine = ClusteringEngine(n_clusters=3)
    engine.generate_elbow_plot(df_mock, max_k=4) # max_k 4 because we only have 5 rows
    df_clustered = engine.apply_clustering(df_mock)
    engine.generate_cluster_plot(df_clustered)
    print("\nClustered DataFrame Preview:")
    print(df_clustered[['Sentence_Length', 'Complex_Word_Percentage', 'Cluster_Label']])
