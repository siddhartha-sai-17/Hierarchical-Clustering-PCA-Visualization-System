import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import plotly.express as px
import random

# Generate sample data (copied from app.py logic)
def generate_sample_data(n_samples=50):
    np.random.seed(42)
    topics = {
        "Topic A": ["News about A " + str(i) for i in range(10)],
        "Topic B": ["News about B " + str(i) for i in range(10)],
        "Topic C": ["News about C " + str(i) for i in range(10)]
    }
    data = []
    for _ in range(n_samples):
        topic = np.random.choice(list(topics.keys()))
        base = np.random.choice(topics[topic])
        data.append({
            "headline": f"{base} - {random.randint(1, 100)}",
            "topic_ground_truth": topic
        })
    return pd.DataFrame(data)

def test_plots():
    print("Generating data...")
    df = generate_sample_data()
    text_col = "headline"
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    dense_matrix = tfidf_matrix.toarray()
    
    # 1. Test Dendrogram
    print("Testing Dendrogram generation...")
    try:
        Z = linkage(dense_matrix, method='ward', metric='euclidean')
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(
            Z,
            labels=df.index,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax
        )
        plt.title("Test Dendrogram")
        plt.savefig("dendrogram_test.png")
        print("Dendrogram saved to dendrogram_test.png")
    except Exception as e:
        print(f"FAILED to generate Dendrogram: {e}")
        
    # 2. Test Clustering & Plotly
    print("Testing Plotly Scatter generation...")
    try:
        model = AgglomerativeClustering(n_clusters=3)
        labels = model.fit_predict(dense_matrix)
        
        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(dense_matrix)
        
        # DEBUG: PCA Output Verification
        print("DEBUG: PCA shape:", components.shape)
        print("DEBUG: Unique clusters:", np.unique(labels))
        print("DEBUG: Any NaN in PCA:", np.isnan(components).any())
        
        clustered_df = df.copy()
        clustered_df['Cluster'] = labels
        clustered_df['PC1'] = components[:, 0]
        clustered_df['PC2'] = components[:, 1]
        
        fig_pca = px.scatter(
            clustered_df, 
            x="PC1", 
            y="PC2", 
            color=clustered_df['Cluster'].astype(str),
            title="Clustered Articles (2D Projection)",
            labels={
                "PC1": "Principal Component 1",
                "PC2": "Principal Component 2",
                "color": "Cluster"
            },
            color_discrete_sequence=px.colors.qualitative.Prism,
            height=600
        )
        
        # Explicitly force marker visibility
        fig_pca.update_traces(
            marker=dict(
                size=8,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            )
        )
        
        # Fix Dark Theme Background explicitly
        fig_pca.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font=dict(color="white"),
            legend_title_text='Cluster'
        )
        
        # Just writing HTML to verify it generates the object successfully
        fig_pca.write_html("scatter_test_fixed.html")
        print("Plotly scatter saved to scatter_test_fixed.html")
    except Exception as e:
        print(f"FAILED to generate Plotly scatter: {e}")

if __name__ == "__main__":
    test_plots()
