
# =========================================================
# 1) IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# 2) PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4B0082; font-weight: 700; margin-bottom: 0px; }
    .sub-text { font-size: 1.1rem; color: #555; margin-bottom: 20px; }
    .metric-box { padding: 15px; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid #4B0082; margin-bottom: 10px; }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">🟣 News Topic Discovery Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Hierarchical Clustering & PCA Visualization System</div>', unsafe_allow_html=True)
st.markdown("---")

# =========================================================
# 3) UTILITY FUNCTIONS & STATE MANAGEMENT
# =========================================================

# Initialize Session State for Clustering Results
if "cluster_results" not in st.session_state:
    st.session_state.cluster_results = None

@st.cache_data
def generate_sample_data():
    """Generates synthetic news data."""
    headlines = [
        "Stock market hits record high as tech rally continues",
        "Federal Reserve signals interest rate cuts next quarter",
        "Oil prices surge amidst geopolitical tensions",
        "Major tech company announces breakthrough in AI",
        "Quarterly earnings report shows strong profit growth",
        "New trade policy impacts global supply chains",
        "Inflation rates drop slightly, easing consumer concerns",
        "Startup raises millions in Series B funding round",
        "Cryptocurrency market sees volatile trading day",
        "European markets close mixed as uncertainty looms",
        "NASA launches new mission to study Mars atmosphere",
        "Scientists discover new species in Amazon rainforest",
        "Global warming concerns rise with record temperatures",
        "New vaccine shows promise in early clinical trials",
        "Electric vehicle sales surpass traditional cars in Norway",
        "SpaceX successfully lands booster rocket on drone ship",
        "Breakthrough in renewable energy efficiency announced",
        "Ocean cleanup project removes tons of plastic waste",
        "Astronomers capture image of distant black hole",
        "Study links mediterranean diet to longer lifespan",
        "Local sports team wins championship after decades",
        "Olympic games preparation enters final stages",
        "Star athlete announces retirement from professional sports",
        "New stadium construction project approved by city council",
        "World Cup finals draw record global viewership",
        "Tennis champion seeks comeback after injury break",
        "Marathon runner breaks world record in Berlin",
        "Formula 1 season finale ends in dramatic fashion",
        "Basketball league announces expansion teams",
        "Soccer transfer market sees record-breaking deals"
    ]
    df = pd.DataFrame({"headline": headlines})
    # Replicate for density
    return pd.concat([df] * 6, ignore_index=True)

@st.cache_data
def load_data(uploaded_file):
    """Robust CSV loader with encoding fallback."""
    if uploaded_file is not None:
        encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                return df
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        return None
    return generate_sample_data()

def detect_text_column(df):
    """Detects text column by average length."""
    max_avg_len = 0
    best_col = None
    cols = df.select_dtypes(include=['object']).columns
    for col in cols:
        avg_len = df[col].astype(str).str.len().mean()
        if avg_len > max_avg_len:
            max_avg_len = avg_len
            best_col = col
    return best_col

@st.cache_data
def perform_vectorization(texts, max_features, stop_words, ngram_range):
    """Cached TF-IDF Vectorization."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_df=0.7
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# =========================================================
# 4) DATA LOADING SECTION
# =========================================================
with st.sidebar:
    st.header("📂 Data Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    # Load
    df = load_data(uploaded_file)
    if df is None:
        st.error("Unable to decode file. Please upload a valid CSV.")
        st.stop()
    elif uploaded_file is not None:
        st.toast("File loaded successfully!", icon="✅")
        
    df = df.dropna()
    
    # Column Selection
    guessed_col = detect_text_column(df)
    all_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not all_cols:
        st.error("Dataset has no text columns!")
        st.stop()
        
    default_idx = all_cols.index(guessed_col) if guessed_col in all_cols else 0
    text_col = st.selectbox("Select Text Column", all_cols, index=default_idx)
    
    if len(df) < 5:
        st.error("Dataset too small (need 5+ rows).")
        st.stop()
        
    st.success(f"Loaded {len(df)} articles.")

# Preview
with st.expander("👀 Raw Data Preview"):
    st.dataframe(df.head())

# =========================================================
# 5) VECTORIZATION & CLUSTERING CONFIG
# =========================================================
with st.sidebar:
    st.divider()
    st.header("⚙️ Configuration")
    
    # Vectorization Params
    max_features = st.slider("Max Features", 100, 2000, 1000)
    remove_stopwords = st.checkbox("Remove Stopwords", value=True)
    ngram_choice = st.selectbox("N-gram Range", ["Unigrams (1,1)", "Bigrams (2,2)"])
    ngram_map = {"Unigrams (1,1)": (1,1), "Bigrams (2,2)": (2,2)}
    
    st.divider()
    
    # Clustering Params
    linkage_method = st.selectbox("Linkage", ["ward", "complete", "average"])
    num_clusters = st.number_input("Number of Clusters (k)", 2, 20, 3)

# =========================================================
# 6) DENDROGRAM (HIERARCHY)
# =========================================================
st.subheader("1. Hierarchical Inspection (Dendrogram)")
col_d1, col_d2 = st.columns([1, 3])

with col_d1:
    st.info("Visualize topic hierarchy before clustering.")
    if st.button("🟦 Generate Dendrogram"):
        with st.spinner("Generating..."):
            try:
                # Subset for speed/clarity
                subset = df.sample(n=min(len(df), 50), random_state=42)
                sub_texts = subset[text_col].fillna("").tolist()
                
                # Vectorize Subset
                stop_bg = 'english' if remove_stopwords else None
                X_sub, _ = perform_vectorization(sub_texts, max_features, stop_bg, ngram_map[ngram_choice])
                
                # Linkage (Ward requires Euclidean)
                Z = linkage(X_sub.toarray(), method=linkage_method, metric="euclidean")
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(Z, ax=ax, no_labels=True, leaf_rotation=90)
                ax.set_title(f"Dendrogram ({linkage_method})")
                ax.set_xlabel("Sample Articles")
                ax.set_ylabel("Distance")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Dendrogram Error: {e}")

# =========================================================
# 7) MAIN CLUSTERING PIPELINE
# =========================================================
st.divider()
st.subheader("2. Cluster Application & Visualization")

if st.button("🟩 Apply Clustering"):
    with st.spinner("Processing full dataset..."):
        try:
            # 1. Prepare
            texts = df[text_col].fillna("").astype(str).tolist()
            stop_val = 'english' if remove_stopwords else None
            
            # 2. Vectorize
            X, vectorizer = perform_vectorization(
                texts, max_features, stop_val, ngram_map[ngram_choice]
            )
            
            # 3. Dense Conversion (CRITICAL)
            X_dense = X.toarray() if hasattr(X, "toarray") else X
            
            # 4. Clustering
            model = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage=linkage_method,
                metric="euclidean"
            )
            labels = model.fit_predict(X_dense)
            
            # 5. Store in Session State
            st.session_state.cluster_results = {
                "X": X_dense,
                "labels": labels,
                "vectorizer": vectorizer,
                "texts": texts,
                "k": num_clusters
            }
            st.success(f"Clustering Complete! Found {num_clusters} clusters.")
            
        except Exception as e:
            st.error(f"Clustering Failed: {e}")
            st.stop()

# =========================================================
# 8) VISUALIZATION (PCA) & RESULTS
# =========================================================
results = st.session_state.cluster_results

if results:
    X_dense = results["X"]
    labels = results["labels"]
    texts = results["texts"]
    
    # --- PCA VISUALIZATION ---
    try:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_dense)
        
        # VALIDATE SHAPES
        if X_pca.shape[0] != len(labels):
            st.error("Fatal Error: PCA/Label Mismatch.")
            st.stop()
            
        # CREATE DATAFRAME
        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": labels.astype(str), # String for categorical color
            "Snippet": [t[:100] + "..." for t in texts]
        })
        
        if pca_df.empty:
            st.error("PCA Dataframe is empty.")
            st.stop()
            
        # PLOT
        fig = px.scatter(
            pca_df, 
            x="PC1", 
            y="PC2", 
            color="Cluster", 
            title=f"PCA Cluster Visualization (k={results['k']})",
            hover_data=["Snippet"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=600
        )
        
        # VISIBILITY GUARANTEES
        fig.update_traces(marker=dict(size=9, opacity=0.95, line=dict(width=0.5, color='black')))
        fig.update_layout(template="plotly_white", plot_bgcolor="white")
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"PCA Visualization Error: {e}")

    # --- SUMMARY METRICS & TABLE ---
    st.subheader("3. Cluster Analysis")
    col_sum1, col_sum2 = st.columns([2, 1])
    
    # Calculate Insights
    summary_list = []
    feat_names = results["vectorizer"].get_feature_names_out()
    
    with col_sum1:
        st.markdown("#### 🧠 Business Insights")
        for cid in range(results['k']):
            mask = labels == cid
            count = mask.sum()
            if count == 0: continue
            
            # Top Keywords (Centroid)
            cluster_vecs = X_dense[mask]
            centroid = cluster_vecs.mean(axis=0)
            top_idx = centroid.argsort()[::-1][:10]
            keywords = feat_names[top_idx]
            
            # Representative
            sims = cosine_similarity(cluster_vecs, centroid.reshape(1,-1))
            best_local_idx = np.argmax(sims)
            best_global_idx = np.where(mask)[0][best_local_idx]
            snippet = texts[best_global_idx][:200] + "..."
            
            summary_list.append({
                "ID": cid,
                "Count": count,
                "Keywords": ", ".join(keywords),
                "Representative": snippet
            })
            
            # Insight Text
            st.markdown(f"**🟣 Cluster {cid} ({count} articles)**")
            st.write(f"Key Themes: **{', '.join(keywords[:4])}**")
            st.caption(f"Example: \"{snippet}\"")
            st.divider()

    with col_sum2:
        st.markdown("#### 📈 Metrics")
        if results['k'] > 1:
            score = silhouette_score(X_dense, labels)
            st.metric("Silhouette Score", f"{score:.4f}")
            if score > 0.5: st.success("Strong Structure")
            elif score > 0.2: st.info("Moderate Structure")
            else: st.warning("Weak Structure")
    
    st.markdown("#### 📑 Summary Table")
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)

# =========================================================
# 9) FOOTER
# =========================================================
st.markdown("---")
st.info("💡 **Tip:** Use the Dendrogram to estimate the optimal number of clusters before applying flat clustering.")
