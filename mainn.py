from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

df=pd.read_csv(r"C:\Users\MY\Desktop\Capstone\skincare_preprocessed.csv")

lst=df['Product_new']

# Step 1: TF-IDF Vectorization on 'title'
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Product_new'])

# Step 2: Normalize numerical features ('price', 'rating')
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[['Price', 'Rating']])

# Step 3: Encode categorical feature ('product_type')
encoder = OneHotEncoder()
categorical_features = encoder.fit_transform(df[['Product Type']])

# Step 4: Combine all features into a single feature matrix
combined_features = hstack([tfidf_matrix, numeric_features, categorical_features])

# Step 5: Dimensionality Reduction using PCA
pca = PCA(n_components=500)  # Reduce to 50 dimensions (adjust based on variance explained)
reduced_features = pca.fit_transform(combined_features.toarray())

# Step 6: Clustering with KMeans
num_clusters = 10  # Adjust based on dataset size and variety
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(reduced_features)

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# Enhanced Recommendation System Function with Clustering
def recommend_products_with_clustering(product_title, num_recommendations=10):
    try:
        # Find the closest match for the product title using fuzzy matching
        #closest_match = process.extractOne(product_title, df['Product_new'])[0]
        product_idx = df[df['Product_new'] == product_title].index[0]
        
        # Find the cluster of the queried product
        product_cluster = df.loc[product_idx, 'Cluster']
        
        # Filter products in the same cluster
        cluster_products = df[df['Cluster'] == product_cluster]
        
        # Compute cosine similarity within the cluster
        cluster_features = reduced_features[df['Cluster'] == product_cluster]
        query_features = reduced_features[product_idx].reshape(1, -1)
        cluster_sim_scores = cosine_similarity(query_features, cluster_features)
        
        # Rank products within the cluster by similarity
        sim_scores = list(enumerate(cluster_sim_scores[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top N recommendations (excluding the queried product)
        recommended_indices = [cluster_products.index[idx] for idx, score in sim_scores if idx != product_idx][:num_recommendations]
        
        # Return the recommended products
        return df.loc[recommended_indices,['Product', 'Price', 'Rating', 'Brand']]
    except IndexError:
        return "Product title not found in the dataset."
