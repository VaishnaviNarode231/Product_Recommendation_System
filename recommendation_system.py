import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

# --- 1. Load uploaded CSV data ---
@st.cache_data
def load_uploaded_data(uploaded_file):
    # Caches the uploaded CSV file to avoid reloading it on every rerun
    df = pd.read_csv(uploaded_file)
    return df

# --- 2. Preprocess data to clean and format appropriately ---
def preprocess_data(df):
    # Convert all column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    # Identify product and rating columns dynamically
    product_col = next((col for col in df.columns if 'product' in col or 'name' in col or 'title' in col), None)
    rating_col = next((col for col in df.columns if 'rating' in col or 'review' in col), None)

    # Raise error if required columns are not present
    if product_col is None or rating_col is None:
        raise ValueError("Dataset must have at least 'Product Name' and 'Rating' columns.")

    # Drop rows with missing product or rating values
    df = df.dropna(subset=[product_col, rating_col])
    
    # Rename the columns to standardized names
    df = df.rename(columns={product_col: 'Product Name', rating_col: 'Rating'})

    # Assign sequential User IDs if not already present
    if 'user id' not in df.columns:
        df['User ID'] = range(1, len(df) + 1)

    # Convert ratings to numeric, and remove invalid entries
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Rating'])

    # Encode product names into numeric IDs if not already present
    if 'product id' not in df.columns:
        label_enc = LabelEncoder()
        df['Product ID'] = label_enc.fit_transform(df['Product Name'])

    return df

# --- 3. Create user-item matrix for user-based collaborative filtering ---
def create_user_item_matrix(df):
    # Create matrix with users as rows, products as columns, and ratings as values
    user_item_matrix = df.pivot_table(index='User ID', columns='Product Name', values='Rating', fill_value=0)
    return user_item_matrix, csr_matrix(user_item_matrix.values)  # Sparse matrix for memory efficiency

# --- 4. Compute cosine similarity between users ---
def compute_user_similarity(sparse_matrix):
    return cosine_similarity(sparse_matrix)

# --- 5. Recommend products for a user based on similar users ---
def recommend_products(user_id, user_item_matrix, similarity_matrix, user_list, top_n_users=5, top_n_products=5):
    if user_id not in user_list:
        return []

    user_idx = user_list.index(user_id)
    similarity_scores = similarity_matrix[user_idx]

    # Find top similar users (excluding the user themself)
    similar_users_idx = np.argsort(similarity_scores)[::-1]
    similar_users_idx = [i for i in similar_users_idx if i != user_idx][:top_n_users]
    similar_users = [user_list[i] for i in similar_users_idx]

    # Get products already rated by the user
    user_rated_products = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
    recommended_products = {}

    # For each similar user, find products not rated by current user
    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        unrated_products = similar_user_ratings[similar_user_ratings > 0].drop(user_rated_products, errors='ignore')

        for product_name, rating in unrated_products.items():
            # Take highest rating for the same product across similar users
            if product_name not in recommended_products:
                recommended_products[product_name] = rating
            else:
                recommended_products[product_name] = max(recommended_products[product_name], rating)

    # Sort recommendations by rating
    recommended_products = dict(sorted(recommended_products.items(), key=lambda x: x[1], reverse=True))
    return list(recommended_products.items())[:top_n_products]

# --- 6. Create item-user matrix for product-based collaborative filtering ---
def create_item_user_matrix(df):
    item_user_matrix = df.pivot_table(index='Product Name', columns='User ID', values='Rating', fill_value=0)
    return item_user_matrix, csr_matrix(item_user_matrix.values)

# --- 7. Compute cosine similarity between products ---
def compute_product_similarity(sparse_matrix):
    return cosine_similarity(sparse_matrix)

# --- 8. Recommend similar products based on one product ---
def recommend_similar_products(product_name, item_user_matrix, similarity_matrix, product_list, top_n=5):
    if product_name not in product_list:
        return []

    product_idx = product_list.index(product_name)
    similarity_scores = similarity_matrix[product_idx]

    # Get top similar products (excluding the same product)
    similar_products_idx = np.argsort(similarity_scores)[::-1]
    similar_products_idx = [i for i in similar_products_idx if i != product_idx][:top_n]
    similar_products = [product_list[i] for i in similar_products_idx]

    return similar_products

# --- 9. Handle Cold Start Problem: Recommend top-rated products ---
def get_top_rated_products(df, top_n=5):
    return df.groupby('Product Name')['Rating'].mean().sort_values(ascending=False).head(top_n)

# --- 10. Display product details in UI ---
def display_product_details(product_name, product_rating):
    st.markdown(f"**{product_name}**")
    st.markdown(f"⭐ {product_rating:.2f}/5")

# --- 11. Streamlit UI and App Logic ---
def main():
    st.set_page_config(page_title="🛍️ Amazon Products Recommender", page_icon="🛒", layout="wide")
    st.title("🛒 Personalized Product Recommendations")

    # Sidebar options for uploading file and tuning parameters
    st.sidebar.title("Upload Your Amazon Dataset 📄")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    top_n_users = st.sidebar.slider("Top similar users to consider", 1, 10, 5)
    top_n_products = st.sidebar.slider("Top products to recommend", 1, 10, 5)

    if uploaded_file:
        try:
            with st.spinner('Loading and preprocessing data...'):
                # Load and process data
                df = load_uploaded_data(uploaded_file)
                df = preprocess_data(df)
                user_item_matrix, user_sparse_matrix = create_user_item_matrix(df)
                user_cosine_sim = compute_user_similarity(user_sparse_matrix)

                item_user_matrix, item_sparse_matrix = create_item_user_matrix(df)
                product_cosine_sim = compute_product_similarity(item_sparse_matrix)

            st.success("Data loaded and processed successfully! 🎯")

            # Select a user from available user IDs
            user_list = user_item_matrix.index.tolist()
            selected_user = st.selectbox("Select a User ID", user_list)

            if selected_user:
                st.subheader(f"👥 Top {top_n_users} similar users to **{selected_user}**:")
                user_idx = user_list.index(selected_user)
                similarity_scores = user_cosine_sim[user_idx]

                similar_users_idx = np.argsort(similarity_scores)[::-1]
                similar_users_idx = [i for i in similar_users_idx if i != user_idx][:top_n_users]
                similar_users = [user_list[i] for i in similar_users_idx]

                st.write(similar_users)

                # Recommend products to selected user
                recommendations = recommend_products(
                    selected_user,
                    user_item_matrix,
                    user_cosine_sim,
                    user_list,
                    top_n_users,
                    top_n_products
                )

                st.subheader(f"🎯 Top {top_n_products} product recommendations for **User {selected_user}**:")
                if recommendations:
                    rec_df = pd.DataFrame(recommendations, columns=['Product Name', 'Predicted Rating'])
                    st.table(rec_df)
                else:
                    st.warning("No product recommendations found for this user. Showing top-rated products instead.")
                    top_rated_products = get_top_rated_products(df, top_n=top_n_products)
                    for product_name, rating in top_rated_products.items():
                        display_product_details(product_name, rating)

                # Product-based recommendations
                user_products = user_item_matrix.columns[user_item_matrix.loc[selected_user] > 0]
                if not user_products.empty:
                    selected_product = st.selectbox("Select a Product you've rated for similar products", user_products)

                    if selected_product:
                        similar_products = recommend_similar_products(
                            selected_product,
                            item_user_matrix,
                            product_cosine_sim,
                            list(item_user_matrix.index),
                            top_n=top_n_products
                        )

                        st.subheader(f"🛍️ Products similar to **{selected_product}**:")
                        if similar_products:
                            for prod in similar_products:
                                avg_rating = df[df['Product Name'] == prod]['Rating'].mean()
                                display_product_details(prod, avg_rating)
                        else:
                            st.info("No similar products found.")
                else:
                    st.info("User has not rated any products yet.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a dataset to get started.")

# --- Run the app ---
if __name__ == "__main__":
    main()
