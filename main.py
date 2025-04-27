import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import uuid
import os
from typing import Optional
from supabase import create_client, Client
import openai
import re

st.set_page_config(
    page_title="Social Media Analytics",
    page_icon="📊",
    layout="wide"
)

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()
openai.api_key = st.secrets["OPENAI_API_KEY"]

## --------------------------
## AUTHENTICATION FUNCTIONS
## --------------------------

def login(email: str, password: str):
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if not response.user:
            st.error("Login failed")
            return False
        return True
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def sign_up(email: str, password: str, full_name: str):
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            supabase.table("profiles").insert({
                "id": response.user.id,
                "email": email,
                "full_name": full_name
            }).execute()
            return True
        return False
    except Exception as e:
        st.error(f"Sign up error: {str(e)}")
        return False

def logout():
    supabase.auth.sign_out()
    st.session_state.clear()
    st.rerun()

def check_auth():
    if "user" not in st.session_state:
        try:
            user = supabase.auth.get_user()
            if user:
                st.session_state.user = user
                return True
        except:
            pass
        return False
    return True

def show_auth():
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Login"):
                if login(email, password):
                    st.rerun()
    
    with tab2:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            full_name = st.text_input("Full Name", key="signup_name")
            if st.form_submit_button("Create Account"):
                if sign_up(email, password, full_name):
                    st.success("Account created! Please login.")
                    st.rerun()

## --------------------------
## DATA MANAGEMENT FUNCTIONS
## --------------------------

def save_uploaded_file(user_id: str, account_id: int, data_type: str, file):
    try:
        file_ext = os.path.splitext(file.name)[1]
        file_name = f"{user_id}/{account_id}/{uuid.uuid4()}{file_ext}"
        
        res = supabase.storage.from_("analytics-uploads").upload(file_name, file.getvalue())
        
        if res:
            supabase.table("analytics_uploads").insert({
                "user_id": user_id,
                "account_id": account_id,
                "data_type": data_type,
                "file_name": file.name,
                "file_path": file_name
            }).execute()
            return True
        return False
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False

def get_user_accounts(user_id: str):
    return supabase.table("social_accounts").select("*").eq("user_id", user_id).execute().data

def get_account_uploads(user_id: str, account_id: int):
    return supabase.table("analytics_uploads").select("*").eq("user_id", user_id).eq("account_id", account_id).execute().data

def get_upload_data(file_path: str):
    try:
        res = supabase.storage.from_("analytics-uploads").download(file_path)
        return pd.read_csv(res)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

## --------------------------
## DATA PROCESSING FUNCTIONS
## --------------------------

def process_twitter_account_data(df):
    # Clean and transform Twitter account overview data
    df['Date'] = pd.to_datetime(df['Date'], format='%a, %b %d, %Y')
    df = df.sort_values('Date')
    df['followers'] = (df['New follows'] - df['Unfollows']).cumsum()
    return df

def process_twitter_post_data(df):
    # Clean and transform Twitter post data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['tweet_length'] = df['Post text'].apply(lambda x: len(str(x)))
    
    # Extract mentions and hashtags
    def extract_mentions(text):
        return re.findall(r'@(\w+)', str(text))
    
    def extract_hashtags(text):
        return re.findall(r'#(\w+)', str(text))
    
    df['mentions'] = df['Post text'].apply(extract_mentions)
    df['hashtags'] = df['Post text'].apply(extract_hashtags)
    df['sentiment'] = df['Post text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    return df

def process_instagram_post_data(df):
    # Clean and transform Instagram post data
    df['Publish time'] = pd.to_datetime(df['Publish time'])
    df = df.sort_values('Publish time')
    df['Post Length'] = df['Description'].str.len().fillna(0)
    
    # Calculate engagement
    engagement_columns = ['Likes', 'Shares', 'Comments', 'Saves']
    available_engagement = [col for col in engagement_columns if col in df.columns]
    df['Engagement'] = df[available_engagement].sum(axis=1)
    
    if 'Reach' in df.columns:
        df['Engagement Rate'] = (df['Engagement'] / df['Reach']) * 100
    
    return df

def process_instagram_story_data(df):
    # Clean and transform Instagram story data
    df['Publish time'] = pd.to_datetime(df['Publish time'])
    df = df.sort_values('Publish time')
    
    # Calculate engagement
    engagement_columns = ['Likes', 'Replies', 'Sticker taps']
    available_engagement = [col for col in engagement_columns if col in df.columns]
    df['Engagement'] = df[available_engagement].sum(axis=1)
    
    if 'Reach' in df.columns:
        df['Engagement Rate'] = (df['Engagement'] / df['Reach']) * 100
    
    return df

## --------------------------
## ANALYSIS FUNCTIONS
## --------------------------

def generate_summary(text, platform):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"""You are a social media analytics expert. Provide a detailed, insightful summary (500-700 words) analyzing {platform} performance. 
                Include: 
                1. Key performance metrics 
                2. Content type analysis 
                3. Engagement patterns 
                4. Optimal posting times 
                5. Content characteristics of top performers 
                6. Actionable recommendations
                7. Sentiment analysis insights
                Use professional but accessible language."""},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def twitter_account_analysis(df):
    st.header("Twitter Account Overview")
    
    # Calculate metrics
    df = process_twitter_account_data(df)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Days", len(df))
    col2.metric("Current Followers", f"{df['followers'].iloc[-1]:,}")
    col3.metric("Avg. Daily Growth", f"{(df['New follows'] - df['Unfollows']).mean():.1f}")
    
    # Visualizations
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['followers'], 
                   name='Followers', line=dict(color='#1DA1F2')))
    fig.update_layout(title="Follower Growth Over Time",
                    xaxis_title="Date", yaxis_title="Followers")
    st.plotly_chart(fig, use_container_width=True)
    
    # Engagement metrics
    fig = go.Figure()
    metrics = ['Impressions', 'Engagements', 'Likes']
    colors = ['#17BF63', '#E0245E', '#FFAD1F']
    
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df[metric],
            name=metric,
            line=dict(color=color)
        ))
    
    fig.update_layout(title="Engagement Metrics Over Time")
    st.plotly_chart(fig, use_container_width=True)

def twitter_post_analysis(df):
    st.header("Twitter Post Analysis")
    
    # Process data
    df = process_twitter_post_data(df)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", len(df))
    if 'Impressions' in df.columns:
        col2.metric("Avg. Impressions", f"{df['Impressions'].mean():,.0f}")
    if 'Likes' in df.columns:
        col3.metric("Avg. Likes", f"{df['Likes'].mean():.1f}")
    
    # Top performing posts
    st.subheader("Top Performing Posts")
    if 'Impressions' in df.columns:
        top_posts = df.sort_values('Impressions', ascending=False).head(5)
        for idx, row in top_posts.iterrows():
            with st.expander(f"{row['Date'].strftime('%Y-%m-%d')} | Impressions: {row['Impressions']:,} | Likes: {row['Likes']}"):
                st.write(f"**Post:** {row['Post text'][:200]}...")
                st.markdown(f"[View Post]({row['Link']})")
    
    # Hashtag analysis
    if 'hashtags' in df.columns:
        st.subheader("Hashtag Performance")
        hashtags = pd.Series([tag for sublist in df['hashtags'] for tag in sublist]).value_counts()
        if not hashtags.empty:
            fig = px.bar(hashtags.head(10), title="Top 10 Hashtags")
            st.plotly_chart(fig, use_container_width=True)

def instagram_post_analysis(df):
    st.header("Instagram Post Analysis")
    
    # Process data
    df = process_instagram_post_data(df)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", len(df))
    if 'Reach' in df.columns:
        col2.metric("Avg. Reach", f"{df['Reach'].mean():,.0f}")
    if 'Engagement Rate' in df.columns:
        col3.metric("Avg. Engagement Rate", f"{df['Engagement Rate'].mean():.2f}%")
    
    # Post type performance
    if 'Post type' in df.columns:
        st.subheader("Performance by Post Type")
        post_type_stats = df.groupby('Post type').agg({
            'Reach': 'mean',
            'Engagement Rate': 'mean',
            'Engagement': 'mean'
        }).reset_index()
        
        fig = px.bar(post_type_stats, x='Post type', y=['Reach', 'Engagement Rate'],
                    barmode='group', title='Performance by Post Type')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performing posts
    st.subheader("Top Performing Posts")
    if 'Reach' in df.columns:
        top_posts = df.sort_values('Reach', ascending=False).head(5)
        for idx, row in top_posts.iterrows():
            with st.expander(f"{row['Publish time'].strftime('%Y-%m-%d')} | Reach: {row['Reach']:,} | Engagement: {row['Engagement']}"):
                st.write(f"**Description:** {row['Description'][:200] if pd.notna(row['Description']) else 'No description'}")
                st.markdown(f"[View Post]({row['Permalink']})")

def instagram_story_analysis(df):
    st.header("Instagram Story Analysis")
    
    # Process data
    df = process_instagram_story_data(df)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stories", len(df))
    if 'Reach' in df.columns:
        col2.metric("Avg. Reach", f"{df['Reach'].mean():,.0f}")
    if 'Engagement Rate' in df.columns:
        col3.metric("Avg. Engagement Rate", f"{df['Engagement Rate'].mean():.2f}%")
    
    # Duration analysis
    if 'Duration (sec)' in df.columns:
        st.subheader("Story Duration Analysis")
        fig = px.scatter(df, x='Duration (sec)', y='Engagement Rate',
                        trendline="lowess", title='Duration vs Engagement')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performing stories
    st.subheader("Top Performing Stories")
    if 'Reach' in df.columns:
        top_stories = df.sort_values('Reach', ascending=False).head(5)
        for idx, row in top_stories.iterrows():
            with st.expander(f"{row['Publish time'].strftime('%Y-%m-%d')} | Reach: {row['Reach']:,} | Engagement: {row['Engagement']}"):
                if 'Description' in df.columns:
                    st.write(f"**Description:** {row['Description'][:200] if pd.notna(row['Description']) else 'No description'}")
                st.markdown(f"[View Story]({row['Permalink']})")

## --------------------------
## COMPARISON FUNCTIONS
## --------------------------

def compare_twitter_accounts(account1_data, account2_data, account1_name, account2_name):
    st.header("Twitter Account Comparison")
    
    # Ensure we have account overview data
    if 'followers' not in account1_data.columns or 'followers' not in account2_data.columns:
        st.warning("Account overview data required for comparison")
        return
    
    # Follower growth comparison
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=account1_data['Date'], y=account1_data['followers'],
                           name=account1_name, line=dict(color='#1DA1F2')))
    fig.add_trace(go.Scatter(x=account2_data['Date'], y=account2_data['followers'],
                           name=account2_name, line=dict(color='#794BC4')))
    fig.update_layout(title="Follower Growth Comparison",
                    xaxis_title="Date", yaxis_title="Followers")
    st.plotly_chart(fig, use_container_width=True)
    
    # Engagement comparison
    metrics = ['Impressions', 'Engagements', 'Likes']
    comparison_data = {
        'Metric': metrics,
        account1_name: [account1_data[m].mean() for m in metrics],
        account2_name: [account2_data[m].mean() for m in metrics]
    }
    
    fig = px.bar(pd.DataFrame(comparison_data), x='Metric', y=[account1_name, account2_name],
                barmode='group', title='Average Engagement Comparison')
    st.plotly_chart(fig, use_container_width=True)

def compare_instagram_accounts(account1_data, account2_data, account1_name, account2_name):
    st.header("Instagram Account Comparison")
    
    # Engagement comparison
    metrics = []
    for metric in ['Reach', 'Engagement', 'Engagement Rate']:
        if metric in account1_data.columns and metric in account2_data.columns:
            metrics.append(metric)
    
    if metrics:
        comparison_data = {
            'Metric': metrics,
            account1_name: [account1_data[m].mean() for m in metrics],
            account2_name: [account2_data[m].mean() for m in metrics]
        }
        
        fig = px.bar(pd.DataFrame(comparison_data), x='Metric', y=[account1_name, account2_name],
                    barmode='group', title='Performance Comparison')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No common metrics available for comparison")

## --------------------------
## MAIN APP LAYOUT
## --------------------------

def main_app():
    st.title("Social Media Analytics Dashboard")
    user = st.session_state.user
    user_id = user.user.id
    
    with st.sidebar:
        st.header(f"Welcome, {user.user.email}")
        if st.button("Logout"):
            logout()
        
        with st.expander("Add Social Account"):
            with st.form("new_account_form"):
                platform = st.selectbox("Platform", ["twitter", "instagram"])
                username = st.text_input("Username")
                display_name = st.text_input("Display Name")
                
                if st.form_submit_button("Save Account"):
                    supabase.table("social_accounts").insert({
                        "user_id": user_id,
                        "platform": platform,
                        "username": username,
                        "display_name": display_name
                    }).execute()
                    st.success("Account added!")
                    st.rerun()
    
    tab1, tab2 = st.tabs(["Upload & Analyze", "Compare Accounts"])
    
    with tab1:
        st.header("Upload Analytics Data")
        accounts = get_user_accounts(user_id)
        
        if accounts:
            account = st.selectbox("Select Account", accounts, 
                                 format_func=lambda x: f"{x['display_name']} ({x['platform']})")
            
            # Dynamic data type selection based on platform
            if account["platform"] == "twitter":
                data_type = st.selectbox("Data Type", 
                                       ["twitter_account_overview", "twitter_post_data"])
            else:
                data_type = st.selectbox("Data Type", 
                                       ["instagram_posts", "instagram_stories"])
            
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
            
            if uploaded_file and st.button("Upload & Analyze"):
                with st.spinner("Processing..."):
                    if save_uploaded_file(user_id, account["id"], data_type, uploaded_file):
                        st.success("File uploaded successfully!")
                        df = get_upload_data(f"{user_id}/{account['id']}/{uploaded_file.name}")
                        
                        if df is not None:
                            st.subheader("Analysis Results")
                            if data_type == "twitter_account_overview":
                                twitter_account_analysis(df)
                            elif data_type == "twitter_post_data":
                                twitter_post_analysis(df)
                            elif data_type == "instagram_posts":
                                instagram_post_analysis(df)
                            elif data_type == "instagram_stories":
                                instagram_story_analysis(df)
    
    with tab2:
        st.header("Compare Accounts")
        accounts = get_user_accounts(user_id)
        
        if len(accounts) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                account1 = st.selectbox("First Account", accounts, 
                                      format_func=lambda x: f"{x['display_name']} ({x['platform']})", 
                                      key="account1")
            with col2:
                account2 = st.selectbox("Second Account", accounts, 
                                      format_func=lambda x: f"{x['display_name']} ({x['platform']})", 
                                      key="account2")
            
            if account1["id"] != account2["id"] and account1["platform"] == account2["platform"]:
                uploads1 = get_account_uploads(user_id, account1["id"])
                uploads2 = get_account_uploads(user_id, account2["id"])
                
                if uploads1 and uploads2:
                    # For comparison, we'll use account overview data for Twitter and post data for Instagram
                    if account1["platform"] == "twitter":
                        st.info("For Twitter comparison, please select account overview data")
                        upload1 = st.selectbox("First Data Set", 
                                             [u for u in uploads1 if u["data_type"] == "twitter_account_overview"],
                                             format_func=lambda x: x["file_name"], 
                                             key="upload1")
                        upload2 = st.selectbox("Second Data Set", 
                                             [u for u in uploads2 if u["data_type"] == "twitter_account_overview"],
                                             format_func=lambda x: x["file_name"], 
                                             key="upload2")
                    else:
                        st.info("For Instagram comparison, please select post data")
                        upload1 = st.selectbox("First Data Set", 
                                             [u for u in uploads1 if u["data_type"] == "instagram_posts"],
                                             format_func=lambda x: x["file_name"], 
                                             key="upload1")
                        upload2 = st.selectbox("Second Data Set", 
                                             [u for u in uploads2 if u["data_type"] == "instagram_posts"],
                                             format_func=lambda x: x["file_name"], 
                                             key="upload2")
                    
                    if st.button("Compare"):
                        df1 = get_upload_data(upload1["file_path"])
                        df2 = get_upload_data(upload2["file_path"])
                        
                        if df1 is not None and df2 is not None:
                            if account1["platform"] == "twitter":
                                compare_twitter_accounts(
                                    process_twitter_account_data(df1), 
                                    process_twitter_account_data(df2),
                                    account1["display_name"], 
                                    account2["display_name"]
                                )
                            else:
                                compare_instagram_accounts(
                                    process_instagram_post_data(df1),
                                    process_instagram_post_data(df2),
                                    account1["display_name"], 
                                    account2["display_name"]
                                )

## --------------------------
## APP ENTRY POINT
## --------------------------

def app():
    if not check_auth():
        show_auth()
    else:
        main_app()

if __name__ == "__main__":
    app()
