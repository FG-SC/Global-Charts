import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import uuid
import os
from typing import Optional, Dict, List
from supabase import create_client, Client
import openai
import re
from io import StringIO, BytesIO
from textblob import TextBlob

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
    
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test connection
        client.auth.get_user()
        return client
    except Exception as e:
        st.error(f"Supabase initialization failed: {str(e)}")
        return None

supabase = init_supabase()
if supabase is None:
    st.error("Failed to initialize Supabase client. Check your credentials.")
    st.stop()

openai.api_key = st.secrets["OPENAI_API_KEY"]

# --------------------------
# AUTHENTICATION FUNCTIONS
# --------------------------
def login(email: str, password: str) -> bool:
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if response.user:
            st.session_state.user = response
            return True
        return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def sign_up(email: str, password: str, full_name: str) -> bool:
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

def check_auth() -> bool:
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

# --------------------------
# FILE VALIDATION FUNCTIONS
# --------------------------
def validate_twitter_account_overview(df: pd.DataFrame) -> bool:
    required_columns = {'Date', 'Impressions', 'Likes', 'Engagements'}
    return required_columns.issubset(df.columns)

def validate_twitter_post_data(df: pd.DataFrame) -> bool:
    required_columns = {'Date', 'Post text', 'Impressions', 'Likes'}
    return required_columns.issubset(df.columns)

def validate_instagram_posts(df: pd.DataFrame) -> bool:
    required_columns = {'Publish time', 'Post type', 'Reach', 'Likes'}
    return required_columns.issubset(df.columns)

def validate_instagram_stories(df: pd.DataFrame) -> bool:
    required_columns = {'Publish time', 'Reach', 'Likes', 'Replies'}
    return required_columns.issubset(df.columns)

def detect_file_type(df: pd.DataFrame, platform: str) -> Optional[str]:
    """Automatically detect the file type based on columns"""
    if platform == "twitter":
        if validate_twitter_account_overview(df):
            return "twitter_account_overview"
        elif validate_twitter_post_data(df):
            return "twitter_post_data"
    elif platform == "instagram":
        if validate_instagram_posts(df):
            return "instagram_posts"
        elif validate_instagram_stories(df):
            return "instagram_stories"
    return None

# --------------------------
# DATA PROCESSING FUNCTIONS
# --------------------------
def process_twitter_account_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process Twitter account overview data"""
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%a, %b %d, %Y')
        df = df.sort_values('Date')
        if 'New follows' in df.columns and 'Unfollows' in df.columns:
            df['followers'] = (df['New follows'] - df['Unfollows']).cumsum()
        return df
    except Exception as e:
        st.error(f"Error processing Twitter account data: {str(e)}")
        return df

def process_twitter_post_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process Twitter post data"""
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['tweet_length'] = df['Post text'].apply(lambda x: len(str(x)))
        
        def extract_mentions(text):
            return re.findall(r'@(\w+)', str(text))
        
        def extract_hashtags(text):
            return re.findall(r'#(\w+)', str(text))
        
        df['mentions'] = df['Post text'].apply(extract_mentions)
        df['hashtags'] = df['Post text'].apply(extract_hashtags)
        df['sentiment'] = df['Post text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        
        return df
    except Exception as e:
        st.error(f"Error processing Twitter post data: {str(e)}")
        return df

def process_instagram_post_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process Instagram post data"""
    try:
        df['Publish time'] = pd.to_datetime(df['Publish time'])
        df = df.sort_values('Publish time')
        df['Post Length'] = df['Description'].str.len().fillna(0)
        
        engagement_columns = ['Likes', 'Shares', 'Comments', 'Saves']
        available_engagement = [col for col in engagement_columns if col in df.columns]
        df['Engagement'] = df[available_engagement].sum(axis=1)
        
        if 'Reach' in df.columns:
            df['Engagement Rate'] = (df['Engagement'] / df['Reach']) * 100
        
        return df
    except Exception as e:
        st.error(f"Error processing Instagram post data: {str(e)}")
        return df

def process_instagram_story_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process Instagram story data"""
    try:
        df['Publish time'] = pd.to_datetime(df['Publish time'])
        df = df.sort_values('Publish time')
        
        engagement_columns = ['Likes', 'Replies', 'Sticker taps']
        available_engagement = [col for col in engagement_columns if col in df.columns]
        df['Engagement'] = df[available_engagement].sum(axis=1)
        
        if 'Reach' in df.columns:
            df['Engagement Rate'] = (df['Engagement'] / df['Reach']) * 100
        
        return df
    except Exception as e:
        st.error(f"Error processing Instagram story data: {str(e)}")
        return df

# --------------------------
# DATA MANAGEMENT FUNCTIONS
# --------------------------
def save_uploaded_file(user_id: str, account_id: int, data_type: str, file) -> Dict:
    try:
        # Read file content
        file_content = file.getvalue() if hasattr(file, 'getvalue') else file.read()
        file_ext = os.path.splitext(file.name)[1].lower()
        
        # Generate storage path with proper structure
        file_path = f"{user_id}/{account_id}/{uuid.uuid4()}{file_ext}"
        
        # Upload with explicit content type
        content_type = "text/csv" if file_ext == '.csv' else "application/vnd.ms-excel"
        res = supabase.storage.from_("analytics-uploads").upload(
            file_path,
            file_content,
            {"content-type": content_type}
        )
        
        # Verify upload
        try:
            # This is the critical fix - list files with the exact path prefix
            existing_files = supabase.storage.from_("analytics-uploads").list(f"{user_id}/{account_id}")
            if not any(f['name'] == file_path.split('/')[-1] for f in existing_files):
                return {"success": False, "message": "File verification failed after upload"}
        except Exception as verify_error:
            return {"success": False, "message": f"Verification error: {str(verify_error)}"}

        # Save metadata
        response = supabase.table("analytics_uploads").insert({
            "user_id": user_id,
            "account_id": account_id,
            "data_type": data_type,
            "file_name": file.name,
            "file_path": file_path
        }).execute()
        
        return {
            "success": True,
            "message": "File uploaded and verified successfully!",
            "file_path": file_path
        }
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}



def get_user_accounts(user_id: str) -> List[Dict]:
    try:
        return supabase.table("social_accounts").select("*").eq("user_id", user_id).execute().data
    except Exception as e:
        st.error(f"Error loading accounts: {str(e)}")
        return []

def get_account_uploads(user_id: str, account_id: int) -> List[Dict]:
    try:
        return supabase.table("analytics_uploads").select("*").eq("user_id", user_id).eq("account_id", account_id).execute().data
    except Exception as e:
        st.error(f"Error loading uploads: {str(e)}")
        return []

def get_upload_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        if not file_path:
            st.error("No file path provided")
            return None

        # Split path to match Supabase storage structure
        path_parts = file_path.split('/')
        folder_path = '/'.join(path_parts[:2])  # user_id/account_id
        file_name = path_parts[-1]  # actual filename
        
        # List files in the specific folder
        try:
            existing_files = supabase.storage.from_("analytics-uploads").list(folder_path)
            if not any(f['name'] == file_name for f in existing_files):
                st.error(f"File not found in storage: {file_name}")
                return None
        except Exception as list_error:
            st.error(f"Error listing files: {str(list_error)}")
            return None

        # Download the file
        try:
            res = supabase.storage.from_("analytics-uploads").download(file_path)
            if not res:
                st.error("Received empty response from storage")
                return None
                
            # Handle file type
            if file_path.endswith('.csv'):
                try:
                    # Try multiple encodings
                    try:
                        return pd.read_csv(StringIO(res.decode('utf-8')))
                    except UnicodeDecodeError:
                        return pd.read_csv(StringIO(res.decode('latin-1')))
                except Exception as csv_error:
                    st.error(f"CSV parsing error: {str(csv_error)}")
                    return None
            elif file_path.endswith('.xlsx'):
                try:
                    return pd.read_excel(BytesIO(res))
                except Exception as excel_error:
                    st.error(f"Excel parsing error: {str(excel_error)}")
                    return None
            else:
                st.error(f"Unsupported file type: {file_path}")
                return None
                
        except Exception as download_error:
            st.error(f"Download failed: {str(download_error)}")
            return None
            
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None
        
# --------------------------
# ANALYSIS FUNCTIONS
# --------------------------
def generate_summary(text: str, platform: str) -> Optional[str]:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"""You are a social media analytics expert. Provide a detailed, insightful summary (500-700 words) analyzing {platform} performance."""},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def twitter_account_analysis(df: pd.DataFrame):
    st.header("Twitter Account Overview")
    df = process_twitter_account_data(df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Days", len(df))
    if 'followers' in df.columns:
        col2.metric("Current Followers", f"{df['followers'].iloc[-1]:,}")
    if 'New follows' in df.columns and 'Unfollows' in df.columns:
        col3.metric("Avg. Daily Growth", f"{(df['New follows'] - df['Unfollows']).mean():.1f}")
    
    if 'followers' in df.columns and 'Date' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['followers'], name='Followers', line=dict(color='#1DA1F2')))
        fig.update_layout(title="Follower Growth Over Time", xaxis_title="Date", yaxis_title="Followers")
        st.plotly_chart(fig, use_container_width=True)
    
    fig = go.Figure()
    metrics = ['Impressions', 'Engagements', 'Likes']
    colors = ['#17BF63', '#E0245E', '#FFAD1F']
    
    for metric, color in zip(metrics, colors):
        if metric in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[metric], name=metric, line=dict(color=color)))
    
    if len(fig.data) > 0:
        fig.update_layout(title="Engagement Metrics Over Time")
        st.plotly_chart(fig, use_container_width=True)

def twitter_post_analysis(df: pd.DataFrame):
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
    if 'Impressions' in df.columns:
        st.subheader("Top Performing Posts")
        top_posts = df.sort_values('Impressions', ascending=False).head(5)
        for idx, row in top_posts.iterrows():
            with st.expander(f"{row['Date'].strftime('%Y-%m-%d')} | Impressions: {row['Impressions']:,} | Likes: {row['Likes']}"):
                st.write(f"**Post:** {row['Post text'][:200]}...")
                if 'Link' in df.columns:
                    st.markdown(f"[View Post]({row['Link']})")
    
    # Hashtag analysis
    if 'hashtags' in df.columns:
        st.subheader("Hashtag Performance")
        hashtags = pd.Series([tag for sublist in df['hashtags'] for tag in sublist]).value_counts()
        if not hashtags.empty:
            fig = px.bar(hashtags.head(10), title="Top 10 Hashtags")
            st.plotly_chart(fig, use_container_width=True)

def instagram_post_analysis(df: pd.DataFrame):
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
    if 'Reach' in df.columns:
        st.subheader("Top Performing Posts")
        top_posts = df.sort_values('Reach', ascending=False).head(5)
        for idx, row in top_posts.iterrows():
            with st.expander(f"{row['Publish time'].strftime('%Y-%m-%d')} | Reach: {row['Reach']:,} | Engagement: {row['Engagement']}"):
                if 'Description' in df.columns:
                    st.write(f"**Description:** {row['Description'][:200] if pd.notna(row['Description']) else 'No description'}")
                if 'Permalink' in df.columns:
                    st.markdown(f"[View Post]({row['Permalink']})")

def instagram_story_analysis(df: pd.DataFrame):
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
    if 'Duration (sec)' in df.columns and 'Engagement Rate' in df.columns:
        st.subheader("Story Duration Analysis")
        fig = px.scatter(df, x='Duration (sec)', y='Engagement Rate',
                        trendline="lowess", title='Duration vs Engagement')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performing stories
    if 'Reach' in df.columns:
        st.subheader("Top Performing Stories")
        top_stories = df.sort_values('Reach', ascending=False).head(5)
        for idx, row in top_stories.iterrows():
            with st.expander(f"{row['Publish time'].strftime('%Y-%m-%d')} | Reach: {row['Reach']:,} | Engagement: {row['Engagement']}"):
                if 'Description' in df.columns:
                    st.write(f"**Description:** {row['Description'][:200] if pd.notna(row['Description']) else 'No description'}")
                if 'Permalink' in df.columns:
                    st.markdown(f"[View Story]({row['Permalink']})")

## --------------------------
## COMPARISON FUNCTIONS
## --------------------------

def compare_twitter_accounts(account1_data: pd.DataFrame, account2_data: pd.DataFrame, 
                            account1_name: str, account2_name: str):
    st.header("Twitter Account Comparison")
    
    # Process data
    account1_data = process_twitter_account_data(account1_data)
    account2_data = process_twitter_account_data(account2_data)
    
    # Follower growth comparison
    if 'followers' in account1_data.columns and 'followers' in account2_data.columns:
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
    available_metrics = [m for m in metrics if m in account1_data.columns and m in account2_data.columns]
    
    if available_metrics:
        comparison_data = {
            'Metric': available_metrics,
            account1_name: [account1_data[m].mean() for m in available_metrics],
            account2_name: [account2_data[m].mean() for m in available_metrics]
        }
        
        fig = px.bar(pd.DataFrame(comparison_data), x='Metric', y=[account1_name, account2_name],
                    barmode='group', title='Average Engagement Comparison')
        st.plotly_chart(fig, use_container_width=True)

def compare_instagram_accounts(account1_data: pd.DataFrame, account2_data: pd.DataFrame,
                              account1_name: str, account2_name: str):
    st.header("Instagram Account Comparison")
    
    # Process data
    account1_data = process_instagram_post_data(account1_data)
    account2_data = process_instagram_post_data(account2_data)
    
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

# --------------------------
# MAIN APP LAYOUT
# --------------------------
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
            
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
            
            if uploaded_file:
                try:
                    # Preview file
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:  # Excel
                            df = pd.read_excel(uploaded_file)
                        st.subheader("File Preview")
                        st.dataframe(df.head(3))
                    except Exception as preview_error:
                        st.error(f"Preview error: {str(preview_error)}")
                        return
                        
                    # Detect file type
                    detected_type = detect_file_type(df, account["platform"])
                    if not detected_type:
                        st.error("Could not detect file type")
                        return
                        
                    st.success(f"Detected: {detected_type.replace('_', ' ').title()}")
                    
                    if st.button("Upload & Analyze"):
                        with st.spinner("Processing..."):
                            # Upload the file
                            result = save_uploaded_file(user_id, account["id"], detected_type, uploaded_file)
                            
                            if result["success"]:
                                st.success(result["message"])
                                
                                # Get the uploaded data using the exact path returned
                                df = get_upload_data(result["file_path"])
                                
                                if df is not None:
                                    st.write("Data loaded successfully. First row:", df.iloc[0])
                                    
                                    # Perform analysis
                                    if detected_type == "twitter_account_overview":
                                        twitter_account_analysis(df)
                                    elif detected_type == "twitter_post_data":
                                        twitter_post_analysis(df)
                                    elif detected_type == "instagram_posts":
                                        instagram_post_analysis(df)
                                    elif detected_type == "instagram_stories":
                                        instagram_story_analysis(df)
                            else:
                                st.error(result["message"])
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
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
                    if account1["platform"] == "twitter":
                        st.info("For Twitter comparison, please select account overview data")
                        upload1_options = [u for u in uploads1 if u["data_type"] == "twitter_account_overview"]
                        upload2_options = [u for u in uploads2 if u["data_type"] == "twitter_account_overview"]
                    else:
                        st.info("For Instagram comparison, please select post data")
                        upload1_options = [u for u in uploads1 if u["data_type"] == "instagram_posts"]
                        upload2_options = [u for u in uploads2 if u["data_type"] == "instagram_posts"]
                    
                    if upload1_options and upload2_options:
                        upload1 = st.selectbox("First Data Set", upload1_options, format_func=lambda x: x["file_name"], key="upload1")
                        upload2 = st.selectbox("Second Data Set", upload2_options, format_func=lambda x: x["file_name"], key="upload2")
                        
                        if st.button("Compare"):
                            df1 = get_upload_data(upload1["file_path"])
                            df2 = get_upload_data(upload2["file_path"])
                            
                            if df1 is not None and df2 is not None:
                                if account1["platform"] == "twitter":
                                    compare_twitter_accounts(df1, df2, account1["display_name"], account2["display_name"])
                                else:
                                    compare_instagram_accounts(df1, df2, account1["display_name"], account2["display_name"])
                    else:
                        st.warning("No compatible files found for comparison")
                        
def debug_storage():
    """Enhanced debug function"""
    try:
        st.subheader("Storage Debug Information")
        user_id = st.session_state.user.user.id
        
        # List all user files
        files = supabase.storage.from_("analytics-uploads").list(user_id)
        st.write("User files in storage:", files)
        
        # Show metadata from database
        uploads = supabase.table("analytics_uploads").select("*").eq("user_id", user_id).execute()
        st.write("Database records:", uploads.data)
        
        # Test download for each file
        for upload in uploads.data:
            try:
                st.write(f"\nTesting download for: {upload['file_path']}")
                res = supabase.storage.from_("analytics-uploads").download(upload['file_path'])
                st.write(f"Downloaded {len(res) if res else 0} bytes")
                if res:
                    try:
                        if upload['file_path'].endswith('.csv'):
                            df = pd.read_csv(StringIO(res.decode('utf-8')))
                        else:
                            df = pd.read_excel(BytesIO(res))
                        st.write("First row:", df.iloc[0])
                    except Exception as parse_error:
                        st.error(f"Parse error: {str(parse_error)}")
            except Exception as download_error:
                st.error(f"Download error: {str(download_error)}")
                
    except Exception as e:
        st.error(f"Debug error: {str(e)}")

# Add this to your sidebar for debugging:
with st.sidebar:
    if st.checkbox("Show debug info"):
        debug_storage()
# --------------------------
# APP ENTRY POINT
# --------------------------
def app():
    if not check_auth():
        show_auth()
    else:
        main_app()

if __name__ == "__main__":
    app()
