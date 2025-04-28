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
from datetime import timedelta

# Constants
SESSION_TIMEOUT_MINUTES = 30  # Session will expire after 30 minutes of inactivity

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
# SECURITY FUNCTIONS
# --------------------------
def generate_csrf_token():
    return str(uuid.uuid4())

def validate_csrf_token(token: str) -> bool:
    if "csrf_token" not in st.session_state:
        return False
    return st.session_state.csrf_token == token

def check_session_timeout():
    """Check if session has timed out due to inactivity"""
    if "last_activity" not in st.session_state:
        return True
    
    elapsed = datetime.now() - st.session_state.last_activity
    return elapsed > timedelta(minutes=SESSION_TIMEOUT_MINUTES)

def update_last_activity():
    """Update the last activity timestamp"""
    st.session_state.last_activity = datetime.now()

def get_client_ip():
    """Get client IP address for logging"""
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers:
            return headers.get("X-Forwarded-For", "unknown")
    except:
        pass
    return "unknown"

# --------------------------
# AUTHENTICATION FUNCTIONS
# --------------------------
def login(email: str, password: str) -> bool:
    try:
        # Rate limiting check
        if "login_attempts" not in st.session_state:
            st.session_state.login_attempts = 0
            st.session_state.last_attempt = datetime.now()
        
        if st.session_state.login_attempts >= 5:
            elapsed = datetime.now() - st.session_state.last_attempt
            if elapsed < timedelta(minutes=5):
                st.error("Too many login attempts. Please try again later.")
                return False
        
        st.session_state.login_attempts += 1
        st.session_state.last_attempt = datetime.now()
        
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            # Successful login - reset attempts
            st.session_state.login_attempts = 0
            st.session_state.user = response
            update_last_activity()
            
            # Log the login
            supabase.table('audit_logs').insert({
                "user_id": response.user.id,
                "action": "login",
                "ip_address": get_client_ip(),
                "timestamp": datetime.now().isoformat()
            }).execute()
            
            return True
        return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def sign_up(email: str, password: str, full_name: str) -> bool:
    try:
        # Input validation
        if len(password) < 8:
            st.error("Password must be at least 8 characters")
            return False
            
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            # Create user profile
            supabase.table("profiles").insert({
                "id": response.user.id,
                "email": email,
                "full_name": full_name,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            # Log the signup
            supabase.table('audit_logs').insert({
                "user_id": response.user.id,
                "action": "signup",
                "ip_address": get_client_ip(),
                "timestamp": datetime.now().isoformat()
            }).execute()
            
            return True
        return False
    except Exception as e:
        st.error(f"Sign up error: {str(e)}")
        return False

def logout():
    try:
        if "user" in st.session_state:
            # Log the logout
            supabase.table('audit_logs').insert({
                "user_id": st.session_state.user.user.id,
                "action": "logout",
                "ip_address": get_client_ip(),
                "timestamp": datetime.now().isoformat()
            }).execute()
            
            supabase.auth.sign_out()
        st.session_state.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Logout error: {str(e)}")
        st.session_state.clear()
        st.rerun()

def check_auth() -> bool:
    # Check for session timeout
    if "user" in st.session_state and check_session_timeout():
        st.warning("Your session has timed out due to inactivity. Please log in again.")
        logout()
        return False
    
    # First check if we have a user in session state
    if "user" in st.session_state:
        try:
            # Verify the session is still valid with Supabase
            current_user = supabase.auth.get_user()
            if current_user and current_user.user.id == st.session_state.user.user.id:
                update_last_activity()
                return True
            # If IDs don't match, clear session
            st.session_state.clear()
            return False
        except Exception as e:
            st.session_state.clear()
            return False
    
    # If no user in session state, try to get from Supabase
    try:
        current_user = supabase.auth.get_user()
        if current_user:
            st.session_state.user = current_user
            update_last_activity()
            return True
    except:
        pass
    
    return False

def show_auth():
    # Initialize CSRF token if not exists
    if "csrf_token" not in st.session_state:
        st.session_state.csrf_token = generate_csrf_token()
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            csrf_token = st.session_state.csrf_token
            
            if st.form_submit_button("Login"):
                if login(email, password):
                    st.rerun()
    
    with tab2:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            full_name = st.text_input("Full Name", key="signup_name")
            csrf_token = st.session_state.csrf_token
            
            if st.form_submit_button("Create Account"):
                if sign_up(email, password, full_name):
                    st.success("Account created! Please login.")
                    st.rerun()

# --------------------------
# FILE VALIDATION FUNCTIONS
# (No changes needed here)
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
# (No changes needed here)
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
        # Verify ownership before upload
        account = supabase.table("social_accounts").select("*").eq("id", account_id).eq("user_id", user_id).execute()
        if not account.data:
            return {"success": False, "message": "Account not found or access denied"}
        
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
            "file_path": file_path,
            "uploaded_at": datetime.now().isoformat()
        }).execute()
        
        # Log the upload
        supabase.table('audit_logs').insert({
            "user_id": user_id,
            "action": "file_upload",
            "details": f"Uploaded {file.name} for account {account_id}",
            "ip_address": get_client_ip(),
            "timestamp": datetime.now().isoformat()
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

        # Verify the file belongs to the current user
        if "user" not in st.session_state or not file_path.startswith(st.session_state.user.user.id):
            st.error("Access denied")
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
    
    # Verify authentication and session
    if not check_auth():
        show_auth()
        return
    
    user = st.session_state.user
    user_id = user.user.id
    
    with st.sidebar:
        st.header(f"Welcome, {user.user.email.split('@')[0]}")
        
        # Session info
        time_left = SESSION_TIMEOUT_MINUTES - (datetime.now() - st.session_state.last_activity).seconds // 60
        st.caption(f"Session expires in {time_left} minutes")
        
        if st.button("Logout"):
            logout()
        
        with st.expander("Add Social Account"):
            with st.form("new_account_form"):
                platform = st.selectbox("Platform", ["twitter", "instagram"])
                username = st.text_input("Username")
                display_name = st.text_input("Display Name")
                csrf_token = st.session_state.csrf_token
                
                if st.form_submit_button("Save Account"):
                    supabase.table("social_accounts").insert({
                        "user_id": user_id,
                        "platform": platform,
                        "username": username,
                        "display_name": display_name,
                        "created_at": datetime.now().isoformat()
                    }).execute()
                    
                    # Log the account creation
                    supabase.table('audit_logs').insert({
                        "user_id": user_id,
                        "action": "account_created",
                        "details": f"Added {platform} account {username}",
                        "ip_address": get_client_ip(),
                        "timestamp": datetime.now().isoformat()
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

# --------------------------
# APP ENTRY POINT
# --------------------------
def app():
    # Initialize session state if needed
    if "csrf_token" not in st.session_state:
        st.session_state.csrf_token = generate_csrf_token()
    
    st.set_page_config(
        page_title="Social Media Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    if not check_auth():
        show_auth()
    else:
        main_app()

if __name__ == "__main__":
    app()
    
