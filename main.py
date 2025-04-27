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

def instagram_stories_analysis(df):
    st.header("Instagram Stories Analysis")
    
    # Calculate metrics
    df['Engagement'] = df['Likes'].fillna(0) + df['Replies'].fillna(0) + df['Sticker taps'].fillna(0)
    if 'Reach' in df.columns:
        df['Engagement Rate'] = (df['Engagement'] / df['Reach']) * 100
    
    # Extract features
    df['Hour'] = pd.to_datetime(df['Publish time']).dt.hour
    df['Day of Week'] = pd.to_datetime(df['Publish time']).dt.day_name()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stories", len(df))
    col2.metric("Avg. Reach", f"{df['Reach'].mean():,.0f}")
    col3.metric("Avg. Engagement Rate", f"{df['Engagement Rate'].mean():.2f}%")
    col4.metric("Best Performing Story", f"{df['Reach'].max():,.0f} reach")
    
    # Visualizations
    fig = px.line(df, x='Publish time', y=['Reach', 'Engagement'], 
                 title='Performance Over Time')
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.scatter(df, x='Duration (sec)', y='Engagement Rate', 
                    trendline="lowess", title='Duration vs Engagement')
    st.plotly_chart(fig, use_container_width=True)

    # AI Summary
    if st.button("Generate AI Insights for Stories"):
        summary_text = f"""
        Instagram Stories Performance Summary:
        - Total Stories: {len(df)}
        - Average Reach: {df['Reach'].mean():,.0f}
        - Average Engagement Rate: {df['Engagement Rate'].mean():.2f}%
        - Best Performing Story Reach: {df['Reach'].max():,.0f}
        """
        summary = generate_summary(summary_text, "Instagram Stories")
        if summary:
            st.subheader("AI-Powered Insights")
            st.markdown(summary)

def instagram_posts_analysis(df):
    st.header("Instagram Posts Analysis")
    
    # Calculate metrics
    df['Engagement'] = df['Likes'].fillna(0) + df['Comments'].fillna(0) + df['Shares'].fillna(0) + df['Saves'].fillna(0)
    if 'Reach' in df.columns:
        df['Engagement Rate'] = (df['Engagement'] / df['Reach']) * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", len(df))
    col2.metric("Avg. Reach", f"{df['Reach'].mean():,.0f}")
    col3.metric("Avg. Engagement Rate", f"{df['Engagement Rate'].mean():.2f}%")
    col4.metric("Best Performing Post", f"{df['Reach'].max():,.0f} reach")
    
    # Visualizations
    if 'Post type' in df.columns:
        fig = px.bar(df.groupby('Post type').agg({
            'Reach': 'mean',
            'Engagement Rate': 'mean'
        }).reset_index(), x='Post type', y=['Reach', 'Engagement Rate'],
                    barmode='group', title='Performance by Post Type')
        st.plotly_chart(fig, use_container_width=True)

    # AI Summary
    if st.button("Generate AI Insights for Posts"):
        summary_text = f"""
        Instagram Posts Performance Summary:
        - Total Posts: {len(df)}
        - Average Reach: {df['Reach'].mean():,.0f}
        - Average Engagement Rate: {df['Engagement Rate'].mean():.2f}%
        - Best Performing Post Reach: {df['Reach'].max():,.0f}
        """
        if 'Post type' in df.columns:
            summary_text += f"\nPost Type Distribution:\n{df['Post type'].value_counts().to_string()}"
        
        summary = generate_summary(summary_text, "Instagram Posts")
        if summary:
            st.subheader("AI-Powered Insights")
            st.markdown(summary)

def twitter_analysis(df):
    st.header("Twitter Analytics")
    
    # Calculate metrics
    if 'Engagements' in df.columns and 'Impressions' in df.columns:
        df['engagement_rate'] = df['Engagements']/df['Impressions']
        mean_engagement_rate = df['engagement_rate'].mean()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tweets", len(df))
    if 'engagement_rate' in df.columns:
        col2.metric("Avg. Engagement Rate", f"{mean_engagement_rate:.2%}")
    if 'Likes' in df.columns:
        col3.metric("Avg. Likes", f"{df['Likes'].mean():,.0f}")
    
    # Visualizations
    if 'Date' in df.columns:
        fig = go.Figure()
        metrics = []
        colors = ['#FF006E', '#8338EC', '#3A86FF', '#FFBE0B']
        
        # Add available metrics to the plot
        for metric in ['Likes', 'Reposts', 'Replies', 'Bookmarks']:
            if metric in df.columns:
                metrics.append(metric)
        
        for metric, color in zip(metrics, colors[:len(metrics)]):
            fig.add_trace(go.Scatter(
                x=df['Date'], 
                y=df[metric],
                name=metric,
                line=dict(color=color)
            ))
        
        fig.update_layout(title="Engagement Metrics Over Time")
        st.plotly_chart(fig, use_container_width=True)

    # AI Summary
    if st.button("Generate AI Insights for Twitter"):
        summary_text = "Twitter Performance Summary:\n"
        summary_text += f"- Total Tweets: {len(df)}\n"
        if 'engagement_rate' in df.columns:
            summary_text += f"- Average Engagement Rate: {mean_engagement_rate:.2%}\n"
        if 'Likes' in df.columns:
            summary_text += f"- Average Likes: {df['Likes'].mean():,.0f}\n"
        
        summary = generate_summary(summary_text, "Twitter")
        if summary:
            st.subheader("AI-Powered Insights")
            st.markdown(summary)

## --------------------------
## COMPARISON FUNCTIONS
## --------------------------

def compare_twitter_accounts(df1, df2, account1_name, account2_name):
    st.header("Twitter Account Comparison")
    
    # Engagement comparison
    metrics = []
    for metric in ['Likes', 'Reposts', 'Replies', 'Bookmarks', 'Impressions']:
        if metric in df1.columns and metric in df2.columns:
            metrics.append(metric)
    
    if metrics:
        comparison = pd.DataFrame({
            'Metric': metrics,
            account1_name: [df1[m].mean() for m in metrics],
            account2_name: [df2[m].mean() for m in metrics]
        })
        
        fig = px.bar(comparison, x='Metric', y=[account1_name, account2_name],
                    barmode='group', title='Engagement Comparison')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No common metrics available for comparison")

def compare_instagram_accounts(df1, df2, account1_name, account2_name):
    st.header("Instagram Account Comparison")
    
    # Performance comparison
    metrics = []
    for metric in ['Reach', 'Engagement', 'Engagement Rate']:
        if metric in df1.columns and metric in df2.columns:
            metrics.append(metric)
    
    if metrics:
        comparison = pd.DataFrame({
            'Metric': metrics,
            account1_name: [df1[m].mean() for m in metrics],
            account2_name: [df2[m].mean() for m in metrics]
        })
        
        fig = px.bar(comparison, x='Metric', y=[account1_name, account2_name],
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
            data_type = st.selectbox("Data Type", 
                                   ["twitter_tweets", "instagram_posts", "instagram_stories"])
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
            
            if uploaded_file and st.button("Upload & Analyze"):
                with st.spinner("Processing..."):
                    if save_uploaded_file(user_id, account["id"], data_type, uploaded_file):
                        st.success("File uploaded successfully!")
                        df = get_upload_data(f"{user_id}/{account['id']}/{uploaded_file.name}")
                        
                        if df is not None:
                            st.subheader("Analysis Results")
                            if data_type == "twitter_tweets":
                                twitter_analysis(df)
                            elif data_type == "instagram_posts":
                                instagram_posts_analysis(df)
                            elif data_type == "instagram_stories":
                                instagram_stories_analysis(df)
    
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
                    upload1 = st.selectbox("First Data Set", uploads1, 
                                         format_func=lambda x: x["file_name"], key="upload1")
                    upload2 = st.selectbox("Second Data Set", uploads2, 
                                         format_func=lambda x: x["file_name"], key="upload2")
                    
                    if st.button("Compare"):
                        df1 = get_upload_data(upload1["file_path"])
                        df2 = get_upload_data(upload2["file_path"])
                        
                        if df1 is not None and df2 is not None:
                            if account1["platform"] == "twitter":
                                compare_twitter_accounts(df1, df2, account1["display_name"], account2["display_name"])
                            else:
                                compare_instagram_accounts(df1, df2, account1["display_name"], account2["display_name"])

## --------------------------
## APP ENTRY POINT
## --------------------------

def app():
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
