"""
YouTube Analytics Dashboard
A comprehensive dashboard for analyzing YouTube channel data with login system,
channel insights, comparison, and predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import hashlib
import json
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #ff0000;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        color: white !important;
    }
    div[data-testid="metric-container"] {
        background-color: #ff0000;
        border: 1px solid #cc0000;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] > div {
        color: white !important;
    }
    div[data-testid="metric-container"] label {
        color: white !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: white !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        color: #ffcccc !important;
    }
    .login-title {
        color: white !important;
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'page' not in st.session_state:
    st.session_state.page = "Channel Insights"

# Authentication functions
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# User database functions
def load_users():
    users_file = Path('users.json')
    if users_file.exists():
        with open(users_file, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

# Loading data and models
@st.cache_data
def load_data():
    """Load preprocessed data from pickle files"""
    data_dir = Path("Data")
    channels_df = pd.read_pickle(data_dir / "channels_df.pkl")
    videos_df = pd.read_pickle(data_dir / "videos_df.pkl")
    return channels_df, videos_df

@st.cache_resource
def load_models():
    """Load trained models"""
    models_dir = Path("models")
    models = {
        'rf_video': joblib.load(models_dir / "rf_video.joblib"),
        'rf_channel': joblib.load(models_dir / "rf_channel.joblib"),
        'scaler': joblib.load(models_dir / "scaler.joblib"),
        'labelenc': joblib.load(models_dir / "labelenc.joblib"),
        'labelenc_channel': joblib.load(models_dir / "labelenc_channel.joblib")
    }
    return models

# Login page
def login_page():

    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/YouTube_Logo_2017.svg/2560px-YouTube_Logo_2017.svg.png' width='300'>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='login-title'>
            <h1 style='color: white;'>📺 YouTube Analytics Dashboard</h1>
            <p style='font-size: 1rem; color: white;'>Analyze YouTube Channels With Advanced Insights and Predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("📧 Email", placeholder="Enter your email")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
                login_button = st.form_submit_button("Login", use_container_width=True, type="primary")
                
                if login_button:
                    users = load_users()
                    if email in users and check_hashes(password, users[email]):
                        st.session_state.logged_in = True
                        st.session_state.username = email
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password")
        
        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input("📧 Email", placeholder="Enter your email", key="signup_email")
                new_password = st.text_input("🔒 Password", type="password", placeholder="Create a password", key="signup_password")
                confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
                signup_button = st.form_submit_button("Sign Up", use_container_width=True, type="primary")
                
                if signup_button:
                    users = load_users()
                    if new_email and new_password:
                        if new_email in users:
                            st.error("❌ Email already exists!")
                        elif new_password != confirm_password:
                            st.error("❌ Passwords don't match!")
                        elif len(new_password) < 6:
                            st.error("❌ Password must be at least 6 characters!")
                        else:
                            users[new_email] = make_hashes(new_password)
                            save_users(users)
                            st.success("✅ Account created successfully! Please login.")
                    else:
                        st.error("❌ Please fill all fields")
    
    # Add designer credits
    st.markdown("""
        <div style='text-align: center; padding: 2rem; color: white;'>
            <p>Designed by: <strong>Anuj & Garema</strong></p>
        </div>
    """, unsafe_allow_html=True)

# Channel Insights Page
def channel_insights_page(channels_df, videos_df):
    st.title("🔍 Channel Insights")
    st.markdown("---")
    
    # Search functionality
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Search for a YouTube Channel", placeholder="Enter Channel Name...")
    
    if search_query:
        # Add progress bar with gif
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            st.markdown("""
                <div style='text-align: center;'>
                    <img src='https://i.pinimg.com/originals/98/69/9f/98699f9fdb9518e4c2f13f1b00488e01.gif' 
                         style='width: 100px; height: 100px; border-radius: 50%;'>
                    <p>Searching channels...</p>
                </div>
            """, unsafe_allow_html=True)
            
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)
        
        progress_placeholder.empty()
        
        # Filter channels
        filtered_channels = channels_df[
            channels_df['title'].str.contains(search_query, case=False, na=False)
        ]
        
        if not filtered_channels.empty:
            # Channel selection
            selected_channel_title = st.selectbox(
                "Select a channel",
                filtered_channels['title'].unique(),
                format_func=lambda x: f"{x} ({filtered_channels[filtered_channels['title']==x]['subscriberCount'].iloc[0]:,} subscribers)"
            )
            
            # channel data
            channel_data = filtered_channels[filtered_channels['title'] == selected_channel_title].iloc[0]
            channel_videos = videos_df[videos_df['channelId'] == channel_data['channelId']]
            
            # Channel header
            st.markdown(f"## {channel_data['title']}")
            st.markdown(f"**Category:** {channel_data['categoryTitle']}")
            
            # Key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Subscribers", 
                    f"{channel_data['subscriberCount']:,}",
                    help="Total number of subscribers"
                )
            
            with col2:
                st.metric(
                    "Total Views", 
                    f"{channel_data['viewCount']:,}",
                    help="Total views across all videos"
                )
            
            with col3:
                st.metric(
                    "Videos", 
                    f"{channel_data['videoCount']:,}",
                    help="Total number of videos published"
                )
            
            with col4:
                if len(channel_videos) > 0:
                    avg_views = channel_videos['viewCount'].mean()
                    st.metric(
                        "Avg Views/Video", 
                        f"{avg_views:,.0f}",
                        help="Average views per video"
                    )
                else:
                    st.metric("Avg Views/Video", "N/A")
            
            with col5:
                engagement_rate = (channel_videos['commentCount'].sum() / 
                                 channel_videos['viewCount'].sum() * 100) if len(channel_videos) > 0 else 0
                st.metric(
                    "Engagement Rate", 
                    f"{engagement_rate:.2f}%",
                    help="Comments to views ratio"
                )
            
            # Visualizations
            if len(channel_videos) > 0:
                st.markdown("---")
                
                # Row 1: Top videos and engagement
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top 10 videos by views
                    top_videos = channel_videos.nlargest(10, 'viewCount')
                    fig_top_videos = px.bar(
                        top_videos,
                        x='viewCount',
                        y='title',
                        orientation='h',
                        title="🏆 Top 10 Videos by Views",
                        color='viewCount',
                        color_continuous_scale='Reds',
                        labels={'viewCount': 'Views', 'title': 'Video Title'}
                    )
                    fig_top_videos.update_layout(
                        height=400,
                        showlegend=False,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig_top_videos, use_container_width=True)
                
                with col2:
                    # pie chart
                    engagement_data = pd.DataFrame({
                        'Metric': ['Views', 'Comments'],
                        'Count': [
                            channel_videos['viewCount'].sum(),
                            channel_videos['commentCount'].sum() * 100  
                        ]
                    })
                    fig_engagement = go.Figure()
                    fig_engagement.add_trace(go.Pie(
                        labels=engagement_data['Metric'],
                        values=engagement_data['Count'],
                        pull=[0, 0.1],
                        marker=dict(colors=['#ff6666', '#ff0000'])
                    ))
                    fig_engagement.update_layout(
                        title="📊 Engagement Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_engagement, use_container_width=True)
                
                # Row 2: Timeline analysis
                st.markdown("---")
                
                # Video publishing timeline
                channel_videos['publishedAt'] = pd.to_datetime(channel_videos['publishedAt'])
                timeline_data = channel_videos.groupby(
                    channel_videos['publishedAt'].dt.to_period('M').astype(str)
                ).agg({
                    'viewCount': 'sum',
                    'videoId': 'count'
                }).reset_index()
                timeline_data.columns = ['Month', 'Total Views', 'Videos Published']
                
                # Create subplot with secondary y-axis
                fig_timeline = make_subplots(
                    specs=[[{"secondary_y": True}]],
                    subplot_titles=["📅 Channel Activity Timeline"]
                )
                
                # Add traces with red colors
                fig_timeline.add_trace(
                    go.Bar(
                        x=timeline_data['Month'],
                        y=timeline_data['Videos Published'],
                        name='Videos Published',
                        marker_color='#ff6666',
                        yaxis='y2'
                    ),
                    secondary_y=True
                )
                
                fig_timeline.add_trace(
                    go.Scatter(
                        x=timeline_data['Month'],
                        y=timeline_data['Total Views'],
                        name='Total Views',
                        line=dict(color='#ff0000', width=3),
                        mode='lines+markers'
                    ),
                    secondary_y=False
                )
                
                fig_timeline.update_xaxes(title_text="Month")
                fig_timeline.update_yaxes(title_text="Total Views", secondary_y=False)
                fig_timeline.update_yaxes(title_text="Videos Published", secondary_y=True)
                fig_timeline.update_layout(height=400, hovermode='x unified')
                
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Row 3: Video performance distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    # Views distribution - Red bars
                    fig_dist = px.histogram(
                        channel_videos,
                        x='viewCount',
                        nbins=20,
                        title="📈 Views Distribution",
                        labels={'viewCount': 'Views', 'count': 'Number of Videos'},
                        color_discrete_sequence=['#ff0000']
                    )
                    fig_dist.update_layout(height=350)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Recent performance
                    recent_videos = channel_videos.nlargest(5, 'publishedAt')
                    fig_recent = px.bar(
                        recent_videos,
                        x='title',
                        y='viewCount',
                        title="🆕 Recent Videos Performance",
                        color='commentCount',
                        color_continuous_scale='Reds',
                        labels={'viewCount': 'Views', 'commentCount': 'Comments'}
                    )
                    fig_recent.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig_recent, use_container_width=True)
                
                # Channel statistics summary
                st.markdown("---")
                st.subheader("📊 Channel Statistics Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"""
                    **Publishing Frequency**  
                    • Videos per month: {len(channel_videos) / max(1, len(timeline_data)):.1f}  
                    • Channel age: {channel_data.get('channel_age_days', 'N/A')} days  
                    • First video: {channel_videos['publishedAt'].min().strftime('%Y-%m-%d') if len(channel_videos) > 0 else 'N/A'}
                    """)
                
                with col2:
                    st.info(f"""
                    **Performance Metrics**  
                    • Best performing video: {channel_videos['viewCount'].max():,} views  
                    • Worst performing video: {channel_videos['viewCount'].min():,} views  
                    • Median views: {channel_videos['viewCount'].median():,.0f}
                    """)
                
                with col3:
                    st.info(f"""
                    **Engagement Metrics**  
                    • Total comments: {channel_videos['commentCount'].sum():,}  
                    • Avg comments/video: {channel_videos['commentCount'].mean():.0f}  
                    • Most commented: {channel_videos['commentCount'].max():,} comments
                    """)
            
            else:
                st.warning("No video data available for this channel.")
        
        else:
            st.warning("No channels found matching your search. Try a different search term.")

# Channel Comparison Page
def channel_comparison_page(channels_df, videos_df):
    st.title("⚖️ Channel Comparison")
    st.markdown("Compare Two YouTube Channels Side by Side")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    channel1_data = None
    channel2_data = None
    
    with col1:
        st.subheader("Channel 1")
        channel1_search = st.text_input("Search First Channel Name", key="ch1_search", placeholder="Enter Channel Name...")
        
        if channel1_search:
            filtered1 = channels_df[
                channels_df['title'].str.contains(channel1_search, case=False, na=False)
            ]
            if not filtered1.empty:
                channel1_title = st.selectbox(
                    "Select Channel 1",
                    filtered1['title'].unique(),
                    key="ch1_select",
                    format_func=lambda x: f"{x} ({filtered1[filtered1['title']==x]['subscriberCount'].iloc[0]:,} subs)"
                )
                channel1_data = filtered1[filtered1['title'] == channel1_title].iloc[0]
            else:
                st.warning("Write Correct Channel Name")
    
    with col2:
        st.subheader("Channel 2")
        channel2_search = st.text_input("Search Second Channel Name  ", key="ch2_search", placeholder="Enter Channel Name...")
        
        if channel2_search:
            filtered2 = channels_df[
                channels_df['title'].str.contains(channel2_search, case=False, na=False)
            ]
            if not filtered2.empty:
                channel2_title = st.selectbox(
                    "Select Channel 2",
                    filtered2['title'].unique(),
                    key="ch2_select",
                    format_func=lambda x: f"{x} ({filtered2[filtered2['title']==x]['subscriberCount'].iloc[0]:,} subs)"
                )
                channel2_data = filtered2[filtered2['title'] == channel2_title].iloc[0]
            else:
                st.warning("Write Correct Channel Name")
    
    if channel1_data is not None and channel2_data is not None:
        st.markdown("---")
        
        # Get videos for both channels
        ch1_videos = videos_df[videos_df['channelId'] == channel1_data['channelId']]
        ch2_videos = videos_df[videos_df['channelId'] == channel2_data['channelId']]
        
        # Comparison header
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"### {channel1_data['title']}")
        with col2:
            st.markdown("### VS")
        with col3:
            st.markdown(f"### {channel2_data['title']}")
        
        st.markdown("---")
        
        # Metrics comparison - separate views with pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for views comparison 
            views_data = pd.DataFrame({
                'Channel': [channel1_data['title'], channel2_data['title']],
                'Views': [channel1_data['viewCount'], channel2_data['viewCount']]
            })
            fig_views_pie = go.Figure()
            fig_views_pie.add_trace(go.Pie(
                labels=views_data['Channel'],
                values=views_data['Views'],
                pull=[0.1, 0],  
                marker=dict(colors=['#ff0000', '#ff6666'])
            ))
            fig_views_pie.update_layout(
                title='📺 Total Views Comparison',
                height=400
            )
            st.plotly_chart(fig_views_pie, use_container_width=True)
        
        with col2:
            # Other metrics comparison
            metrics = ['Subscribers', 'Videos', 'Avg Views/Video', 'Channel Age (days)']
            ch1_values = [
                channel1_data['subscriberCount'],
                channel1_data['videoCount'],
                channel1_data['viewCount'] / max(channel1_data['videoCount'], 1),
                channel1_data.get('channel_age_days', 0)
            ]
            ch2_values = [
                channel2_data['subscriberCount'],
                channel2_data['videoCount'],
                channel2_data['viewCount'] / max(channel2_data['videoCount'], 1),
                channel2_data.get('channel_age_days', 0)
            ]
            
            comparison_df = pd.DataFrame({
                'Metric': metrics * 2,
                'Value': ch1_values + ch2_values,
                'Channel': [channel1_data['title']] * 4 + [channel2_data['title']] * 4
            })
            
            fig_comparison = px.bar(
                comparison_df,
                x='Metric',
                y='Value',
                color='Channel',
                barmode='group',
                title='📊 Channel Metrics Comparison',
                color_discrete_map={
                    channel1_data['title']: '#ff0000',
                    channel2_data['title']: '#ff6666'
                }
            )
            fig_comparison.update_layout(height=400)
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Performance indicators 
        col1, col2, col3 = st.columns(3)
        
        with col1:
            subs_diff = channel1_data['subscriberCount'] - channel2_data['subscriberCount']
            winner = channel1_data['title'] if subs_diff > 0 else channel2_data['title']
            st.metric(
                "Subscriber Leader",
                winner,
                f"{abs(subs_diff):,} more subscribers",
                delta_color="off"
            )
        
        with col2:
            views_diff = channel1_data['viewCount'] - channel2_data['viewCount']
            winner = channel1_data['title'] if views_diff > 0 else channel2_data['title']
            st.metric(
                "Views Leader",
                winner,
                f"{abs(views_diff):,} more views",
                delta_color="off"
            )
        
        with col3:
            eng1 = ch1_videos['commentCount'].sum() / max(ch1_videos['viewCount'].sum(), 1) * 100 if len(ch1_videos) > 0 else 0
            eng2 = ch2_videos['commentCount'].sum() / max(ch2_videos['viewCount'].sum(), 1) * 100 if len(ch2_videos) > 0 else 0
            eng_diff = eng1 - eng2
            winner = channel1_data['title'] if eng_diff > 0 else channel2_data['title']
            st.metric(
                "Engagement Leader",
                winner,
                f"{abs(eng_diff):.2f}% higher",
                delta_color="off"
            )
        
        # Detailed comparison
        st.markdown("---")
        st.subheader("📈 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Growth Metrics", "Content Analysis", "Category Comparison"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{channel1_data['title']}**")
                daily_growth1 = channel1_data['subscriberCount'] / max(channel1_data.get('channel_age_days', 1), 1)
                video_rate1 = channel1_data['videoCount'] / max(channel1_data.get('channel_age_days', 1), 1) * 30
                
                st.info(f"""
                • Daily subscriber growth: {daily_growth1:.0f} subs/day  
                • Monthly video upload rate: {video_rate1:.1f} videos/month  
                • Views per subscriber: {channel1_data['viewCount'] / max(channel1_data['subscriberCount'], 1):.1f}
                """)
            
            with col2:
                st.markdown(f"**{channel2_data['title']}**")
                daily_growth2 = channel2_data['subscriberCount'] / max(channel2_data.get('channel_age_days', 1), 1)
                video_rate2 = channel2_data['videoCount'] / max(channel2_data.get('channel_age_days', 1), 1) * 30
                
                st.info(f"""
                • Daily subscriber growth: {daily_growth2:.0f} subs/day  
                • Monthly video upload rate: {video_rate2:.1f} videos/month  
                • Views per subscriber: {channel2_data['viewCount'] / max(channel2_data['subscriberCount'], 1):.1f}
                """)
        
        with tab2:
            if len(ch1_videos) > 0 and len(ch2_videos) > 0:
                content_comparison = pd.DataFrame({
                    'Channel': [channel1_data['title'], channel2_data['title']],
                    'Avg Views per Video': [ch1_videos['viewCount'].mean(), ch2_videos['viewCount'].mean()],
                    'Median Views': [ch1_videos['viewCount'].median(), ch2_videos['viewCount'].median()],
                    'Max Views': [ch1_videos['viewCount'].max(), ch2_videos['viewCount'].max()],
                    'Total Comments': [ch1_videos['commentCount'].sum(), ch2_videos['commentCount'].sum()]
                })
                
                fig_content = px.bar(
                    content_comparison.melt(id_vars=['Channel'], var_name='Metric', value_name='Value'),
                    x='Metric',
                    y='Value',
                    color='Channel',
                    barmode='group',
                    title='Content Performance Metrics',
                    color_discrete_map={
                        channel1_data['title']: '#ff0000',
                        channel2_data['title']: '#ff6666'
                    }
                )
                st.plotly_chart(fig_content, use_container_width=True)
            else:
                st.warning("Not enough video data for content analysis")
        
        with tab3:
            st.info(f"""
            **{channel1_data['title']}**: {channel1_data['categoryTitle']}  
            **{channel2_data['title']}**: {channel2_data['categoryTitle']}
            """)
            
            if channel1_data['categoryTitle'] == channel2_data['categoryTitle']:
                st.success(f"Both channels are in the same category: {channel1_data['categoryTitle']}")
            else:
                st.warning("Channels are in different categories, which might affect direct comparisons")

# Predictions Page
def predictions_page(channels_df, videos_df, models):
    st.title("🔮 Channel Growth Predictions")
    st.markdown("Predict future channel growth and video performance using machine learning")
    st.markdown("---")
    
    # Channel selection
    search_query = st.text_input(
        "🔎 Search for a YouTube Channel to Predict Growth",
        placeholder="Enter channel name..."
    )
    
    if search_query:
        filtered_channels = channels_df[
            channels_df['title'].str.contains(search_query, case=False, na=False)
        ]
        
        if not filtered_channels.empty:
            selected_channel_title = st.selectbox(
                "Select a channel",
                filtered_channels['title'].unique(),
                format_func=lambda x: f"{x} ({filtered_channels[filtered_channels['title']==x]['subscriberCount'].iloc[0]:,} subscribers)"
            )
            
            channel_data = filtered_channels[filtered_channels['title'] == selected_channel_title].iloc[0]
            channel_videos = videos_df[videos_df['channelId'] == channel_data['channelId']]
            
            # Current statistics
            st.markdown("### 📊 Current Channel Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Subscribers", f"{channel_data['subscriberCount']:,}")
            with col2:
                st.metric("Total Views", f"{channel_data['viewCount']:,}")
            with col3:
                st.metric("Videos", f"{channel_data['videoCount']:,}")
            with col4:
                st.metric("Avg Views/Video", f"{channel_data.get('avg_video_views', 0):,.0f}")
            
            st.markdown("---")
            
            # Prediction settings
            st.markdown("### ⚙️ Prediction Settings")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                prediction_months = st.slider("Months to predict", 1, 24, 12)
            with col2:
                growth_scenario = st.selectbox(
                    "Growth Scenario",
                    ["Conservative", "Moderate", "Optimistic"],
                    index=1
                )
            with col3:
                video_frequency = st.number_input(
                    "Expected Videos Per Month",
                    min_value=1,
                    max_value=100,
                    value=int(channel_data['videoCount'] / max(channel_data.get('channel_age_days', 30) / 30, 1))
                )
            
            growth_factors = {
                "Conservative": {"sub_growth": 0.02, "view_growth": 0.03},
                "Moderate": {"sub_growth": 0.05, "view_growth": 0.07},
                "Optimistic": {"sub_growth": 0.10, "view_growth": 0.15}
            }
            
            selected_factors = growth_factors[growth_scenario]
            
            # Generate predictions
            st.markdown("---")
            st.markdown("### 📈 Growth Predictions")
            
            months = []
            predicted_subs = []
            predicted_views = []
            predicted_total_videos = []
            
            current_subs = channel_data['subscriberCount']
            current_views = channel_data['viewCount']
            current_videos = channel_data['videoCount']
            
            for month in range(1, prediction_months + 1):
                month_subs = current_subs * ((1 + selected_factors['sub_growth']) ** month)
                month_views = current_views * ((1 + selected_factors['view_growth']) ** month)
                month_videos = current_videos + (video_frequency * month)
                
                months.append(f"Month {month}")
                predicted_subs.append(month_subs)
                predicted_views.append(month_views)
                predicted_total_videos.append(month_videos)
            
            # Create prediction visualizations
            tab1, tab2, tab3 = st.tabs(["Subscriber Growth", "View Growth", "Combined Analysis"])
            
            with tab1:
                fig_subs = go.Figure()
                
                historical_months = [-6, -5, -4, -3, -2, -1, 0]
                historical_subs = [current_subs * (0.85 ** abs(m)) for m in historical_months[:-1]] + [current_subs]
                
                fig_subs.add_trace(go.Scatter(
                    x=historical_months,
                    y=historical_subs,
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#ff0000', width=3),
                    marker=dict(size=8)
                ))
                
                fig_subs.add_trace(go.Scatter(
                    x=list(range(len(months))),
                    y=predicted_subs,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#ff6666', width=3, dash='dash'),
                    marker=dict(size=8)
                ))
                
                fig_subs.update_layout(
                    title='Subscriber Growth Prediction',
                    xaxis_title='Months from Now',
                    yaxis_title='Subscribers',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_subs, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    total_growth = predicted_subs[-1] - current_subs
                    growth_percent = (predicted_subs[-1] / current_subs - 1) * 100
                    st.success(f"""
                    **Subscriber Growth Summary**  
                    • Starting: {current_subs:,}  
                    • Predicted: {predicted_subs[-1]:,.0f}  
                    • Growth: +{total_growth:,.0f} ({growth_percent:.1f}%)
                    """)
                
                with col2:
                    monthly_avg = total_growth / prediction_months
                    st.info(f"""
                    **Monthly Averages**  
                    • Avg monthly growth: {monthly_avg:,.0f} subs  
                    • Growth rate: {selected_factors['sub_growth']*100:.1f}% per month  
                    • Time to double: {np.log(2) / np.log(1 + selected_factors['sub_growth']):.1f} months
                    """)
            
            with tab2:
                fig_views = go.Figure()
                
                fig_views.add_trace(go.Scatter(
                    x=months,
                    y=predicted_views,
                    mode='lines+markers',
                    name='Total Views',
                    line=dict(color='#ff0000', width=3),
                    marker=dict(size=8)
                ))
                
                avg_views_per_video = [views / videos for views, videos in zip(predicted_views, predicted_total_videos)]
                
                fig_views.add_trace(go.Scatter(
                    x=months,
                    y=avg_views_per_video,
                    mode='lines+markers',
                    name='Avg Views per Video',
                    line=dict(color='#ff6666', width=3),
                    marker=dict(size=8),
                    yaxis='y2'
                ))
                
                fig_views.update_layout(
                    title='View Growth Prediction',
                    xaxis_title='Time Period',
                    yaxis_title='Total Views',
                    yaxis2=dict(
                        title='Avg Views per Video',
                        overlaying='y',
                        side='right'
                    ),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_views, use_container_width=True)
                
                st.success(f"""
                **View Growth Summary**  
                • Current total views: {current_views:,}  
                • Predicted total views: {predicted_views[-1]:,.0f}  
                • Growth: +{predicted_views[-1] - current_views:,.0f} ({(predicted_views[-1] / current_views - 1) * 100:.1f}%)  
                • Predicted avg views/video: {avg_views_per_video[-1]:,.0f}
                """)
            
            with tab3:
                st.markdown("#### 🎯 Key Performance Indicators Projection")
                
                kpi_data = pd.DataFrame({
                    'Metric': ['Subscribers', 'Total Views', 'Videos', 'Views/Sub', 'Views/Video'],
                    'Current': [
                        current_subs,
                        current_views,
                        current_videos,
                        current_views / max(current_subs, 1),
                        current_views / max(current_videos, 1)
                    ],
                    f'After {prediction_months} months': [
                        predicted_subs[-1],
                        predicted_views[-1],
                        predicted_total_videos[-1],
                        predicted_views[-1] / predicted_subs[-1],
                        predicted_views[-1] / predicted_total_videos[-1]
                    ]
                })
                
                kpi_display = kpi_data.copy()
                for col in ['Current', f'After {prediction_months} months']:
                    kpi_display[col] = kpi_display[col].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(kpi_display, use_container_width=True)
                
                # Revenue projection
                st.markdown("---")
                st.markdown("#### 💰 Estimated Revenue Projection")
                st.info("Note: These are rough estimates based on average YouTube CPM rates")
                
                cpm_rates = {
                    'Entertainment': 2.5,
                    'Gaming': 3.0,
                    'Music': 1.5,
                    'Education': 4.0,
                    'Tech': 5.0,
                    'Default': 2.0
                }
                
                category_cpm = cpm_rates.get(channel_data['categoryTitle'], cpm_rates['Default'])
                
                monthly_revenue = []
                for views in predicted_views:
                    revenue = (views / 1000) * category_cpm * 0.55
                    monthly_revenue.append(revenue)
                
                fig_revenue = px.area(
                    x=months,
                    y=monthly_revenue,
                    title='Estimated Monthly Revenue Projection',
                    labels={'x': 'Month', 'y': 'Revenue ($)'},
                    color_discrete_sequence=['#ff0000']
                )
                fig_revenue.update_layout(height=400)
                st.plotly_chart(fig_revenue, use_container_width=True)
                
                total_revenue = sum(monthly_revenue)
                st.success(f"""
                **Revenue Projection Summary**  
                • Estimated CPM: ${category_cpm:.2f}  
                • Total projected revenue: ${total_revenue:,.2f}  
                • Average monthly revenue: ${total_revenue/prediction_months:,.2f}
                """)
            
            # ML-based predictions
            st.markdown("---")
            st.markdown("### 🤖 Machine Learning Predictions")
            
            if st.button("Generate ML Predictions", type="primary"):
                with st.spinner("Running machine learning models..."):
                    try:
                        rf_channel = models['rf_channel']
                        le_channel = models['labelenc_channel']
                        
                        if channel_data['categoryTitle'] in le_channel.classes_:
                            cat_encoded = le_channel.transform([channel_data['categoryTitle']])[0]
                        else:
                            cat_encoded = 0
                        
                        features = np.array([[
                            channel_data['viewCount'],
                            channel_data['videoCount'],
                            channel_data.get('avg_video_views', 0),
                            channel_data.get('channel_age_days', 365),
                            cat_encoded
                        ]])
                        
                        predicted_subs_ml = rf_channel.predict(features)[0]
                        
                        st.success(f"""
                        **ML Model Prediction**  
                        • Potential subscribers: {predicted_subs_ml:,.0f}  
                        • Difference from current: {predicted_subs_ml - current_subs:,.0f} ({(predicted_subs_ml/current_subs - 1)*100:.1f}%)
                        """)
                        
                        if hasattr(rf_channel, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': ['Total Views', 'Video Count', 'Avg Video Views', 'Channel Age', 'Category'],
                                'Importance': rf_channel.feature_importances_
                            }).sort_values('Importance', ascending=True)
                            
                            fig_importance = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Feature Importance for Subscriber Prediction',
                                color='Importance',
                                color_continuous_scale='Reds'
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error in ML prediction: {str(e)}")
        
        else:
            st.warning("No channels found matching your search.")

# Main dashboard
def main_dashboard():
    try:
        channels_df, videos_df = load_data()
        models = load_models()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <img src='https://cdn.dribbble.com/userupload/22057406/file/original-102766353c43b90107b9bb69d981be3a.gif' 
                     style='max-width: 300px; width: 100%; border-radius: 10px;'>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"### 👋 Welcome, {st.session_state.username.split('@')[0]}")
        st.markdown("---")
        
        # Navigation
        pages = {
            "Channel Insights": "🔍",
            "Channel Comparison": "⚖️",
            "Predictions & Growth": "🔮"
        }
        
        selected_page = st.radio(
            "Navigate to",
            list(pages.keys()),
            format_func=lambda x: f"{pages[x]} {x}"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Dataset Overview")
        st.info(f"""
        • Total Channels: {len(channels_df):,}  
        • Total Videos: {len(videos_df):,}  
        • Categories: {channels_df['categoryTitle'].nunique()}
        """)
        
        st.markdown("---")
         
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <p style='font-size: 0.9rem;'>Designed by: <strong>Anuj & Garema</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    
    # Main content
    if selected_page == "Channel Insights":
        channel_insights_page(channels_df, videos_df)
    elif selected_page == "Channel Comparison":
        channel_comparison_page(channels_df, videos_df)
    elif selected_page == "Predictions & Growth":
        predictions_page(channels_df, videos_df, models)

# Main app
def main():
    if not st.session_state.logged_in:
        
        st.markdown("""
        <style>
        /* Black background for login */
        .stApp {
            background-color: #000000;
        }
        .main {
            background-color: #000000;
            color: white;
        }
        /* Style inputs for dark theme */
        .stTextInput > div > div > input {
            background-color: #333333;
            color: white;
            border: 1px solid #555555;
        }
        .stTextInput label {
            color: white;
        }
        .stButton > button {
            background-color: #ff0000;
            color: white;
            border: none;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #000000;
        }
        .stTabs [data-baseweb="tab"] {
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()
