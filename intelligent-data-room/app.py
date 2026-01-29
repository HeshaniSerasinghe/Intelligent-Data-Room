"""
Intelligent Data Room - Main Streamlit Application

A multi-agent system where users can upload CSV/Excel files and 
ask natural language questions about their data.

Agents:
- Planner: Generates execution plans
- Executor: Executes plans and returns results

Features:
- Natural language queries
- Automatic code generation and execution
- Interactive chart visualization
- Conversation memory for follow-ups
- User-friendly error handling
"""

import streamlit as st
import pandas as pd
import os
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from config import (
    APP_TITLE,
    APP_DESCRIPTION,
    CONTEXT_WINDOW,
    ERROR_NO_FILE,
    ERROR_API_KEY,
    SUCCESS_FILE_UPLOADED,
    ENABLE_DF_CACHE,
    CACHE_TTL,
    PREVIEW_PAGE_SIZE,
    DEFAULT_PREVIEW_ROWS,
)
from agents import MultiAgentSystem
from utils import (
    load_dataframe, 
    sanitize_query, 
    extract_chart_type,
    QueryCache,
    generate_query_hash,
    is_large_dataset,
)

# ============================================================================
# ERROR DIAGNOSIS HELPER
# ============================================================================

def get_troubleshooting_tips(error_text: str) -> str:
    """
    Analyze error and provide specific troubleshooting tips.
    
    Args:
        error_text: The error message string
        
    Returns:
        Formatted troubleshooting tips
    """
    error_lower = error_text.lower()
    
    # API Key issues
    if "api" in error_lower and ("key" in error_lower or "invalid" in error_lower or "authentication" in error_lower):
        return """
        **🔑 API Key Issue Detected**
        
        Your API key is missing or invalid. Here's how to fix it:
        
        1. **Get your API key:**
           - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
           - Click "Create API Key"
           - Copy the key
        
        2. **Update your .env file:**
           - Open `.env` in the project root
           - Add: `GOOGLE_API_KEY=your_api_key_here`
           - Save the file
        
        3. **Restart the app** and try again
        
        ℹ️ The app requires a valid Google Gemini API key to work.
        """
    
    # Rate limit / Quota exceeded
    elif "quota" in error_lower or "rate limit" in error_lower or "429" in error_lower or "too many" in error_lower:
        return """
        **⏱️ API Quota/Rate Limit Exceeded**
        
        You've hit the API usage limit. Here are your options:
        
        1. **Wait a while:**
           - API limits reset after some time (usually hourly)
           - Try again in 5-10 minutes
        
        2. **Upgrade your plan:**
           - Go to [Google Cloud Console](https://console.cloud.google.com/)
           - Check your billing and plan options
           - Consider upgrading to a paid plan for higher limits
        
        3. **Rephrase your question:**
           - Simpler questions use fewer API tokens
           - Try asking something more concise
        
        💡 Tip: You have free tier limits. Monitor usage in your API console.
        """
    
    # Network/Connection issues
    elif "connection" in error_lower or "timeout" in error_lower or "http" in error_lower or "network" in error_lower:
        return """
        **🌐 Network/Connection Error**
        
        There's a problem connecting to the API. Try these steps:
        
        1. **Check your internet connection**
           - Make sure you're connected to the internet
           - Try refreshing the page (F5)
        
        2. **Wait and retry:**
           - The API service might be temporarily down
           - Wait 30 seconds and try again
        
        3. **Check your firewall:**
           - Some firewalls block API calls
           - If behind corporate firewall, check with IT
        
        4. **Restart the app:**
           - Press Ctrl+C in the terminal and restart
           - This often resolves connection issues
        
        💡 If the problem persists, check: https://status.google.com/
        """
    
    # No results / Empty response
    elif "no result" in error_lower or "returned no data" in error_lower or "empty" in error_lower:
        return """
        **📭 No Results Found**
        
        Your query didn't return any data. Here's what to try:
        
        1. **Check your question:**
           - Are you asking about columns that exist in your data?
           - Try simpler questions first
           - Example: "Show me the first 5 rows"
        
        2. **Verify your data:**
           - Look at the file info on the left
           - Make sure the file has data (Rows > 0)
           - Check that columns are spelled correctly
        
        3. **Rephrase your question:**
           - Try: "What are the column names?"
           - Try: "How many rows do I have?"
           - Then ask more specific questions
        
        4. **Try a different approach:**
           - Ask about a different column
           - Ask for data grouped differently
           - Ask for a simpler analysis first
        
        💡 Pro Tip: Start with exploratory questions before complex analysis.
        """
    
    # Column not found
    elif "column" in error_lower and ("not" in error_lower or "found" in error_lower):
        return """
        **🔍 Column Not Found**
        
        The dataset doesn't have the column you're asking about. Here's how to fix it:
        
        1. **Check available columns:**
           - Look at the sidebar to see what data you have
           - Column names are case-sensitive
        
        2. **Try valid column names:**
           - Use exact names as shown in your file
           - Example: "Sales" not "sales" (if capitalized in file)
        
        3. **Ask what columns exist:**
           - Try: "What columns do I have?"
           - Try: "List all columns in my data"
           - This helps you see valid column names
        
        4. **Rephrase with correct column names:**
           - Once you know the columns, ask again with correct names
        
        💡 Hint: You can see the first few rows in the sidebar "Data Preview"
        """
    
    # Syntax error in generated code
    elif "syntax" in error_lower or "indented block" in error_lower:
        return """
        **⚙️ Code Generation Error**
        
        The AI generated invalid code. Here's what to do:
        
        1. **Try rephrasing your question:**
           - Be more specific about what you want
           - Avoid ambiguous terms
           - Use simpler language
        
        2. **Break complex questions down:**
           - Instead of: "Compare sales by region over time with charts"
           - Try first: "What are sales by region?"
           - Then: "Show a chart of sales by region"
        
        3. **Provide more context:**
           - Mention specific columns if possible
           - Example: "Show sales (column) by region (column)"
        
        4. **Try again:**
           - Sometimes just retrying works
           - The AI will generate different code
        
        💡 Complex multi-step queries sometimes cause this. Keep questions focused.
        """
    
    # Generic/Unknown error
    else:
        return f"""
        **❓ Unexpected Error**
        
        Error message: `{error_text}`
        
        Try these general troubleshooting steps:
        
        1. **Refresh the page** (F5)
           - Clear any temporary issues
        
        2. **Check your API key:**
           - See the "🔑 API Key Issue" section if API-related
        
        3. **Try a simpler question:**
           - Start with basic exploratory questions
           - Example: "Show me first 10 rows"
        
        4. **Check the browser console:**
           - Press F12 → Console tab
           - Look for detailed error messages
        
        5. **Restart the app:**
           - Close and reopen the browser tab
           - Or restart the Streamlit app
        
        6. **Check project files:**
           - Make sure .env file exists with API key
           - Make sure all Python files are saved
        
        ⚠️ If the error persists, check:
        - [Google API Status](https://status.google.com/)
        - Your [API Console](https://console.cloud.google.com/)
        - Browser console (F12) for detailed logs
        """

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_dataframe_cached(file_path: str, file_name: str) -> tuple:
    """
    Cached version of load_dataframe for better performance.
    
    Args:
        file_path: Path to file
        file_name: Name of file (for cache key)
        
    Returns:
        Tuple of (DataFrame or None, error_message)
    """
    logger.info(f"🔄 Loading DataFrame (cache miss): {file_name}")
    return load_dataframe(file_path)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_dataframe_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get cached DataFrame statistics.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with stats
    """
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
    }

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Intelligent Data Room",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Hide Streamlit branding for professional look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container padding */
    .main {
        padding-top: 1rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 5px;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
    }
    
    /* Chart styling - make charts pop */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        padding: 10px;
        background-color: #262730;
    }
    
    /* Status container styling */
    .stStatus {
        border-radius: 8px;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Bordered stat boxes for neat appearance */
    [data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 2px solid #007AFF;
        border-radius: 10px;
        padding: 15px 10px;
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
        border-color: #0A84FF;
    }
    
    /* Metric label styling */
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - FILE UPLOAD & INFO
# ============================================================================

with st.sidebar:
    st.header("📂 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX, XLS (Max 10MB)"
    )
    
    if uploaded_file:
        # Save file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load DataFrame with caching
        if ENABLE_DF_CACHE:
            df, error = load_dataframe_cached(temp_path, uploaded_file.name)
        else:
            df, error = load_dataframe(temp_path)
        
        if df is not None:
            st.success(SUCCESS_FILE_UPLOADED)
            st.session_state["df"] = df
            st.session_state["file_name"] = uploaded_file.name
            
            # Check if large dataset
            is_large = is_large_dataset(df)
            if is_large:
                st.info(f"📊 Large dataset detected ({len(df):,} rows). Using optimized display.")
            
            # Show interactive data preview with enhanced features
            with st.expander("📊 Data Preview (Interactive)", expanded=True):
                # Stats first - more prominent
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("📋 Columns", df.shape[1])
                with col3:
                    stats = get_dataframe_stats(df)
                    st.metric("💾 Memory", f"{stats['memory_mb']:.2f} MB")
                with col4:
                    # Add dataset type indicator
                    dataset_type = "Large" if is_large else "Standard"
                    st.metric("📈 Type", dataset_type)
                
                st.caption(f"**Columns:** {', '.join(df.columns.tolist()[:8])}{'...' if len(df.columns) > 8 else ''}")
                st.divider()
                
                # Interactive dataframe with column configuration
                if is_large:
                    max_page = (len(df) - 1) // PREVIEW_PAGE_SIZE
                    page = st.slider("📄 Page", 0, max_page, 0, help=f"Showing {PREVIEW_PAGE_SIZE} rows per page")
                    start_idx = page * PREVIEW_PAGE_SIZE
                    end_idx = min(start_idx + PREVIEW_PAGE_SIZE, len(df))
                    display_df = df.iloc[start_idx:end_idx]
                    st.caption(f"Showing rows {start_idx:,} to {end_idx:,} of {len(df):,}")
                else:
                    preview_rows = st.slider("📄 Rows to preview", 5, min(100, len(df)), min(DEFAULT_PREVIEW_ROWS, len(df)))
                    display_df = df.head(preview_rows)
                
                # Build column config for numeric columns with formatting
                column_config = {}
                for col in display_df.columns:
                    if pd.api.types.is_numeric_dtype(display_df[col]):
                        # Format numeric columns nicely
                        if display_df[col].max() > 1000:
                            column_config[col] = st.column_config.NumberColumn(
                                col,
                                format="%.2f",
                                help=f"Numeric column: {col}"
                            )
                
                # Show interactive dataframe with sorting, filtering
                st.dataframe(
                    display_df,
                    width='stretch',
                    height=300,
                    column_config=column_config if column_config else None
                )
                
                # Add download button
                st.download_button(
                    label="📥 Download Full Dataset (CSV)",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=f"data_{st.session_state.get('file_name', 'export')}.csv",
                    mime='text/csv',
                    help="Download the complete dataset as CSV",
                    use_container_width=True
                )
        else:
            st.error(f"❌ {error}")
    
    # Info section
    st.divider()
    st.header("ℹ️ How It Works")
    st.markdown("""
    **1. Upload** your data (CSV/Excel)
    
    **2. Ask** a natural language question
    
    **3. Planner** 🤔 analyzes your question
    
    **4. Executor** ⚙️ generates and runs code
    
    **5. Results** are displayed with charts
    
    **💡 Tip:** Use follow-ups like "Now show as a pie chart"
    """)
    
    # Usage tips
    with st.expander("💬 Example Questions"):
        st.markdown("""
        - "What are the top 5 products by sales?"
        - "Show me sales by region as a bar chart"
        - "How many customers in each segment?"
        - "What's the profit margin by category?"
        - "Now visualize it as a pie chart"
        """)
    
    # Settings
    st.divider()
    st.header("⚙️ Settings")
    show_thinking = st.checkbox(
        "Show 'Thinking' by default",
        value=False,
        help="When OFF, thinking is hidden but expandable"
    )
    st.session_state["show_thinking"] = show_thinking
    
    # Cache management
    if st.button("🗑️ Clear Cache", help="Clear query result cache"):
        st.session_state["query_cache"].clear()
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()
    
    # Cache stats
    if st.session_state.get("query_cache"):
        cache_stats = st.session_state["query_cache"].get_stats()
        st.caption(f"📊 Cache: {cache_stats['size']}/{cache_stats['max_size']} items")

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if "df" not in st.session_state:
    st.session_state["df"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "show_thinking" not in st.session_state:
    st.session_state["show_thinking"] = False

# Initialize query cache
if "query_cache" not in st.session_state:
    st.session_state["query_cache"] = QueryCache()

if "multi_agent_system" not in st.session_state:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ **API Key Missing**")
        st.markdown("""
        Please configure your Google Gemini API key:
        
        1. Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. Create a `.env` file in the project root
        3. Add: `GOOGLE_API_KEY=your_key_here`
        4. Refresh the page
        """)
        st.stop()
    try:
        st.session_state["multi_agent_system"] = MultiAgentSystem(api_key)
    except Exception as e:
        st.error(f"❌ **Failed to initialize**: {str(e)}")
        st.stop()

# ============================================================================
# MAIN CONTENT AREA - CHAT INTERFACE (like ChatGPT/Gemini)
# ============================================================================

# Header (stays at top)
col1, col2 = st.columns([3, 1])
with col1:
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
with col2:
    if st.session_state["df"] is not None:
        st.metric("Messages", len([m for m in st.session_state["messages"] if m["role"] == "user"]))

# Main content
if st.session_state["df"] is None:
    st.info("👆 **Upload a file to get started!** Choose a CSV or Excel file to begin analyzing your data.")
else:
    # File info (stays at top)
    file_col1, file_col2, file_col3 = st.columns(3)
    with file_col1:
        st.metric("📁 File", st.session_state.get('file_name', 'Unknown')[:20])
    with file_col2:
        st.metric("📊 Rows", f"{st.session_state['df'].shape[0]:,}")
    with file_col3:
        st.metric("📋 Columns", st.session_state['df'].shape[1])
    
    st.divider()
    
    # === CHAT MESSAGES SECTION (SCROLLABLE) ===
    # Display all messages from session state
    if len(st.session_state["messages"]) > 0:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                # Main content
                st.markdown(msg["content"])
                
                # Show result as interactive table if available
                if msg.get("result") is not None:
                    result = msg["result"]
                    if isinstance(result, pd.DataFrame):
                        if len(result) > 0:
                            st.dataframe(
                                result,
                                width='stretch',
                                height=min(400, (len(result) + 1) * 35)
                            )
                    elif isinstance(result, pd.Series):
                        # Convert Series to DataFrame for better display
                        result_df = result.reset_index()
                        result_df.columns = ['Index', 'Value']
                        st.dataframe(
                            result_df,
                            width='stretch',
                            height=min(400, (len(result_df) + 1) * 35)
                        )
                
                # Show plan if available (with setting to hide by default)
                if msg.get("plan"):
                    with st.expander("📋 View Thinking Process", expanded=st.session_state["show_thinking"]):
                        st.markdown(msg["plan"])
                
                # Show chart if available
                if msg.get("chart"):
                    try:
                        st.plotly_chart(msg["chart"], width='stretch')
                    except Exception as e:
                        st.warning(f"⚠️ Could not display chart: {str(e)}")
    else:
        st.info("💬 **No messages yet.** Start by asking a question about your data!")
    
    # === INPUT SECTION (FIXED AT BOTTOM) ===
    # Add spacing
    st.divider()
    st.markdown("### 💬 Ask a Question")
    
    # Chat input
    user_input = st.chat_input(
        "Ask a question about your data (e.g., 'What are the top 5 products by sales?')",
        key="chat_input"
    )
    
    # === PROCESS QUERY ===
    # Handle chat_input
    if user_input:
        # Sanitize and validate user input
        try:
            user_query = sanitize_query(user_input)
        except ValueError as e:
            # Handle validation errors from sanitize_query
            st.error(f"❌ **Invalid Query**: {str(e)}")
            st.info("💡 Please use natural language questions without code or special characters.")
            st.stop()
        
        # Add user message to state BEFORE processing
        st.session_state["messages"].append({
            "role": "user",
            "content": user_query,
        })
        
        # Generate query hash for caching
        query_hash = generate_query_hash(user_query, st.session_state["df"].shape)
        
        # Check cache first
        cached_result = st.session_state["query_cache"].get(query_hash)
        
        if cached_result is not None:
            # Use cached result
            logger.info("✅ Using cached query result")
            st.info("⚡ Using cached result (faster!)")
            result = cached_result
        else:
            # Visual multi-agent handoff using st.status
            with st.status("🤖 Orchestrating Multi-Agent System...", expanded=True) as status:
                try:
                    # Step 1: Planner Agent
                    st.write("🧠 **Planner Agent:** Analyzing query and data schema...")
                    plan = st.session_state["multi_agent_system"].planner.generate_plan(
                        user_query=user_query,
                        df=st.session_state["df"],
                        chat_history=st.session_state["messages"][:-1],
                    )
                    st.write("✅ **Planner Agent:** Execution plan generated")
                    
                    # Show plan preview
                    with st.expander("📋 View Generated Plan", expanded=False):
                        st.code(plan[:500] + "..." if len(plan) > 500 else plan, language="markdown")
                    
                    # Step 2: Executor Agent
                    st.write("⚙️ **Executor Agent:** Generating Python code...")
                    result_data, chart, explanation = st.session_state["multi_agent_system"].executor.execute_plan(
                        user_query=user_query,
                        plan=plan,
                        df=st.session_state["df"],
                    )
                    st.write("✅ **Executor Agent:** Code executed successfully")
                    
                    # Package result
                    result = {
                        "success": True,
                        "plan": plan,
                        "result": result_data,
                        "chart": chart,
                        "explanation": explanation,
                        "error": None,
                    }
                    
                    # Cache successful results
                    st.session_state["query_cache"].set(query_hash, result)
                    
                    # Update status to complete
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                    
                except Exception as e:
                    error_text = str(e)
                    status.update(label="❌ Analysis Failed", state="error", expanded=True)
                    
                    # Store error message with troubleshooting tips
                    troubleshooting = get_troubleshooting_tips(error_text)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"❌ **Unexpected error**: {error_text}\n\n---\n\n{troubleshooting}",
                    })
                    
                    # Log for debugging
                    logger.error(f"Unexpected error: {error_text}", exc_info=True)
                    
                    # Rerun to display the error with troubleshooting in chat
                    st.rerun()
        
        # Process result (whether from cache or fresh)
        try:
            if result["success"]:
                # Store assistant message with all details including raw result
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": result["explanation"],
                    "plan": result["plan"],
                    "chart": result["chart"],
                    "result": result["result"],  # Store raw result for table display
                })
                st.success("✅ Analysis complete!")
                st.rerun()  # Rerun to display the new messages
            else:
                # Handle specific error cases with targeted troubleshooting
                error_msg = result.get('error', 'Unknown error occurred')
                
                # Store error message with troubleshooting tips
                troubleshooting = get_troubleshooting_tips(error_msg)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"❌ **Analysis failed**: {error_msg}\n\n---\n\n{troubleshooting}",
                })
                
                # Rerun to display the error message with troubleshooting in chat
                st.rerun()
        except Exception as e:
            error_text = str(e)
            
            # Store error message with troubleshooting tips
            troubleshooting = get_troubleshooting_tips(error_text)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"❌ **Unexpected error**: {error_text}\n\n---\n\n{troubleshooting}",
            })
            
            # Log for debugging
            logger.error(f"Unexpected error: {error_text}", exc_info=True)
            
            # Rerun to display the error with troubleshooting in chat
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================


st.markdown("""
---
<div style='text-align: center; color: gray; font-size: 15px;'>
    © 2026 Intelligent Data Room | Made by Heshani Serasinghe
</div>
""", unsafe_allow_html=True)
