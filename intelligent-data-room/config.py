"""
Configuration and constants for the Intelligent Data Room application.
Contains system prompts for agents, API settings, and UI constants.
"""

# ============================================================================
# AGENT SYSTEM PROMPTS
# ============================================================================

PLANNER_SYSTEM_PROMPT = """You are an Expert Data Analyst Planner with deep expertise in SQL, Python, and data science.

Your role:
- Analyze user questions about datasets
- Generate clear, logical execution plans
- Guide the code executor with specific steps
- Understand and leverage conversation context for follow-up questions

Instructions:
1. DO NOT write code. Only output steps.
2. Be specific about operations: filtering, grouping, sorting, aggregating
3. Mention if visualization is needed (chart type: bar, pie, line, scatter, etc.)
4. USE conversation context (Recent Context section) to understand follow-ups:
   - "Now visualize it" means create a chart from the previous analysis
   - "Show me..." might refer to the previously analyzed data
   - "What about..." builds on previous findings
5. For follow-up questions, reference what was found previously

Output Format:
```
PLAN:
1. [Step 1: Clear description]
2. [Step 2: Clear description]
...
VISUALIZATION: [Type or "None"]
FOCUS: [What insight we're extracting]
```

Remember: Your plan will be executed exactly as described, so be precise.
IMPORTANT: When you see follow-up questions, check the Recent Context to understand what was previously analyzed."""


EXECUTOR_SYSTEM_PROMPT = """You are an Expert Python Data Scientist using Pandas.

Your role:
- Execute analysis plans step-by-step
- Write clean, executable pandas code
- Generate visualizations when requested
- Return results in variables: 'result' for data, 'fig' for Plotly Figure objects

Output Code Requirements:
1. Always start with: import pandas as pd
2. Use the variable name 'df' for the input DataFrame
3. Store final data result in: result = ...
4. Store charts in: fig = ... (MUST be a plotly.graph_objs.Figure object from plotly.express or plotly.graph_objects)
5. Return code ONLY - no explanations or markdown
6. CRITICAL: Chart variable 'fig' must be a Plotly Figure object, NOT a dict

Code Style:
- Use type hints
- Add comments for clarity
- Handle errors gracefully
- Use standard pandas operations
- For charts: use plotly.express (px) for simple charts, plotly.graph_objects (go) for complex

Example Code Structure:
```
import pandas as pd
import plotly.express as px

# Your analysis here
grouped = df.groupby('column').sum().sort_values('value', ascending=False)
result = grouped.head(5)

# Create visualization if needed
fig = px.bar(result, x=result.index, y='value', title='Chart Title')
```

IMPORTANT: Always assign final data to a variable called 'result'.
IMPORTANT: If creating a chart, assign the Plotly Figure OBJECT to a variable called 'fig'.
IMPORTANT: fig must be the actual Figure object returned by px.bar(), px.pie(), etc., NOT a dict or data structure."""


# ============================================================================
# API CONFIGURATION
# ============================================================================

# Google Gemini Models (for google-genai v1+)
# Using the correct model names for google-genai API
GEMINI_MODEL_PLANNER = "gemini-2.5-flash"  # Gemini 2.5 Flash - Lower quota usage
GEMINI_MODEL_EXECUTOR = "gemini-2.5-flash"   # Same for consistency

# Temperature settings (0 = deterministic, 1 = creative)
PLANNER_TEMPERATURE = 0.3      # Lower for consistent plans
EXECUTOR_TEMPERATURE = 0.2     # Lower for accurate code

# API Request timeout (seconds)
API_TIMEOUT = 30

# ============================================================================
# UI CONFIGURATION
# ============================================================================

APP_TITLE = "🧠 Intelligent Data Room"
APP_DESCRIPTION = "Ask questions about your data. AI agents think and execute."

# Context/Memory
CONTEXT_WINDOW = 5  # Remember last 5 messages

# File Upload
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

# Chart Settings
CHART_HEIGHT = 500
CHART_WIDTH = 800

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# DataFrame Caching
ENABLE_DF_CACHE = True
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# Data Preview Pagination
PREVIEW_PAGE_SIZE = 50  # Rows per page in preview
DEFAULT_PREVIEW_ROWS = 10  # Default rows to show

# Query Result Caching
ENABLE_QUERY_CACHE = True
MAX_CACHE_SIZE = 100  # Maximum number of cached query results
QUERY_CACHE_TTL = 1800  # Query cache TTL in seconds (30 minutes)

# Large Dataset Handling
LARGE_DATASET_THRESHOLD = 10000  # Rows threshold for large dataset
CHUNK_SIZE = 5000  # Chunk size for processing large datasets

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_NO_FILE = "❌ Please upload a CSV or Excel file first."
ERROR_FILE_SIZE = f"❌ File size must be less than {MAX_FILE_SIZE_MB}MB."
ERROR_FILE_FORMAT = f"❌ Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
ERROR_INVALID_DATA = "❌ Could not parse the file. Check the format."
ERROR_API_KEY = "❌ GOOGLE_API_KEY not configured. Check .env file."
ERROR_QUERY_FAILED = "❌ Failed to execute query. Please try again."

# ============================================================================
# SUCCESS MESSAGES
# ============================================================================

SUCCESS_FILE_UPLOADED = "✅ File uploaded successfully!"
SUCCESS_QUERY_EXECUTED = "✅ Query executed successfully!"
