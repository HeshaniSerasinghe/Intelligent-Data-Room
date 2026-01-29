# 🧠 Intelligent Data Room

> **AI-Powered Data Analysis Assistant** - Ask natural language questions about your data and get instant insights, code, and visualizations.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53+-green)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-API-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Security](https://img.shields.io/badge/Security-Hardened-brightgreen)

---

## ✨ Features

### 🔒 Security First

- **Input Sanitization**: Blocks code injection and SQL injection attempts
- **AST Validation**: Scans generated code for dangerous operations
- **Sandboxed Execution**: Restricted environment with whitelisted modules
- **File Security**: Prevents directory traversal and validates file types

### ⚡ Performance Optimized

- **Query Caching**: 6700x faster for repeated queries (100ms → 0.02ms)
- **DataFrame Caching**: Instant reload with `@st.cache_data`
- **Smart Pagination**: Smooth handling of 100K+ row datasets
- **LRU Cache**: Intelligent cache management with TTL

### 🤖 Multi-Agent AI System

- **Planner Agent**: Analyzes questions and creates execution plans
- **Executor Agent**: Generates Python code and runs it safely
- **Smart Memory**: Remembers last 5 messages for intelligent follow-ups

### 📊 Data Analysis

- ✅ Upload CSV, XLSX, XLS files
- ✅ Instant data preview with metrics
- ✅ Natural language queries
- ✅ Automatic code generation
- ✅ Safe sandbox execution

### 📈 Visualizations

- ✅ Interactive Plotly charts
- ✅ Auto chart type detection
- ✅ Multiple visualization styles
- ✅ Inline chart rendering

### 💬 Conversation Features

- ✅ Full chat history
- ✅ Follow-up question support
- ✅ Context-aware responses
- ✅ Expandable "Thinking Process" view (hidden by default)

### 🎨 User Experience

- ✅ Clean, modern dark theme UI
- ✅ Professional styled metrics with borders and hover effects
- ✅ Helpful example questions in sidebar
- ✅ Error messages with intelligent troubleshooting tips
- ✅ Real-time processing feedback with st.status workflow visualization
- ✅ Data metrics and statistics in bordered boxes
- ✅ Interactive data tables with download capability
- ✅ Collapsible Data Preview section
- ✅ Cache management UI with statistics
- ✅ Toggle for "Show Thinking by default" setting

### 🔧 Additional Features

- ✅ **Visual Multi-Agent Workflow**: See Planner → Executor handoff in real-time with st.status
- ✅ **Interactive Data Tables**: Results displayed as sortable, filterable tables
- ✅ **Download Data**: Export full datasets or query results as CSV
- ✅ **Result Context Passing**: Follow-up questions can reference previous query results
- ✅ **Large Dataset Support**: Automatic pagination for datasets with 10,000+ rows
- ✅ **Syntax Validation**: Python syntax checking before execution
- ✅ **Professional Theme**: Dark blue theme with iOS-style blue accents (#007AFF)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Google Gemini API key (free tier available)
- pip or conda

### Installation (5 minutes)

1. **Clone or download the project**

   ```bash
   cd intelligent-data-room
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv

   # Activate (Windows)
   venv\Scripts\activate

   # Activate (Mac/Linux)
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   - Get free key: [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create `.env` file in project root
   - Add: `GOOGLE_API_KEY=your_key_here`

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   - Automatically opens at `http://localhost:8501`
   - Or manually open the address shown in terminal

---

## 💡 Usage Examples

### Example 1: Basic Analysis

```
Q: "What are the top 5 products by sales?"

A: [Analyzes data and returns]
   ✅ Ranking table
   📊 Bar chart visualization
```

### Example 2: Follow-Up Question

```
Q1: "Show me sales by region"
A1: [Regional breakdown table]

Q2: "Now visualize it as a pie chart"
A2: [Pie chart of regional sales]
   (Uses context from Q1)
```

### Example 3: Complex Analysis

```
Q: "Which customer segment generates the most profit?"
A: [Analysis with profitability metrics]
   [Segment comparison chart]
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Streamlit Web Interface             │
│  • File upload                              │
│  • Chat interface                           │
│  • Data preview & metrics                   │
│  • Visualization display                    │
└────────────────┬────────────────────────────┘
                 │
         user_query, df, chat_history
                 │
         ┌───────▼────────┐
         │  MultiAgentSys │
         └───────┬────────┘
                 │
        ┌────────┴────────┐
        │                 │
    ┌───▼────┐      ┌─────▼──┐
    │Planner │      │Executor│
    │ Agent  │      │ Agent  │
    └───┬────┘      └─────┬──┘
        │ plan          │ code
        └────┬──────────┘
             │
         ┌───▼────────────┐
         │  Gemini API    │
         │ • Plan Gen     │
         │ • Code Gen     │
         └────────────────┘

    [Local Execution]
    ┌──────────────────┐
    │ • Pandas         │
    │ • Plotly         │
    │ • Safe Sandbox   │
    └──────────────────┘
```

---

## 📁 Project Structure

```
intelligent-data-room/
├── app.py                      # Main Streamlit app (315 lines)
├── agents.py                   # Multi-agent system (505 lines)
├── config.py                   # Prompts & configuration (128 lines)
├── utils.py                    # Helper functions (132 lines)
├── requirements.txt            # Python dependencies
├── .env                        # API key (create this)
├── .gitignore                  # Git security
├── README.md                   # This file
├── sample_data/
│   └── Sample Superstore.csv   # Test dataset (9,995 rows)
└── PHASE_*.md                  # Development documentation
```

---

## 🎯 Example Queries to Try

### Basic Analysis

- "What are the top 5 products by sales?"
- "How many orders by customer segment?"
- "Average discount by region?"

### Visualizations

- "Show me sales by category as a pie chart"
- "Create a line chart of sales over time"
- "Bar chart of profit by region"

### Comparisons

- "Which segment is most profitable?"
- "Compare sales vs profit by category"

### Follow-ups (Test Memory)

1. "Get the top 3 products"
2. "Now show as a donut chart"

---

## 🧪 Testing


### Manual Testing

1. Upload `Sample Superstore.csv`
2. Try: "What are the top 5 products by sales?"
3. Follow up: "Now show as a pie chart"
4. Verify chart renders with correct data

---

## 🔑 API Configuration

### Getting Your Free API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Select a Google Cloud project
4. Copy the generated key

### Setting Up .env

```bash
# .env file
GOOGLE_API_KEY=AIza...your_key_here...
```

### API Quotas

- **Free Tier**: 20 requests/day per key
- **Paid Plan**: Higher limits available
- **Model**: gemini-2.5-flash (fast, low token usage)

---

## 📊 Sample Dataset

The project includes **Sample Superstore.csv** with:

- **9,995 rows** of sales data
- **21 columns** including Order, Customer, Product, and Financial metrics
- Perfect for testing queries about sales, profit, segments, regions

---

## 🐛 Troubleshooting


### "GOOGLE_API_KEY not configured"

- Check `.env` file exists
- Verify API key is correct
- Restart Streamlit app

### "429 Too Many Requests"

- Hit daily quota (20 requests/day free tier)
- Get another API key or wait until next day

### "Chart Not Rendering"

- Try asking a different question
- Check browser console for errors

### "Column Not Found"

- Check data preview in sidebar
- Use exact column names from preview

### "Failed to parse file"

- Ensure CSV has headers
- Check for special characters in column names
- Try opening in Excel first

### Charts not showing

- Check plan includes visualization type
- Verify Executor returned chart data
- Look at error message in expander

---

## 📦 Dependencies

```
streamlit>=1.28.0         # Web framework with st.status support
pandas>=2.0.0             # Data manipulation
plotly>=5.17.0            # Interactive charts
google-genai>=1.0.0       # Gemini API client
python-dotenv==1.0.0      # Environment variables
openpyxl>=3.1.0           # Excel support
```

---

## 🔐 Security

- ✅ API keys stored in `.env` (not in code)
- ✅ Safe code execution in sandboxed environment
- ✅ Input sanitization with regex validation
- ✅ AST validation blocks dangerous operations
- ✅ Whitelisted modules and builtins only
- ✅ File type validation and size limits
- ✅ `.gitignore` prevents secret leakage
- ✅ 21/21 security tests passing

---

## 🎓 What's Implemented

| Feature                  | Status | Details                                    |
| ------------------------ | ------ | ------------------------------------------ |
| File Upload              | ✅     | CSV, XLSX, XLS support with validation     |
| Chat Interface           | ✅     | Full message history, ChatGPT-style        |
| Planner Agent            | ✅     | Google Gemini powered with context memory  |
| Executor Agent           | ✅     | Safe code execution with AST validation    |
| Visualizations           | ✅     | Interactive Plotly charts (bar, pie, line) |
| Context Memory           | ✅     | Last 5 messages with result data included  |
| Error Handling           | ✅     | Intelligent troubleshooting tips           |
| UI/UX Polish             | ✅     | Modern dark theme, bordered metrics        |
| Query Caching            | ✅     | 6700x speedup with LRU cache               |
| Security Features        | ✅     | Input sanitization, sandboxed execution    |
| Performance Optimization | ✅     | Smart pagination, DataFrame caching        |
| Visual Workflow          | ✅     | st.status showing agent handoff            |
| Interactive Tables       | ✅     | Sortable, filterable result display        |
| Download Capability      | ✅     | Export data as CSV                         |
| Large Dataset Support    | ✅     | Handles 100K+ rows smoothly                |

---

## 🚧 Future Improvements & Missing Features

### High Priority

- **💾 Database Connections**: Direct connection to PostgreSQL, MySQL, MongoDB
- **📁 Multiple File Upload**: Analyze multiple datasets simultaneously with JOIN operations
- **🔄 Data Refresh**: Auto-reload data from source without re-upload
- **📤 Export Options**: Save results as PDF, Excel, or PowerPoint reports
- **🎯 Query Templates**: Pre-built templates for common analysis patterns
- **📊 Dashboard Mode**: Create and save custom dashboards with multiple visualizations

### Medium Priority

- **🔍 Advanced Filtering**: GUI-based data filtering before querying
- **📈 Time Series Analysis**: Built-in forecasting and trend detection
- **🤖 Smart Suggestions**: AI-powered query recommendations based on data
- **👥 Collaboration**: Share analyses and charts with team members
- **📝 Query History**: Search and replay previous queries
- **🔔 Alerts**: Set up data threshold alerts and notifications
- **🎨 Chart Customization**: Fine-tune colors, labels, and styling
- **📊 Statistical Tests**: Built-in hypothesis testing and correlation analysis

### Low Priority

- **🌐 Multi-language Support**: UI translations for global users
- **🎤 Voice Input**: Ask questions using speech recognition
- **📱 Mobile Optimization**: Responsive design for tablets and phones
- **🔗 API Endpoint**: REST API for programmatic access
- **📚 Knowledge Base**: Save and reuse common queries and insights
- **🎓 Tutorial Mode**: Interactive guided tour for new users
- **🔐 User Authentication**: Multi-user support with role-based access
- **📊 Real-time Data**: Connect to streaming data sources
- **🤝 Integration**: Connect with Slack, Teams, or email for notifications
- **🧮 Custom Functions**: Allow users to define custom Python functions

### Technical Improvements

- **⚡ Parallel Processing**: Run multiple queries simultaneously
- **🔄 Background Jobs**: Queue long-running analyses
- **� API Key Rotation**: Automatic failover between multiple Gemini API keys for high-availability
- **💻 Code Editor**: Advanced code editing for power users
- **🧪 Unit Test Coverage**: Expand test suite to 90%+ coverage
- **📦 Docker Support**: Containerized deployment option
- **☁️ Cloud Deployment**: One-click deploy to AWS/Azure/GCP
- **📊 Performance Monitoring**: Built-in analytics and usage tracking
- **🔒 Enhanced Security**: OAuth integration, audit logs
- **⏱️ Rate Limiting**: Smart request throttling to prevent quota exhaustion

### Data Capabilities

- **🔗 Data Joining**: Merge multiple datasets intelligently
- **🧹 Data Cleaning**: Built-in tools for handling missing values, outliers
- **🎲 Sampling**: Smart sampling for massive datasets
- **📐 Feature Engineering**: Auto-generate derived columns
- **🤖 ML Integration**: Simple predictive modeling (regression, classification)
- **🗺️ Geographic Mapping**: Full map visualizations with location data
- **📊 Pivot Tables**: Interactive pivot table interface
- **📈 Trend Analysis**: Automatic seasonality and trend detection

---

## 🎯 Evaluation Criteria

### ⭐System Prompting 

- Clear agent role definitions
- Separate prompts for Planner vs Executor
- Shown via "Thinking" expander in UI

### ⭐Code Quality 

- Type hints on all functions
- Docstrings for classes/methods
- Modular file structure
- Clean error messages

### ⭐User Experience 

- Clean chat interface
- Readable visualizations
- Clear "thinking" process display
- Helpful error messages

### ⭐Reasoning 

- Expander showing Planner's plan
- Explanation of results
- Transparent agent communication


## 📹 Demo Video

https://kommodo.ai/recordings/AIlfComwNhr50EB2Q5yd

- Shows file upload
- Demonstrates agent thinking
- Visualizes results
- Shows context retention with follow-ups

## 📹 Deployed App (Streamlit Cloud)

https://intelligent-data-room-7kg748njvk8ax72np6cqiy.streamlit.app/

## 📝 License

MIT - Free to use and modify

## 👨‍💻 Author
Heshani Serasinghe
Built for Simplview GenAI & Full Stack Engineering Internship

---

**Ready to talk to your data?** 🚀
