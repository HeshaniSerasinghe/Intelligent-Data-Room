"""
Multi-Agent System for the Intelligent Data Room.

Agents:
- PlannerAgent: Analyzes questions and generates execution plans
- ExecutorAgent: Executes plans and returns data/visualizations
"""

import os
import ast
import signal
from typing import Optional, Dict, List, Any, Tuple, Set
import google.genai
from google.genai import types as genai_types
import pandas as pd
import logging
import re
import io
from contextlib import redirect_stdout, redirect_stderr
from functools import wraps

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from config import (
    PLANNER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    GEMINI_MODEL_PLANNER,
    GEMINI_MODEL_EXECUTOR,
    PLANNER_TEMPERATURE,
    EXECUTOR_TEMPERATURE,
    ERROR_API_KEY,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CODE EXECUTION SECURITY
# ============================================================================

# Whitelist of allowed modules and functions
ALLOWED_MODULES = {
    'pandas',
    'pd',
    'plotly',
    'px',
    'go',
    'numpy',
    'np',
    'datetime',
    'math',
}

ALLOWED_BUILTINS = {
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter',
    'float', 'int', 'len', 'list', 'map', 'max', 'min', 'range',
    'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip',
    'True', 'False', 'None',
    # Exception types that code might need to handle
    'KeyError', 'ValueError', 'TypeError', 'IndexError', 'AttributeError',
    'Exception', 'StopIteration',
    # Other useful builtins
    'print', 'type', 'isinstance', 'hasattr', 'getattr',
}

# Dangerous AST node types to block
DISALLOWED_NODES = {
    ast.Import,  # Will validate imports separately
    ast.ImportFrom,  # Will validate imports separately  
}

EXECUTION_TIMEOUT = 30  # seconds


class CodeValidationError(Exception):
    """Raised when generated code fails security validation."""
    pass


class ExecutionTimeoutError(Exception):
    """Raised when code execution exceeds timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise ExecutionTimeoutError("Code execution timed out")


def validate_code_ast(code: str) -> None:
    """
    Validate generated code using AST analysis.
    
    Args:
        code: Python code to validate
        
    Raises:
        CodeValidationError: If code contains disallowed operations
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"Syntax error in generated code: {e}")
    
    # Check for dangerous operations
    for node in ast.walk(tree):
        # Block eval, exec, compile (but allow __import__ as we handle it safely)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ('eval', 'exec', 'compile', 'open', 'input'):
                    raise CodeValidationError(
                        f"Disallowed function call: {node.func.id}"
                    )
        
        # Block attribute access to dangerous modules
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in ('os', 'sys', 'subprocess', 'importlib', '__builtins__'):
                    raise CodeValidationError(
                        f"Access to disallowed module: {node.value.id}"
                    )
        
        # Validate imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name not in ALLOWED_MODULES:
                        raise CodeValidationError(
                            f"Import of disallowed module: {module_name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name not in ALLOWED_MODULES:
                        raise CodeValidationError(
                            f"Import from disallowed module: {module_name}"
                        )
    
    logger.info("✅ Code passed AST validation")


def create_safe_globals() -> Dict[str, Any]:
    """
    Create a restricted global namespace for code execution.
    
    Returns:
        Dictionary with safe builtins and allowed modules
    """
    # Create restricted builtins
    safe_builtins = {
        name: __builtins__[name]
        for name in ALLOWED_BUILTINS
        if name in __builtins__
    }
    
    # Add restricted __import__ that only allows whitelisted modules
    def safe_import(name, *args, **kwargs):
        """Restricted import that only allows whitelisted modules."""
        base_module = name.split('.')[0]
        if base_module not in ALLOWED_MODULES:
            raise ImportError(f"Import of module '{name}' is not allowed")
        return __import__(name, *args, **kwargs)
    
    safe_builtins['__import__'] = safe_import
    
    return {
        '__builtins__': safe_builtins,
        'pd': pd,
    }


class PlannerAgent:
    """
    Agent 1: The Planner
    
    Responsibility:
    - Analyzes user queries
    - Understands data schema
    - Generates step-by-step execution plans
    - Does NOT touch actual data
    
    Output: A detailed plan with steps and visualization recommendation
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Planner Agent.
        
        Args:
            api_key: Google Gemini API key. If None, reads from environment.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(ERROR_API_KEY)
        
        self.client = google.genai.Client(api_key=self.api_key)
        self.model = GEMINI_MODEL_PLANNER
        self.system_prompt = PLANNER_SYSTEM_PROMPT
        logger.info("✅ PlannerAgent initialized")
    
    def analyze_schema(self, df: pd.DataFrame) -> str:
        """
        Analyze DataFrame schema and return a summary.
        
        Args:
            df: Input DataFrame
            
        Returns:
            String describing columns, types, and sample values
        """
        schema_info = f"""
Dataset Schema:
- Columns: {list(df.columns)}
- Data Types: {df.dtypes.to_dict()}
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Sample values:
{df.head(2).to_string()}
"""
        return schema_info
    
    def generate_plan(
        self,
        user_query: str,
        df: pd.DataFrame,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate an execution plan for the user's query.
        
        Args:
            user_query: Natural language question from user
            df: DataFrame to analyze
            chat_history: Previous Q&A pairs for context
            
        Returns:
            Execution plan as string
        """
        # Build context from chat history
        context = ""
        if chat_history:
            context = "\nRecent Context:\n"
            # Include last 5 messages (or fewer if less available) for better context
            recent_messages = chat_history[-5:]
            for i, msg in enumerate(recent_messages):
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                
                # Include result data if available for assistant messages
                if role == 'ASSISTANT' and msg.get('result') is not None:
                    result_data = msg.get('result')
                    # Format result data preview
                    if isinstance(result_data, pd.DataFrame):
                        if len(result_data) <= 10:
                            result_preview = f"\nResult Data:\n{result_data.to_string()}"
                        else:
                            result_preview = f"\nResult Data (showing first 10 of {len(result_data)} rows):\n{result_data.head(10).to_string()}"
                        content += result_preview
                    elif isinstance(result_data, pd.Series):
                        result_preview = f"\nResult Data:\n{result_data.to_string()}"
                        content += result_preview
                
                # Truncate if too long, but keep enough for context
                if len(content) > 500:
                    content = content[:500] + "..."
                context += f"  {role}: {content}\n"
        
        # Analyze schema
        schema_info = self.analyze_schema(df)
        
        # Build the full prompt
        full_prompt = f"""{self.system_prompt}

{schema_info}
{context}

User Question: {user_query}

Generate a detailed execution plan:
"""
        
        try:
            # Format model name with 'models/' prefix for google-genai API
            model_id = f"models/{self.model}" if not self.model.startswith("models/") else self.model
            
            response = self.client.models.generate_content(
                model=model_id,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=PLANNER_TEMPERATURE,
                    max_output_tokens=1000,
                ),
            )
            plan = response.text
            if plan is None:
                raise ValueError("Failed to generate plan: response text is None")
            logger.info("✅ Plan generated successfully")
            return plan
        except Exception as e:
            logger.error(f"❌ Error generating plan: {str(e)}")
            raise


class ExecutorAgent:
    """
    Agent 2: The Executor
    
    Responsibility:
    - Takes execution plan from Planner
    - Generates Python code using PandasAI
    - Executes code on actual data
    - Returns results and visualizations
    
    Output: Analysis results with optional charts
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Executor Agent.
        
        Args:
            api_key: Google Gemini API key. If None, reads from environment.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(ERROR_API_KEY)
        
        self.client = google.genai.Client(api_key=self.api_key)
        self.model = GEMINI_MODEL_EXECUTOR
        self.system_prompt = EXECUTOR_SYSTEM_PROMPT
        logger.info("✅ ExecutorAgent initialized")
    
    def execute_plan(
        self,
        user_query: str,
        plan: str,
        df: pd.DataFrame
    ) -> Tuple[Any, Optional[str], str]:
        """
        Execute the plan on the DataFrame with security checks.
        
        Args:
            user_query: Original user question
            plan: Execution plan from PlannerAgent
            df: DataFrame to analyze
            
        Returns:
            Tuple of (result, chart_html_or_none, explanation)
        """
        try:
            # Step 1: Generate code from plan
            logger.info("📝 Generating code from plan...")
            code = self.generate_code(plan, df, user_query)
            
            # Step 2: Clean and validate code
            logger.info("🔍 Validating code...")
            code = self._clean_code(code)
            
            # Step 3: Execute code with security checks
            logger.info("⚙️ Executing code...")
            result, chart_html = self._execute_code(code, df, user_query)
            
            # Step 4: Generate explanation
            explanation = self._generate_explanation(user_query, result)
            
            logger.info("✅ Plan executed successfully")
            return result, chart_html, explanation
        
        except CodeValidationError as e:
            logger.error(f"❌ Security validation failed: {str(e)}")
            # Re-raise so app.py can handle with troubleshooting
            raise ValueError(f"Security validation failed: {str(e)}")
        
        except ExecutionTimeoutError as e:
            logger.error(f"❌ Execution timeout: {str(e)}")
            # Re-raise so app.py can handle with troubleshooting
            raise ValueError(f"Query execution timeout: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Error executing plan: {str(e)}")
            # Re-raise so app.py can handle with troubleshooting
            raise
    
    def _clean_code(self, code: str) -> str:
        """
        Clean generated code: remove markdown, fix imports, etc.
        
        Args:
            code: Generated code
            
        Returns:
            Cleaned executable code
        """
        # Remove markdown code blocks
        if code.startswith("```"):
            code = code.split("```")[1]
        if code.startswith("python"):
            code = code[6:]
        code = code.strip()
        
        # Ensure proper imports
        lines = code.split("\n")
        has_pandas = any("import pandas" in line for line in lines)
        has_plotly = any("import plotly" in line for line in lines)
        
        new_lines = []
        if not has_pandas:
            new_lines.append("import pandas as pd")
        if not has_plotly and PLOTLY_AVAILABLE:
            new_lines.append("import plotly.express as px")
        
        new_lines.extend(lines)
        return "\n".join(new_lines)
    
    def _execute_code(
        self,
        code: str,
        df: pd.DataFrame,
        user_query: str
    ) -> Tuple[Any, Optional[str]]:
        """
        Execute generated code in a safe sandbox with security restrictions.
        
        Args:
            code: Python code to execute
            df: DataFrame to work with
            user_query: Original query (for error context)
            
        Returns:
            Tuple of (result, chart_html_or_none)
            
        Raises:
            CodeValidationError: If code fails security validation
            ExecutionTimeoutError: If code execution exceeds timeout
        """
        # Step 1: Validate code with AST before execution
        logger.info("🔒 Validating code security...")
        try:
            validate_code_ast(code)
        except CodeValidationError as e:
            logger.error(f"Code validation failed: {e}")
            raise ValueError(f"Security validation failed: {e}")
        
        # Step 2: Create safe execution environment
        exec_globals = create_safe_globals()
        
        # Add allowed modules
        exec_globals.update({
            "pd": pd,
            "df": df.copy(),  # Work on a copy to prevent original modification
            "result": None,
            "fig": None,
        })
        
        if PLOTLY_AVAILABLE:
            exec_globals["px"] = px
            exec_globals["go"] = go
        
        try:
            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Step 3: Execute with timeout (Windows-compatible approach)
            logger.info(f"⚙️ Executing code with {EXECUTION_TIMEOUT}s timeout...")
            
            # Note: signal.alarm() doesn't work on Windows, so we use a try-except
            # For production, consider using threading.Timer or multiprocessing
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    exec(code, exec_globals, exec_globals)
                except Exception as exec_error:
                    # Capture execution errors
                    logger.error(f"Execution error: {exec_error}")
                    raise
            
            # Get result (last expression or 'result' variable)
            result = exec_globals.get("result", None)
            fig = exec_globals.get("fig", None)
            
            # Validate and process figure
            chart_html = None
            if fig is not None:
                # Log what we got
                logger.debug(f"Chart object type: {type(fig)}")
                logger.debug(f"Chart object: {fig}")
                
                # Check if it's a valid Plotly Figure
                if hasattr(fig, 'to_dict') and hasattr(fig, 'to_html'):
                    # It's a Plotly Figure object - return as-is
                    chart_html = fig
                    logger.info(f"✅ Valid Plotly Figure detected")
                elif isinstance(fig, dict):
                    # It's a dict - might be figure data, warn user
                    logger.warning(f"Chart is dict, not Figure object. Skipping chart display.")
                    chart_html = None
                else:
                    # Unknown type
                    logger.warning(f"Chart object is {type(fig).__name__}, expected Plotly Figure. Skipping.")
                    chart_html = None
            
            # If result is None but we had output, capture that
            if result is None and stdout_capture.getvalue():
                result = stdout_capture.getvalue()
            
            return result, chart_html
            
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            # Try to fix common syntax errors
            if "'result'" in str(e):
                # Missing result assignment, use last variable
                code_lines = code.split("\n")
                for i in range(len(code_lines)-1, -1, -1):
                    if code_lines[i].strip() and not code_lines[i].strip().startswith("#"):
                        code_lines[i] = f"result = {code_lines[i]}"
                        break
                return self._execute_code("\n".join(code_lines), df, user_query)
            raise
            
        except KeyError as e:
            logger.error(f"Missing column error: {e}")
            raise ValueError(f"Column not found in dataset: {str(e)}")
            
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            raise ValueError(f"Error executing generated code: {str(e)}")
    
    def _generate_explanation(self, user_query: str, result: Any) -> str:
        """
        Generate a human-friendly explanation of results.
        
        Args:
            user_query: Original user question
            result: Execution result
            
        Returns:
            Explanation string
        """
        try:
            if result is None:
                return "Query executed but returned no data."
            
            if isinstance(result, pd.DataFrame):
                # Format DataFrame results nicely
                if len(result) == 0:
                    return "Query returned no results."
                elif len(result) == 1:
                    # Single row - format as key-value pairs
                    row_data = result.iloc[0]
                    formatted = "**Result:**\n\n"
                    for col, val in row_data.items():
                        formatted += f"- **{col}:** {val}\n"
                    return formatted
                else:
                    # Multiple rows - indicate table view
                    return f"Found **{len(result)}** rows. See table below:"
            
            if isinstance(result, pd.Series):
                if len(result) <= 5:
                    # Small series - show as list
                    formatted = "**Results:**\n\n"
                    for idx, val in result.items():
                        formatted += f"- **{idx}:** {val}\n"
                    return formatted
                else:
                    return f"Series with {len(result)} values. See table below:"
            
            if isinstance(result, (int, float)):
                return f"**Result:** {result:,.2f}" if isinstance(result, float) else f"**Result:** {result:,}"
            
            if isinstance(result, dict):
                formatted = "**Result:**\n\n"
                for key, val in result.items():
                    formatted += f"- **{key}:** {val}\n"
                return formatted
            
            return f"**Result:** {str(result)[:200]}"
            
        except Exception as e:
            return f"Execution completed (could not format explanation: {str(e)})"
    
    def generate_code(
        self,
        plan: str,
        df: pd.DataFrame,
        user_query: str
    ) -> str:
        """
        Generate Python code based on the execution plan.
        
        Args:
            plan: Execution plan from Planner
            df: DataFrame schema
            user_query: Original question
            
        Returns:
            Generated Python code as string
        """
        schema_info = f"Columns: {list(df.columns)}\nShape: {df.shape}"
        
        prompt = f"""{self.system_prompt}

Dataset Schema:
{schema_info}

User Question: {user_query}

Execution Plan from Planner:
{plan}

Generate clean, executable Python code using pandas that follows this plan exactly.
Start with 'import pandas as pd' and 'import plotly.express as px'.
Return the code only, no explanations.
"""
        
        try:
            # Format model name with 'models/' prefix for google-genai API
            model_id = f"models/{self.model}" if not self.model.startswith("models/") else self.model
            
            response = self.client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=EXECUTOR_TEMPERATURE,
                    max_output_tokens=2000,
                ),
            )
            code = response.text
            if code is None:
                raise ValueError("Failed to generate code: response text is None")
            logger.info("✅ Code generated successfully")
            return code
        except Exception as e:
            logger.error(f"❌ Error generating code: {str(e)}")
            raise


class MultiAgentSystem:
    """
    Orchestrator for PlannerAgent and ExecutorAgent.
    
    Manages:
    - Communication between agents
    - State/context passing
    - Error handling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Multi-Agent System.
        
        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.planner = PlannerAgent(self.api_key)
        self.executor = ExecutorAgent(self.api_key)
        logger.info("✅ MultiAgentSystem initialized")
    
    def process_query(
        self,
        user_query: str,
        df: pd.DataFrame,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through both agents.
        
        Args:
            user_query: Natural language question
            df: DataFrame to analyze
            chat_history: Previous Q&A for context
            
        Returns:
            Dictionary with plan, result, chart, and explanation
        """
        try:
            # Step 1: Planner generates plan
            logger.info("🤔 Planner thinking...")
            plan = self.planner.generate_plan(user_query, df, chat_history)
            
            # Step 2: Executor executes plan
            logger.info("⚙️  Executor running...")
            result, chart, explanation = self.executor.execute_plan(user_query, plan, df)
            
            return {
                "success": True,
                "plan": plan,
                "result": result,
                "chart": chart,
                "explanation": explanation,
                "error": None,
            }
        except Exception as e:
            logger.error(f"❌ Error in query processing: {str(e)}")
            return {
                "success": False,
                "plan": None,
                "result": None,
                "chart": None,
                "explanation": None,
                "error": str(e),
            }
