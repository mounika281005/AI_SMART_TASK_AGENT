# -*- coding: utf-8 -*-
"""
Task Prioritizer Agent - Enhanced Version with Fixed Agent Behavior
Created on Wed May 14 13:20:11 2025
@author: merto
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI # Import Google's client
from langchain_core.tools import tool
from langchain.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
import dateparser
from typing import Optional, List, Dict, Any, Union
from collections import defaultdict
import uuid
from dateutil import parser as date_parser
import ast


logging.basicConfig(
    level=logging.INFO,  # Set the default level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# --- Agent Configuration Constants ---
AGENT_MAX_ITERATIONS = 5
AGENT_EARLY_STOPPING_METHOD = "force" # Standard Langchain method

# --- ActionTracker Thresholds ---
# Warn if tool calls exceed this before hitting max_iterations
ACTION_TRACKER_TOOL_CALL_WARNING_THRESHOLD = AGENT_MAX_ITERATIONS - 1 # e.g., 4 if max_iterations is 5
ACTION_TRACKER_MAX_CONSECUTIVE_FORMAT_ERRORS = 2
ACTION_TRACKER_MAX_SAME_ACTION_REPETITIONS = 2 # Stop if same action repeated more than this many times

# --- Constants ---
TASKS_FILE = "tasks.json"
TEMPLATES_FILE = "task_templates.json"
ANALYTICS_FILE = "task_analytics.json"

# Load environment variables
load_dotenv()
google_api_key_raw = os.getenv("GOOGLE_API_KEY") # Expecting the key in GOOGLE_API_KEY

if not google_api_key_raw:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it. This key should be from Google AI Studio.")

google_api_key = google_api_key_raw.strip()

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is empty or consists only of whitespace in .env file. Please provide a valid key from Google AI Studio.")

print("Attempting to initialize LLM with Google AI Studio key...")
try:
    # Initialize LLM with Google Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", # Using a common alias for the latest Gemini 1.5 Flash model
                                        # If "Gemini 2.0 Flash" has a specific model ID, you can use that.
        google_api_key=google_api_key,
        temperature=0.2,
        max_output_tokens=2000, # Note: parameter name is max_output_tokens for this client
        convert_system_message_to_human=True # Often helpful for ReAct agents with Gemini
    )
    print("LLM initialized successfully with ChatGoogleGenerativeAI.")
except Exception as e:
    print(f"Error initializing ChatGoogleGenerativeAI: {e}")
    print("Please ensure you have the 'langchain-google-genai' package installed (`pip install langchain-google-genai`)")
    print("And that your GOOGLE_API_KEY in the .env file is correct and active for the 'gemini-1.5-flash-latest' model (or your specific model).")
    raise

# --- Custom Callback Handler to Track Actions ---
class ActionTracker(BaseCallbackHandler):
    def __init__(self):
        self.actions_taken = []
        self.repeated_actions = 0
        self.tool_call_count = 0
        self.format_errors = 0
        self.last_tool_result = None
        self.consecutive_format_errors = 0
        self.same_action_count = {}
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Track when a tool is called"""
        tool_name = serialized.get("name", "unknown")
        self.tool_call_count += 1
        
        # Track same action repetitions
        action_key = f"{tool_name}:{input_str}"
        self.same_action_count[action_key] = self.same_action_count.get(action_key, 0) + 1
        
        # Warn on repetition - include input in warning
        if self.same_action_count[action_key] > 1:
            self.repeated_actions += 1
            print(f"\n⚠️ WARNING: Repeated action #{self.same_action_count[action_key]}: {tool_name} with input {input_str[:100]}...\n")
        
        # Create action record
        action = {
            "tool": tool_name,
            "input": input_str,
            "timestamp": datetime.now().isoformat()
        }
        
        self.actions_taken.append(action)
        
        # Warn if too many tools used - changed threshold to 4
        if self.tool_call_count > ACTION_TRACKER_TOOL_CALL_WARNING_THRESHOLD:
            print(f"\n⚠️ WARNING: {self.tool_call_count} tools used. Agent may be looping before hitting max iterations.\n")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Store the result of the last tool call"""
        self.last_tool_result = output
        self.consecutive_format_errors = 0  # Reset on successful tool use
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Track format errors"""
        error_str = str(error)
        # Expanded condition for detecting format errors
        if "Invalid Format" in error_str or "Missing" in error_str or "Could not parse LLM output" in error_str:
            self.format_errors += 1
            self.consecutive_format_errors += 1
            print(f"\n⚠️ LLM FORMAT ERROR DETECTED: {error_str}\n")
            if self.consecutive_format_errors > ACTION_TRACKER_MAX_CONSECUTIVE_FORMAT_ERRORS: # MODIFIED
                print("\n⚠️ WARNING: Multiple consecutive format errors. Agent struggling with format.\n")
    
    def reset(self):
        """Reset the tracker for new conversation"""
        self.actions_taken = []
        self.repeated_actions = 0
        self.tool_call_count = 0
        self.format_errors = 0
        self.last_tool_result = None
        self.consecutive_format_errors = 0
        self.same_action_count = {}
    
    def should_stop(self) -> bool:
        """Determine if the agent should stop based on behavior"""
        # Stop if same exact action repeated more than MAX_SAME_ACTION_REPETITIONS times
        if any(count > ACTION_TRACKER_MAX_SAME_ACTION_REPETITIONS for count in self.same_action_count.values()): # MODIFIED
            print(f"\n🛑 STOPPING: Same exact action repeated more than {ACTION_TRACKER_MAX_SAME_ACTION_REPETITIONS} times.")
            return True
        # Stop if too many format errors
        if self.consecutive_format_errors > ACTION_TRACKER_MAX_CONSECUTIVE_FORMAT_ERRORS: # MODIFIED
            print(f"\n🛑 STOPPING: Too many consecutive format errors ({self.consecutive_format_errors}).")
            return True
        # Stop if excessive tool usage
        if self.tool_call_count >= AGENT_MAX_ITERATIONS: # MODIFIED (using >= to catch it before it exceeds if this is called mid-chain)
             print(f"\n🛑 STOPPING (ActionTracker): Excessive tool usage ({self.tool_call_count} calls, limit {AGENT_MAX_ITERATIONS}).")
             return True
        return False

# Initialize action tracker
action_tracker = ActionTracker()

memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True,
    output_key="output" # Ensures memory correctly captures agent's final output
)

# --- Enhanced Task Manager Class ---
class EnhancedTaskManager:
    def __init__(self, filename=TASKS_FILE):
        self.filename = filename
        self.templates_file = TEMPLATES_FILE
        self.analytics_file = ANALYTICS_FILE
        self.tasks = self._load_tasks()
        self.templates = self._load_templates()
        self.analytics = self._load_analytics()
        self._migrate_old_tasks()
        self._remove_duplicate_tasks()
        self.last_action = None  # Track last action to prevent loops
    
    def _load_tasks(self) -> List[Dict]:
        """Loads tasks from a JSON file with enhanced structure."""
        logger.debug(f"[_load_tasks] Attempting to load from {self.filename}. TaskManager ID: {id(self)}")
        try:
            with open(self.filename, 'r') as f:
                tasks_data = json.load(f)
            # Ensure tasks_data is a list
            if not isinstance(tasks_data, list):
                logger.warning(f"Data in {self.filename} is not a list. Re-initializing as empty list.")
                tasks_data = []
            logger.info(f"Loaded {len(tasks_data)} tasks from {self.filename}")
            return tasks_data
        except FileNotFoundError:
            logger.info(f"No existing tasks file found at {self.filename}. Starting fresh.")
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding JSON from {self.filename}. Re-initializing. Error: {e}") # Add error detail
            try:
                with open(self.filename, 'w') as f:
                    json.dump([], f, indent=4)
                # print(f"Initialized {self.filename} with an empty task list.")
            except IOError:
                # print(f"Could not write to {self.filename} to initialize it.")
                pass
            return []
        except Exception as e:
            # print(f"An unexpected error occurred while loading tasks from {self.filename}: {e}. Starting with an empty list.")
            return []
    
    def _load_templates(self) -> Dict[str, Dict]:
        """Loads task templates."""
        try:
            with open(self.templates_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _load_analytics(self) -> Dict:
        """Loads analytics data."""
        try:
            with open(self.analytics_file, 'r') as f:
                return json.load(f)
        except:
            return {
                "completed_tasks": [],
                "productivity_patterns": {},
                "time_tracking": {}
            }
    
    def _migrate_old_tasks(self):
        """Migrates old task format to new enhanced format."""
        for task in self.tasks:
            # Add new fields if they don't exist
            if 'id' not in task:
                task['id'] = str(uuid.uuid4())
            if 'category' not in task:
                task['category'] = "General"
            if 'project' not in task:
                task['project'] = None
            if 'tags' not in task:
                task['tags'] = []
            if 'subtasks' not in task:
                task['subtasks'] = []
            if 'dependencies' not in task:
                task['dependencies'] = []
            if 'recurring' not in task:
                task['recurring'] = None
            if 'time_spent' not in task:
                task['time_spent'] = 0
            if 'created_at' not in task:
                task['created_at'] = datetime.now().isoformat()
            if 'completed_at' not in task:
                task['completed_at'] = None
            if 'completed' not in task:
                task['completed'] = False
    
    def _remove_duplicate_tasks(self):
        """Removes duplicate tasks based on description and deadline."""
        seen = set()
        unique_tasks = []
        
        for task in self.tasks:
            # Create a unique key based on description and deadline
            key = f"{task.get('description', '').lower()}_{task.get('deadline', '')}"
            
            # If it's a recurring task, also include that in the key
            if task.get('recurring'):
                key += f"_{task.get('id', '')[:8]}"  # Use partial ID for recurring tasks
            
            if key not in seen:
                seen.add(key)
                unique_tasks.append(task)
        
        if len(unique_tasks) < len(self.tasks):
            logger.info(f"Removed {len(self.tasks) - len(unique_tasks)} duplicate tasks during initial load.")
            self.tasks = unique_tasks
            # Note: save_tasks() call removed from here
    
    def save_tasks(self) -> None:
        """Saves tasks to a JSON file."""
        logger.debug(f"[save_tasks] Attempting to save {len(self.tasks)} tasks to {self.filename}. TaskManager ID: {id(self)}")
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.tasks, f, indent=4)
            logger.info(f"Successfully saved {len(self.tasks)} tasks to {self.filename}")
        except IOError as e:
            logger.error(f"Could not save tasks to {self.filename}: {e}")
            pass
        except Exception as e:
            # print(f"An unexpected error occurred during save_tasks (e.g., serialization): {e}")
            pass
    
    def save_templates(self) -> None:
        """Saves templates to a JSON file."""
        try:
            with open(self.templates_file, 'w') as f:
                json.dump(self.templates, f, indent=4)
        except IOError:
            # print(f"Error: Could not save templates")
            pass
    
    def save_analytics(self) -> None:
        """Saves analytics data."""
        try:
            with open(self.analytics_file, 'w') as f:
                json.dump(self.analytics, f, indent=4)
        except IOError:
            # print(f"Error: Could not save analytics")
            pass
    
    def _check_duplicate_task(self, description: str, deadline_parsed: str) -> Optional[Dict]:
        """Check if a similar task already exists."""
        for task in self.tasks:
            # Skip completed tasks
            if task.get('completed'):
                continue
                
            # Check for exact description match (case-insensitive)
            if task.get('description', '').lower() == description.lower():
                # Check deadline - must be exact match
                if task.get('deadline') == deadline_parsed:
                    return task
        
        return None
    
    def add_task(self, description: str, deadline: str, importance: str = "Medium", 
                 effort: str = "1 hour", category: str = "General", project: str = None,
                 tags: List[str] = None, recurring: str = None, dependencies: List[str] = None) -> str:
        """Adds a new enhanced task with improved duplicate detection."""
        
        # Parse the deadline first
        parsed_deadline = parse_and_format_date(deadline, allow_past_dates=True)
        if not parsed_deadline:
            # Return structured error for unparseable date
            error_detail = {
                "type": "error",
                "error_type": "UnparseableDateError",
                "message": f"Could not parse deadline '{deadline}'. Please use a valid date format (e.g., YYYY-MM-DD, tomorrow, next Friday).",
                "suggested_user_action": "Please provide the deadline in a recognized date format."
            }
            return json.dumps(error_detail) # Tool returns JSON string

        # NEW CHECK: Prevent adding tasks with past due dates
        today_date = datetime.now().date()
        deadline_date_obj = datetime.strptime(parsed_deadline, '%Y-%m-%d').date()
        if deadline_date_obj < today_date:
            error_detail = {
                "type": "error", 
                "error_type": "PastDateError",
                "message": f"Cannot add task '{description}'. The deadline '{display_date_nicely(parsed_deadline)}' is in the past.",
                "suggested_user_action": "Please provide a future date for the deadline."
            }
            return json.dumps(error_detail)
        # Check for duplicate with exact matching
        existing_task = self._check_duplicate_task(description, parsed_deadline)
        if existing_task:
            existing_category = existing_task.get('category', 'General')
            return (f"⚠️ A similar task already exists: '{existing_task['description']}' "
                   f"in category '{existing_category}' with deadline {display_date_nicely(existing_task['deadline'])}. "
                   f"The new task you tried to add had category '{category}'.")
        
        # Normalize importance
        importance_map = {"high": "High", "h": "High", "medium": "Medium", "m": "Medium", "low": "Low", "l": "Low"}
        importance_normalized = importance_map.get(importance.lower(), "Medium")
        
        # Create the new task
        new_task = {
            "id": str(uuid.uuid4()),
            "description": description,
            "deadline": parsed_deadline,
            "importance": importance_normalized,
            "effort": effort,
            "category": category,
            "project": project,
            "tags": tags or [],
            "subtasks": [],
            "dependencies": dependencies or [],
            "recurring": recurring,
            "completed": False,
            "time_spent": 0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        self.tasks.append(new_task)
        
        # Handle recurring tasks
        if recurring:
            self._create_recurring_instances(new_task)
        
        self.save_tasks()
        return f"✅ Successfully added task: '{description}' with ID {new_task['id'][:8]}... (Deadline: {display_date_nicely(parsed_deadline)})"
    
    def _create_recurring_instances(self, task: Dict, instances: int = 4):
        """Creates recurring instances of a task."""
        if not task['recurring']:
            return
        
        try:
            base_deadline = datetime.strptime(task['deadline'], '%Y-%m-%d')
        except ValueError:
            return
        
        # Handle string recurring values
        recurring_str = task['recurring']
        if isinstance(recurring_str, str):
            recurring_map = {
                "daily": timedelta(days=1),
                "weekly": timedelta(weeks=1),
                "monthly": timedelta(days=30),  # Approximation
                "yearly": timedelta(days=365)
            }
            delta = recurring_map.get(recurring_str.lower())
        else:
            return
        
        if not delta:
            return
        
        for i in range(1, instances + 1):
            new_deadline = base_deadline + (delta * i)
            recurring_task = task.copy()
            recurring_task['id'] = str(uuid.uuid4())
            recurring_task['deadline'] = new_deadline.strftime('%Y-%m-%d')
            recurring_task['description'] = f"{task['description']} (Recurring)"
            recurring_task['created_at'] = datetime.now().isoformat()
            recurring_task['completed'] = False
            recurring_task['completed_at'] = None
            self.tasks.append(recurring_task)
    
    def find_task_by_partial_id(self, partial_id: str) -> Optional[Dict]:
        """Finds a task by partial ID match."""
        for task in self.tasks:
            if task['id'].startswith(partial_id):
                return task
        return None
    
    def _find_task_by_id(self, task_id: str) -> Optional[Dict]:
        """Finds a task by its ID (full or partial)."""
        # Try exact match first
        for task in self.tasks:
            if task['id'] == task_id:
                return task
        # Try partial match
        return self.find_task_by_partial_id(task_id)
    
    def add_subtask(self, parent_task_id: str, subtask_description: str) -> str:
        """Adds a subtask to an existing task."""
        parent_task = self._find_task_by_id(parent_task_id)
        if not parent_task:
            return f"❌ Task with ID {parent_task_id} not found"
        
        subtask = {
            "id": str(uuid.uuid4()),
            "description": subtask_description,
            "completed": False
        }
        
        parent_task['subtasks'].append(subtask)
        self.save_tasks()
        return f"✅ Added subtask '{subtask_description}' to task '{parent_task['description']}'"
    
    def search_tasks(self, query: str) -> List[Dict]:
        """Searches tasks by various criteria."""
        # FIX: Strip whitespace from the query and convert to lowercase
        query = query.strip().lower()
        results = []
        
        for task in self.tasks:
            # Search in description
            if query in task['description'].lower():
                results.append(task)
                continue
            
            # Search in category
            if query in task.get('category', '').lower():
                results.append(task)
                continue
            
            # Search in project
            if task.get('project') and query in task['project'].lower():
                results.append(task)
                continue
            
            # Search in tags
            if any(query in tag.lower() for tag in task.get('tags', [])):
                results.append(task)
                continue
        
        return results
    
    def filter_tasks_advanced(self, filters: Dict[str, Any]) -> List[Dict]:
        """Advanced filtering with multiple criteria."""
        filtered = list(self.tasks)  # Changed from self.tasks.copy()
        
        # Filter by category
        if filters.get('category'):
            filtered = [t for t in filtered if t.get('category', '').lower() == filters['category'].lower()]
        
        # Filter by importance
        if filters.get('importance'):
            filtered = [t for t in filtered if t.get('importance', '').lower() == filters['importance'].lower()]
        
        # Filter by project
        if filters.get('project'):
            filtered = [t for t in filtered if t.get('project', '').lower() == filters['project'].lower()]
        
        # Filter by tags
        if filters.get('tags'):
            tags_to_match = [tag.lower() for tag in filters['tags']]
            filtered = [t for t in filtered if any(tag.lower() in tags_to_match for tag in t.get('tags', []))]
        
        # Filter by time estimate
        if filters.get('max_effort'):
            max_minutes = self._parse_effort_to_minutes(filters['max_effort'])
            filtered = [t for t in filtered if self._parse_effort_to_minutes(t.get('effort', '0')) <= max_minutes]
        
        # Filter by completion status
        if 'completed' in filters:
            filtered = [t for t in filtered if t.get('completed') == filters['completed']]
        
        # Filter by deadline range
        if filters.get('deadline_start') or filters.get('deadline_end'):
            filtered = self._filter_by_deadline_range(filtered, filters.get('deadline_start'), filters.get('deadline_end'))
        
        return filtered
    
    def _filter_by_deadline_range(self, tasks: List[Dict], start_date: str = None, end_date: str = None) -> List[Dict]:
        """Filters tasks by deadline range."""
        filtered = []
        
        for task in tasks:
            if not task.get('deadline'):
                continue
                
            try:
                task_deadline = datetime.strptime(task['deadline'], '%Y-%m-%d').date()
                
                if start_date:
                    start = datetime.strptime(parse_and_format_date(start_date), '%Y-%m-%d').date()
                    if task_deadline < start:
                        continue
                
                if end_date:
                    end = datetime.strptime(parse_and_format_date(end_date), '%Y-%m-%d').date()
                    if task_deadline > end:
                        continue
                
                filtered.append(task)
            except ValueError:
                continue
        
        return filtered
    
    def _parse_effort_to_minutes(self, effort_str: str) -> int:
        """Converts effort string to minutes."""
        if not effort_str:
            return 0
        
        effort_str = effort_str.lower()
        total_minutes = 0
        
        # Parse hours
        hours_match = re.search(r'(\d+)\s*(?:hours?|hrs?)', effort_str)
        if hours_match:
            total_minutes += int(hours_match.group(1)) * 60
        
        # Parse minutes
        mins_match = re.search(r'(\d+)\s*(?:minutes?|mins?)', effort_str)
        if mins_match:
            total_minutes += int(mins_match.group(1))
        
        # Parse days
        days_match = re.search(r'(\d+)\s*days?', effort_str)
        if days_match:
            total_minutes += int(days_match.group(1)) * 8 * 60  # Assume 8-hour workday
        
        # If no pattern matched, try to parse as just a number (assume hours)
        if total_minutes == 0:
            try:
                num_val = float(re.findall(r"[\d\.]+", effort_str)[0]) # Try to extract number
                if "min" in effort_str:
                     total_minutes = int(num_val)
                else: # Assume hours if no unit
                     total_minutes = int(num_val * 60)
            except:
                pass # Could not parse as number
        
        return total_minutes
    
    def track_time(self, task_id: str, minutes: int) -> str:
        """Tracks time spent on a task."""
        task = self._find_task_by_id(task_id)
        if not task:
            return f"❌ Task with ID {task_id} not found"
        
        task['time_spent'] = task.get('time_spent', 0) + minutes
        
        # Update analytics
        today = datetime.now().date().isoformat()
        if today not in self.analytics['time_tracking']:
            self.analytics['time_tracking'][today] = {}
        
        if task_id not in self.analytics['time_tracking'][today]:
            self.analytics['time_tracking'][today][task_id] = 0
        
        self.analytics['time_tracking'][today][task_id] += minutes
        
        self.save_tasks()
        self.save_analytics()
        
        return f"✅ Added {minutes} minutes to task '{task['description']}'. Total time: {task['time_spent']} minutes"
    
    def get_analytics(self, period: str = "week") -> Dict[str, Any]:
        """Gets productivity analytics for a given period."""
        now = datetime.now()
        
        if period == "day":
            start_date = now.date()
        elif period == "week":
            start_date = now.date() - timedelta(days=now.weekday())
        elif period == "month":
            start_date = now.date() - timedelta(days=30) # Approximation, consider calendar month for more accuracy if needed
        else: # Default to year or handle other specific periods
            start_date = now.date() - timedelta(days=365) # Approximation
        
        # Calculate completed tasks
        completed_in_period = []
        for task in self.tasks:
            if task.get('completed') and task.get('completed_at'):
                try:
                    completed_date = datetime.fromisoformat(task['completed_at']).date()
                    if completed_date >= start_date:
                        completed_in_period.append(task)
                except ValueError: # Handle cases where completed_at might not be a valid ISO format
                    logger.warning(f"Task {task.get('id')} has invalid completed_at format: {task.get('completed_at')}")
                    continue
        
        # Calculate productivity by category
        category_stats = defaultdict(int)
        for task in completed_in_period:
            category_stats[task.get('category', 'General')] += 1
        
        # Calculate average completion time
        completion_times = []
        for task in completed_in_period:
            if task.get('created_at') and task.get('completed_at'):
                try:
                    created = datetime.fromisoformat(task['created_at'])
                    completed = datetime.fromisoformat(task['completed_at'])
                    completion_times.append((completed - created).total_seconds() / 3600)  # in hours
                except ValueError:
                    logger.warning(f"Task {task.get('id')} has invalid created_at or completed_at format for time calculation.")
                    continue
        
        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
        
        # --- START OF UNCOMMENTED AND MODIFIED SECTION ---
        time_by_category = defaultdict(int)
        # Ensure self.analytics and 'time_tracking' exist
        time_tracking_data = self.analytics.get('time_tracking', {})
        if isinstance(time_tracking_data, dict): # Check if it's a dictionary
            for date_str, tasks_time in time_tracking_data.items():
                try:
                    # Ensure date_str is valid before parsing
                    if not date_str: continue
                    date_obj = date_parser.parse(date_str).date() # Use dateutil.parser for more robust date parsing
                    if date_obj >= start_date:
                        if isinstance(tasks_time, dict): # Ensure tasks_time is a dictionary
                            for task_id, minutes in tasks_time.items():
                                task = self._find_task_by_id(task_id) # This finds the task from self.tasks
                                if task:
                                    time_by_category[task.get('category', 'General')] += minutes
                        else:
                            logger.warning(f"Expected dict for tasks_time on {date_str}, got {type(tasks_time)}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse date string '{date_str}' in analytics time_tracking: {e}")
                    continue
        else:
            logger.warning(f"Expected dict for self.analytics['time_tracking'], got {type(time_tracking_data)}")
        # --- END OF UNCOMMENTED AND MODIFIED SECTION ---
        
        # Count overdue tasks
        overdue_count = 0
        for task in self.tasks:
            if not task.get('completed') and task.get('deadline'):
                try:
                    deadline_date = datetime.strptime(task['deadline'], '%Y-%m-%d').date()
                    if deadline_date < now.date():
                        overdue_count += 1
                except ValueError: # Handle invalid deadline format
                    logger.warning(f"Task {task.get('id')} has invalid deadline format: {task.get('deadline')}")
                    pass
        
        return {
            "period": period,
            "tasks_completed": len(completed_in_period),
            "completion_by_category": dict(category_stats),
            "average_completion_hours": round(avg_completion_time, 2),
            # --- UNCOMMENTED AND VERIFIED LINE ---
            "time_spent_by_category": {k: f"{v//60}h {v%60}m" for k, v in time_by_category.items()},
            "overdue_tasks": overdue_count
        }
    
    def create_template(self, name: str, template_data: Dict) -> str:
        """Creates a reusable task template."""
        self.templates[name] = template_data
        self.save_templates()
        return f"✅ Created template '{name}'"
    
    def use_template(self, template_name: str, customizations: Dict = None) -> str:
        """Creates a task from a template."""
        if template_name not in self.templates:
            return f"❌ Template '{template_name}' not found"
        
        template = self.templates[template_name].copy()
        
        # Apply customizations
        if customizations:
            template.update(customizations)
        
        # Ensure we have required fields
        if 'description' not in template:
            return "❌ Template must have a description"
        if 'deadline' not in template:
            template['deadline'] = "tomorrow"
        
        return self.add_task(**template)
    
    def _calculate_task_priority(self, task: Dict, consider_time_of_day_for_effort: bool = False) -> int:
        """
        Calculates the priority score for a task.
        Includes deadline urgency, importance, and conditional effort scoring.
        """
        score = 0
        now = datetime.now()
        current_hour = now.hour # Needed for time-of-day effort consideration

        # Deadline urgency
        try:
            if task.get('deadline'):
                deadline_dt = datetime.strptime(task['deadline'], '%Y-%m-%d').date()
                days_until = (deadline_dt - now.date()).days
                if days_until < 0:
                    score += 100  # Overdue
                elif days_until == 0:
                    score += 50   # Due today
                elif days_until <= 2:
                    score += 30   # Due soon
                else:
                    # Diminishing returns for far-off deadlines, but still positive if not too far
                    score += max(0, 20 - days_until) 
        except (ValueError, TypeError):
            pass # No deadline or invalid format

        # Importance
        importance_scores = {"High": 30, "Medium": 15, "Low": 5}
        score += importance_scores.get(task.get('importance', 'Medium'), 10) # Default to 10 if importance is weird

        # Conditional Effort Scoring
        if consider_time_of_day_for_effort:
            effort_minutes = self._parse_effort_to_minutes(task.get('effort', '1 hour'))
            if current_hour >= 16:  # Late afternoon, prefer shorter tasks
                # Score inversely proportional to effort (higher score for less effort)
                # Max boost of 30 for 0-minute tasks, diminishing.
                score += max(0, 30 - (effort_minutes / 10)) 
            # You could add other effort-based scoring here if needed when consider_time_of_day_for_effort is True
        else:
            # Optional: Add a static or different effort score if consider_time_of_day_for_effort is False
            # For now, we'll match the original _calculate_priority_score_for_agent which had minimal/no effort scoring.
            # effort_minutes = self._parse_effort_to_minutes(task.get('effort', '1 hour'))
            # if effort_minutes <= 30: score += 5 # Example: Slight boost for quick wins always
            pass

        return score
    
    def get_smart_suggestions(self) -> Dict[str, Any]:
        now = datetime.now()
        suggestions = {
            "next_task": None,
            "optimal_time": None,
            "insights": []
        }

        all_incomplete_tasks = [t for t in self.tasks if not t.get('completed')]

        if not all_incomplete_tasks:
            suggestions["insights"].append("All tasks completed! Time to add new ones.")
            return suggestions

        # Filter out overdue tasks for "next_task" consideration
        # These are tasks that are not completed AND not overdue.
        candidate_next_tasks = [t for t in all_incomplete_tasks if not self._is_overdue(t)]

        if candidate_next_tasks:
            # Sort candidate (non-overdue) tasks by priority
            candidate_next_tasks.sort(
                key=lambda task: self._calculate_task_priority(task, consider_time_of_day_for_effort=True),
                reverse=True
            )
            suggestions["next_task"] = candidate_next_tasks[0]

            # Suggest optimal time based on effort for the chosen next_task
            effort_minutes = self._parse_effort_to_minutes(suggestions["next_task"].get('effort', '1 hour'))
            current_hour_for_optimal_time = now.hour
            if effort_minutes <= 30:
                suggestions["optimal_time"] = "Now (short task, easy win)"
            elif effort_minutes <= 120 and current_hour_for_optimal_time < 12: # up to 2 hours
                suggestions["optimal_time"] = "This morning (while energy is high)"
            elif effort_minutes > 120: # more than 2 hours
                suggestions["optimal_time"] = "Block out dedicated time (e.g., tomorrow morning)"
            else: # Default for tasks between 30 mins and 2 hours in the afternoon
                suggestions["optimal_time"] = "This afternoon"
        else:
            # No non-overdue tasks available for suggestion as "next_task"
            suggestions["insights"].append("No upcoming (non-overdue) tasks available to suggest as 'next'.")

        # --- Generate general insights based on ALL incomplete tasks ---
        overdue_tasks_list = [t for t in all_incomplete_tasks if self._is_overdue(t)]
        if overdue_tasks_list:
            suggestions["insights"].append(f"You have {len(overdue_tasks_list)} overdue task(s). Consider addressing these separately.")

        # High priority count among ALL incomplete tasks (including overdue ones for this general insight)
        high_priority_incomplete_tasks = [t for t in all_incomplete_tasks if t.get('importance') == 'High']
        if len(high_priority_incomplete_tasks) > 3:
            suggestions["insights"].append(f"You have {len(high_priority_incomplete_tasks)} high-priority tasks (including any overdue). Consider delegating or rescheduling some.")
        # If no "next" non-overdue task was suggested, but there are high-priority tasks (which might be overdue), mention them.
        elif not suggestions["next_task"] and high_priority_incomplete_tasks:
            suggestions["insights"].append(f"Note: You have {len(high_priority_incomplete_tasks)} high-priority task(s) remaining (these might be overdue).")


        # Category distribution of ALL incomplete tasks
        category_counts = defaultdict(int)
        for task in all_incomplete_tasks:
            category_counts[task.get('category', 'General')] += 1
        
        if len(category_counts) > 1: # Only show if there's more than one category to compare
            if category_counts: # Check if category_counts is not empty before calling max
                dominant_category = max(category_counts.items(), key=lambda x: x[1])
                suggestions["insights"].append(f"Most of your incomplete tasks are in the '{dominant_category[0]}' category ({dominant_category[1]} tasks).")
        elif not all_incomplete_tasks and not suggestions["insights"]: # If no tasks at all and no other insights
             suggestions["insights"].append("No tasks to analyze for category distribution.")


        # If after all this, there's no next_task and no insights, add a generic message.
        if not suggestions["next_task"] and not suggestions["insights"]:
             suggestions["insights"].append("No specific suggestions or insights at this time. All caught up or add more tasks!")
        
        return suggestions
    
    def _is_overdue(self, task: Dict) -> bool:
        """Checks if a task is overdue."""
        if not task.get('deadline'):
            return False
        try:
            deadline_date = datetime.strptime(task['deadline'], '%Y-%m-%d').date()
            return deadline_date < datetime.now().date()
        except:
            return False
    
    def export_tasks(self, format: str = "json", filename: str = None) -> str:
        """Exports tasks in various formats."""
        if format == "json":
            export_data = {
                "tasks": self.tasks,
                "exported_at": datetime.now().isoformat()
            }
            content = json.dumps(export_data, indent=2)
            extension = "json"
        
        elif format == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'description', 'deadline', 'importance', 'effort', 'category', 
                'project', 'tags', 'completed', 'created_at'
            ])
            writer.writeheader()
            
            for task in self.tasks:
                row = {
                    'description': task['description'],
                    'deadline': task['deadline'],
                    'importance': task['importance'],
                    'effort': task['effort'],
                    'category': task.get('category', ''),
                    'project': task.get('project', ''),
                    'tags': ','.join(task.get('tags', [])),
                    'completed': task.get('completed', False),
                    'created_at': task.get('created_at', '')
                }
                writer.writerow(row)
            
            content = output.getvalue()
            extension = "csv"
        
        elif format == "markdown":
            content = "# Task List\n\n"
            
            # Group by category
            tasks_by_category = defaultdict(list)
            for task in self.tasks:
                tasks_by_category[task.get('category', 'General')].append(task)
            
            for category, tasks in tasks_by_category.items():
                content += f"## {category}\n\n"
                for task in tasks:
                    status = "x" if task.get('completed') else " "
                    content += f"- [{status}] **{task['description']}**\n"
                    content += f"  - Deadline: {display_date_nicely(task['deadline'])}\n"
                    content += f"  - Importance: {task['importance']}\n"
                    content += f"  - Effort: {task['effort']}\n"
                    if task.get('tags'):
                        content += f"  - Tags: {', '.join(task['tags'])}\n"
                    content += "\n"
            
            extension = "md"
        
        else:
            return f"❌ Unsupported format: {format}"
        
        # Save to file
        if not filename:
            filename = f"tasks_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"✅ Tasks exported to {filename}"
        except Exception as e:
            return f"❌ Error exporting tasks: {str(e)}"
    
    def import_tasks(self, filename: str, format: str = "json") -> str:
        """Imports tasks from a file."""
        try:
            if format == "json":
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and 'tasks' in data:
                    imported_tasks = data['tasks']
                elif isinstance(data, list):
                    imported_tasks = data
                else:
                    return "❌ Invalid JSON format for tasks"
                
                # Better validation
                if not isinstance(imported_tasks, list):
                    return "❌ Invalid JSON format for tasks"
                
                # Add imported tasks
                count = 0
                for task in imported_tasks:
                    # Ensure task has required fields
                    if 'id' not in task:
                        task['id'] = str(uuid.uuid4())
                    
                    # Check if ID already exists
                    if not any(t['id'] == task['id'] for t in self.tasks):
                        self.tasks.append(task)
                        count += 1
                
                self.save_tasks()
                return f"✅ Imported {count} new tasks from {filename}."
            
            else:
                return f"❌ Unsupported import format: {format}"
        
        except FileNotFoundError:
            return f"❌ Error: File '{filename}' not found"
        except Exception as e:
            return f"❌ Error importing tasks: {str(e)}"
    
    def find_tasks_by_name(self, name: str, exact_match: bool = False) -> List[Dict]:
        name_lower = name.lower()
        results = []
        for task in self.tasks:
            if not task.get('completed'): # Usually, we only want to delete active tasks by name
                if exact_match:
                    if task['description'].lower() == name_lower:
                        results.append(task)
                else: # Partial match
                    if name_lower in task['description'].lower():
                        results.append(task)
        return results
    

    def add_multiple_tasks(self, tasks_data: List[Dict]) -> Dict[str, Any]:
        """
        Adds multiple tasks to the task list from a list of task data dictionaries.
        Performs validation for each task (e.g., no past dates, duplicate checks).

        Args:
            tasks_data: A list of dictionaries, where each dictionary contains
                        task details like 'description', 'deadline', and optional
                        'importance', 'category', 'effort'.

        Returns:
            A dictionary summarizing the results, e.g.,
            {"succeeded": count, "failed": count, "messages": list_of_status_messages}
        """
        succeeded_count = 0
        failed_count = 0
        status_messages = []
        added_task_ids_this_batch = [] # To prevent intra-batch duplicates based on description/deadline

        for i, task_info in enumerate(tasks_data, 1):
            description = task_info.get("description")
            deadline = task_info.get("deadline")

            if not description or not deadline:
                status_messages.append(f"Task #{i} skipped: Missing description or deadline.")
                failed_count += 1
                continue

            # --- Date Validation (Prevent Past Dates) ---
            parsed_deadline = parse_and_format_date(deadline, allow_past_dates=True) # Parse first
            if not parsed_deadline:
                status_messages.append(f"Task '{description}': Could not parse deadline '{deadline}'. Skipped.")
                failed_count += 1
                continue

            today_date = datetime.now().date()
            try:
                deadline_date_obj = datetime.strptime(parsed_deadline, '%Y-%m-%d').date()
                if deadline_date_obj < today_date:
                    status_messages.append(f"Task '{description}': Deadline '{display_date_nicely(parsed_deadline)}' is in the past. Skipped.")
                    failed_count += 1
                    continue
            except ValueError: # Should be caught by parse_and_format_date, but as a safeguard
                status_messages.append(f"Task '{description}': Invalid deadline format '{parsed_deadline}'. Skipped.")
                failed_count += 1
                continue
            
            # --- Duplicate Check (within existing tasks and this batch) ---
            # Check against already existing tasks
            existing_task_check = self._check_duplicate_task(description, parsed_deadline)
            if existing_task_check:
                status_messages.append(f"Task '{description}': Similar task already exists. Skipped.")
                failed_count += 1
                continue
            
            # Check against tasks already successfully added in *this current batch*
            # to prevent adding "Task X by Tomorrow" twice in the same bulk request.
            is_intra_batch_duplicate = False
            for added_desc, added_deadline_str in added_task_ids_this_batch:
                if description.lower() == added_desc.lower() and parsed_deadline == added_deadline_str:
                    is_intra_batch_duplicate = True
                    break
            if is_intra_batch_duplicate:
                status_messages.append(f"Task '{description}': Duplicate within this bulk request. Skipped.")
                failed_count += 1
                continue

            # --- Prepare and Add Task ---
            importance = task_info.get("importance", "Medium")
            effort = task_info.get("effort", "1 hour")
            category = task_info.get("category", "General")
            # Add other fields as needed (project, tags, etc.)

            importance_map = {"high": "High", "h": "High", "medium": "Medium", "m": "Medium", "low": "Low", "l": "Low"}
            importance_normalized = importance_map.get(str(importance).lower(), "Medium")

            new_task = {
                "id": str(uuid.uuid4()),
                "description": description,
                "deadline": parsed_deadline,
                "importance": importance_normalized,
                "effort": str(effort),
                "category": str(category),
                "project": task_info.get("project"),
                "tags": task_info.get("tags", []),
                "subtasks": [],
                "dependencies": [],
                "recurring": task_info.get("recurring"),
                "completed": False,
                "time_spent": 0,
                "created_at": datetime.now().isoformat(),
                "completed_at": None
            }
            self.tasks.append(new_task)
            added_task_ids_this_batch.append((description, parsed_deadline)) # Track for intra-batch duplicate check
            status_messages.append(f"Task '{description}' (Deadline: {display_date_nicely(parsed_deadline)}): Successfully added.")
            succeeded_count += 1

        if succeeded_count > 0 or failed_count > 0: # Only save if there was an attempt
            self.save_tasks()
            
        return {
            "succeeded": succeeded_count,
            "failed": failed_count,
            "messages": status_messages
        }
    
    

# --- Helper functions ---
def parse_and_format_date(date_string: str, allow_past_dates: bool = False) -> str | None:
    """
    Revised date parsing, prioritizing dateparser with specific settings.
    It attempts to parse a date string into 'YYYY-MM-DD' format.

    Args:
        date_string: The string representation of the date.
        allow_past_dates: If True, dates parsed as being in the past are accepted.
                          If False:
                            - Ambiguous dates (no year) resolving to past this year are moved to next year.
                            - Explicitly past-dated years are still parsed as such; the caller
                              must decide if this is acceptable.
                          This flag also influences dateparser's 'PREFER_DATES_FROM' setting.

    Returns:
        A string in 'YYYY-MM-DD' format if parsing is successful, otherwise None.
    """
    if not date_string:
        return None

    original_date_string = str(date_string) # Ensure it's a string
    normalized_date_string = original_date_string.strip().lower()
    today = datetime.now().date()

    # Handle very common relative terms directly for speed and reliability
    if normalized_date_string == "today" or normalized_date_string == "now":
        return today.strftime('%Y-%m-%d')
    if normalized_date_string == "tomorrow":
        return (today + timedelta(days=1)).strftime('%Y-%m-%d')
    if normalized_date_string == "yesterday":
        # 'yesterday' is only returned if allow_past_dates is true,
        # otherwise, it defaults to 'today' as per previous logic.
        # This could also return None or raise an error if past dates are strictly forbidden.
        return (today - timedelta(days=1)).strftime('%Y-%m-%d') if allow_past_dates else today.strftime('%Y-%m-%d')

    final_parsed_date = None

    # Heuristic to check if the original input string likely contained a full year.
    # This helps decide how to treat dates that parse to the past.
    original_had_year = bool(re.search(r'\b(19\d{2}|20\d{2})\b', original_date_string))

    # Try explicit common formats before dateparser if they are unambiguous
    # DD/MM/YYYY or DD.MM.YYYY
    if re.match(r'^\d{1,2}[/.-]\d{1,2}[/.-]\d{4}$', normalized_date_string): # Added hyphen
        # Normalize separators to / for strptime
        temp_date_str = re.sub(r'[.-]', '/', normalized_date_string)
        try:
            dt = datetime.strptime(temp_date_str, '%d/%m/%Y')
            final_parsed_date = dt.date()
        except ValueError:
            # If DD/MM/YYYY fails, try MM/DD/YYYY as a common ambiguity
            try:
                dt = datetime.strptime(temp_date_str, '%m/%d/%Y')
                # Heuristic: if parsed day > 12, it was likely DD/MM/YYYY format intended
                if dt.day <= 12: # Could be MM/DD
                    final_parsed_date = dt.date()
                # else: pass, DD/MM/YYYY was likely intended but failed.
            except ValueError:
                pass # Both failed
    # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
    elif re.match(r'^\d{4}[/.-]\d{1,2}[/.-]\d{1,2}$', normalized_date_string):
        temp_date_str = re.sub(r'[./]', '-', normalized_date_string) # Normalize to -
        try:
            final_parsed_date = datetime.strptime(temp_date_str, '%Y-%m-%d').date()
        except ValueError:
            pass

    # Handle day names with more direct logic (e.g., "next monday", "friday")
    if final_parsed_date is None:
        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        current_weekday = today.weekday()

        for i, day_name in enumerate(day_names):
            if f"next {day_name}" == normalized_date_string:
                days_ahead = i - current_weekday
                if days_ahead <= 0:  # Target day already happened this week or is today
                    days_ahead += 7
                final_parsed_date = today + timedelta(days=days_ahead)
                break
            elif day_name == normalized_date_string: # e.g., "monday"
                days_ahead = i - current_weekday
                # If it's for a day that has already passed this week, assume next week's day.
                # If allow_past_dates is True, this might need adjustment if user means *last* monday.
                # For now, "monday" means current or next monday.
                if days_ahead < 0 :
                    days_ahead += 7
                final_parsed_date = today + timedelta(days=days_ahead)
                break

    # If specific formats or day names didn't match, use dateparser
    if final_parsed_date is None:
        settings = {
            'PREFER_DATES_FROM': 'future' if not allow_past_dates else 'current_period', # 'current_period' is often better than 'past'
            'DATE_ORDER': 'DMY',  # Day, Month, Year preference for ambiguous dates like 01/02/03
            'STRICT_PARSING': False, # Allow more flexibility
            'RELATIVE_BASE': datetime.combine(today, datetime.min.time()) # Use today as base for relative dates
        }
        try:
            # Use original_date_string as case might matter (e.g., "May" vs "may")
            parsed_obj = dateparser.parse(original_date_string, settings=settings)
            if parsed_obj:
                final_parsed_date = parsed_obj.date()
            else: # Fallback to dateutil.parser if dateparser fails
                try:
                    # date_parser.parse is powerful; dayfirst=True helps with DMY preference
                    final_parsed_date = date_parser.parse(original_date_string, dayfirst=True).date()
                except (ValueError, TypeError, OverflowError):
                    logger.warning(f"Could not parse date string: '{original_date_string}' with any method.")
                    return None
        except Exception: # Catch any other dateparser/dateutil issues
            return None


    if final_parsed_date is None:
        # print(f"Warning: Date parsing resulted in None for: '{original_date_string}'")
        return None

    # Date Adjustment Logic:
    # This section handles how to treat dates that parse to the past,
    # based on `allow_past_dates` and whether a year was explicitly provided.
    if final_parsed_date < today:
        if not allow_past_dates:
            # If past dates are not allowed AND the user did NOT provide an explicit year
            if not original_had_year:
                # Assume the user meant the next occurrence of this month/day.
                try:
                    final_parsed_date = final_parsed_date.replace(year=final_parsed_date.year + 1)
                except ValueError:  # Handles Feb 29 for non-leap year if next year isn't leap
                    if final_parsed_date.month == 2 and final_parsed_date.day == 29:
                        final_parsed_date = datetime(final_parsed_date.year + 1, 3, 1).date()
            # else (original_had_year is True):
            # An explicit past year was given, but allow_past_dates is False.
            # The function's job is to parse. It has parsed an explicit past date.
            # The *caller* should decide if this is an error or acceptable.
            # For `add_task`, it will call this with `allow_past_dates=True`, so this branch isn't hit.
            # If another part of the system calls with `allow_past_dates=False` and gets an explicit past date,
            # it's up to that caller to handle it. We return the parsed date.
            pass
        # else (allow_past_dates is True):
        # Past dates are allowed, so no adjustment needed if it's in the past.
        pass

    return final_parsed_date.strftime('%Y-%m-%d')

def display_date_nicely(iso_date_string):
    """Formats an ISO date string into a more readable format."""
    if not iso_date_string:
        return "N/A"
    try:
        dt_obj = datetime.strptime(iso_date_string, '%Y-%m-%d')
        return dt_obj.strftime('%a, %b %d, %Y')
    except ValueError:
        return iso_date_string

# Initialize TaskManager
task_manager = EnhancedTaskManager()


@tool
def add_task(task_input_str: str) -> str:
    """
    ADDS a new task to the task list. Use this ONLY when user wants to CREATE a new task.
    Input: A JSON string with fields: "description", "deadline".
    Optional fields: "importance", "effort", "category", "project", "tags", "recurring".
    Example: {"description": "Review report", "deadline": "Friday", "importance": "High", "category": "Work"}
    The LLM calling this tool is responsible for formatting the user's request into this JSON structure.
    Returns: Success message with task ID or a JSON error object if input is invalid or task creation fails.
    """
    logger.debug(f"[add_task_tool] Received input: {task_input_str}")
    try:
        # Check for repeated calls with same input (existing logic is good)
        current_exec_id = getattr(task_manager, 'current_execution_id', None)
        last_exec_id_for_add = getattr(task_manager, 'last_add_execution_id', None)
        if hasattr(task_manager, 'last_add_task_input') and \
           task_manager.last_add_task_input == task_input_str and \
           current_exec_id == last_exec_id_for_add and \
           action_tracker.tool_call_count > 1:
            # This message should ideally be a JSON object too for consistency,
            # but a simple string is also handled by the agent's Final Answer logic.
            # For now, keeping it simple.
            return "✅ Task already processed in this interaction. No need to add again."

        task_manager.last_add_task_input = task_input_str
        task_manager.last_add_execution_id = current_exec_id

        try:
            params = json.loads(task_input_str)
            if not isinstance(params, dict):
                # This case should ideally not happen if LLM follows prompt, but good to have.
                raise json.JSONDecodeError("Input must be a JSON object.", task_input_str, 0)
        except json.JSONDecodeError as e:
            logger.warning(f"[add_task_tool] Invalid JSON input: {task_input_str}. Error: {e}")
            # Try to parse simple "description by/on/due deadline" as a last resort
            # This is a concession to potential LLM mistakes, but the primary path is JSON.
            match = re.match(r'^(.*?)\s+(by|on|due)\s+(.+)$', task_input_str, re.IGNORECASE)
            if match:
                description_val = match.group(1).strip()
                deadline_val = match.group(3).strip()
                # Check if deadline_val itself contains further details like importance
                # e.g., "Submit report by Friday, importance High"
                deadline_parts = re.match(r'^(.*?)(?:,\s*importance[:\s]+(\w+))?(?:,\s*category[:\s]+(\w+))?(?:,\s*effort[:\s]+([\w\s]+))?$', deadline_val, re.IGNORECASE)
                if deadline_parts:
                    params = {"description": description_val, "deadline": deadline_parts.group(1).strip()}
                    if deadline_parts.group(2): params["importance"] = deadline_parts.group(2).strip()
                    if deadline_parts.group(3): params["category"] = deadline_parts.group(3).strip()
                    if deadline_parts.group(4): params["effort"] = deadline_parts.group(4).strip()
                else:
                    params = {"description": description_val, "deadline": deadline_val}

                logger.info(f"add_task tool: Fallback - Parsed simple string '{task_input_str}' to: {params}")
            else:
                return json.dumps({
                    "type": "error",
                    "error_type": "InvalidInputFormat",
                    "message": f"Input for add_task was not a valid JSON string. Original error: {e}. Input: '{task_input_str[:100]}...'",
                    "suggested_user_action": "Please provide task details in JSON format. For example: {\"description\": \"My Task\", \"deadline\": \"tomorrow\"}"
                })

        # Ensure description and deadline are present after parsing
        description = params.get("description")
        deadline = params.get("deadline")

        if not description or not deadline:
            return json.dumps({
                "type": "error",
                "error_type": "MissingArgumentError",
                "message": "Task 'description' and 'deadline' are required but one or both were missing or empty in the parsed input.",
                "suggested_user_action": "Please ensure your input includes non-empty 'description' and 'deadline' fields."
            })

        # Convert recurring dict to string if needed (this is fine)
        if 'recurring' in params and isinstance(params['recurring'], dict):
            params['recurring'] = params['recurring'].get('frequency')

        # Call the task manager method
        # task_manager.add_task itself returns JSON errors for past dates, unparseable dates etc.
        result = task_manager.add_task(
            description=params.get("description"),
            deadline=params.get("deadline"),
            importance=params.get("importance", "Medium"),
            effort=params.get("effort", "1 hour"),
            category=params.get("category", "General"),
            project=params.get("project"),
            tags=params.get("tags"), # Will default to None if not present, handled by task_manager
            recurring=params.get("recurring"),
            dependencies=params.get("dependencies")
        )
        return result

    except Exception as e:
        logger.error(f"[add_task_tool] UNEXPECTED ERROR: {str(e)} for input '{task_input_str}'", exc_info=True)
        return json.dumps({
            "type": "error",
            "error_type": "ToolExecutionError",
            "message": f"Unexpected error in add_task tool: {str(e)}",
            "suggested_user_action": "An unexpected error occurred while trying to add the task. Please try rephrasing or check your input."
        })

@tool
def show_tasks(filter_string: str = "all") -> str:
    """
    SHOWS tasks with optional filtering. Use this ONLY when user wants to SEE their tasks.
    Input: Optional filter string. Examples: "all", "today", "this_week", "overdue",
           "completed", "incomplete", "high priority", "medium priority", "low priority",
           or a JSON filter string for advanced filtering (e.g., {"category": "Work", "completed": false}).
           For combined filters with relative dates, use "filter_due_today": true or "filter_due_this_week": true.
           Example: {"category": "Work", "filter_due_this_week": true}.
    Returns: JSON string with 'text_summary' (concise) and 'structured_tasks' list, or 'No tasks found' message.
    """
    filter_string = filter_string.strip()
    logger.debug(f"[show_tasks_tool] Received filter: {filter_string}")
    try:
        current_exec_id = getattr(task_manager, 'current_execution_id', None)
        last_exec_id_for_show = getattr(task_manager, 'last_show_execution_id', None)
        if hasattr(task_manager, 'last_show_filter') and \
           task_manager.last_show_filter == filter_string and \
           current_exec_id == last_exec_id_for_show and \
           action_tracker.tool_call_count > 1:
             return json.dumps({
                 "type": "cached_task_list",
                 "text_summary": "✅ Tasks already shown above for this filter in the current interaction.",
                 "structured_tasks": [],
                 "filter_used": filter_string
             })

        task_manager.last_show_filter = filter_string
        task_manager.last_show_execution_id = current_exec_id

        tasks_to_display = []
        is_json_like = filter_string.startswith('{') and filter_string.endswith('}')
        filter_lower = filter_string.lower()

        if is_json_like:
            try:
                filters_from_json = json.loads(filter_string)
                processed_filters = {} # Filters to be passed to task_manager.filter_tasks_advanced

                # --- START MODIFIED SECTION FOR BOOLEAN DATE FLAGS ---
                now = datetime.now()
                process_date_flags_implicitly = True

                # Check for explicit deadline_start/end from the original JSON input
                if "deadline_start" in filters_from_json and "deadline_end" in filters_from_json:
                    process_date_flags_implicitly = False
                    processed_filters["deadline_start"] = filters_from_json.pop("deadline_start")
                    processed_filters["deadline_end"] = filters_from_json.pop("deadline_end")

                if process_date_flags_implicitly:
                    if filters_from_json.pop("filter_due_today", False):
                        today_str = now.strftime('%Y-%m-%d')
                        processed_filters["deadline_start"] = today_str
                        processed_filters["deadline_end"] = today_str
                    elif filters_from_json.pop("filter_due_this_week", False):
                        start_of_week = now.date() - timedelta(days=now.weekday())
                        end_of_week = start_of_week + timedelta(days=6)
                        processed_filters["deadline_start"] = start_of_week.strftime('%Y-%m-%d')
                        processed_filters["deadline_end"] = end_of_week.strftime('%Y-%m-%d')
                
                # Define keys that filter_tasks_advanced directly understands or are standard
                VALID_FILTER_KEYS_FOR_TRANSFER = {
                    "category", "importance", "project", "tags",
                    "max_effort", "completed"
                    # deadline_start, deadline_end are handled if explicitly provided or set by flags
                }

                for key in VALID_FILTER_KEYS_FOR_TRANSFER:
                    if key in filters_from_json:
                        processed_filters[key] = filters_from_json.pop(key)
                
                # Any remaining keys in filters_from_json are unknown
                unknown_keys = list(filters_from_json.keys())
                # --- END MODIFIED SECTION FOR BOOLEAN DATE FLAGS ---
                
                if unknown_keys:
                    SUPPORTED_INPUT_JSON_KEYS = VALID_FILTER_KEYS_FOR_TRANSFER.union(
                        {"filter_due_today", "filter_due_this_week", "deadline_start", "deadline_end"}
                    )
                    return json.dumps({
                        "type": "error",
                        "error_type": "InvalidFilterKeys",
                        "text_summary": f"❌ The filter contains unrecognized keys: {unknown_keys}. Supported keys: {list(SUPPORTED_INPUT_JSON_KEYS)}.",
                        "suggested_user_action": f"Please use supported filter keys such as {', '.join(list(SUPPORTED_INPUT_JSON_KEYS))}."
                    })
                tasks_to_display = task_manager.filter_tasks_advanced(processed_filters)
            except json.JSONDecodeError as e:
                return json.dumps({
                    "type": "error", "error_type": "InvalidJSONFilter",
                    "text_summary": f"❌ Invalid JSON filter: {e}. Input: '{filter_string[:100]}...'.",
                    "suggested_user_action": "The filter looked like JSON but was invalid. Please provide valid JSON or a simple filter term."
                })
        elif filter_lower == "all":
            tasks_to_display = list(task_manager.tasks)
        elif filter_lower == "high priority":
            tasks_to_display = task_manager.filter_tasks_advanced({"importance": "High", "completed": False})
        elif filter_lower == "medium priority":
            tasks_to_display = task_manager.filter_tasks_advanced({"importance": "Medium", "completed": False})
        elif filter_lower == "low priority":
            tasks_to_display = task_manager.filter_tasks_advanced({"importance": "Low", "completed": False})
        else:
            snapshot = list(task_manager.tasks)
            filter_map = {
                "today": lambda: [t for t in snapshot if is_due_today(t.get('deadline')) and not t.get('completed')],
                "this_week": lambda: [t for t in snapshot if is_due_this_week(t.get('deadline')) and not t.get('completed')],
                "week": lambda: [t for t in snapshot if is_due_this_week(t.get('deadline')) and not t.get('completed')],
                "overdue": lambda: [t for t in snapshot if is_overdue(t.get('deadline')) and not t.get('completed')],
                "completed": lambda: [t for t in snapshot if t.get('completed')],
                "incomplete": lambda: [t for t in snapshot if not t.get('completed')]
            }
            if filter_lower in filter_map:
                tasks_to_display = filter_map[filter_lower]()
            else:
                # Fallback to general search if no specific filter matches
                tasks_to_display = task_manager.search_tasks(filter_string)

        if not tasks_to_display:
            return json.dumps({
                "type": "task_list",
                "text_summary": f"📋 No tasks found matching filter: {filter_string}",
                "structured_tasks": [],
                "filter_used": filter_string
            })

        text_summary_content = f"📋 Found {len(tasks_to_display)} task(s) matching filter: {filter_string}."

        structured_list = []
        for task in tasks_to_display:
            structured_list.append({
                "id": task.get('id', 'unknown'),
                "description": task.get('description', 'N/A'),
                "deadline": task.get('deadline', 'N/A'),
                "deadline_display": display_date_nicely(task.get('deadline')),
                "importance": task.get('importance', 'Medium'),
                "effort": task.get('effort', 'N/A'),
                "category": task.get('category', 'General'),
                "tags": task.get('tags', []),
                "completed": task.get('completed', False),
                "time_spent": task.get('time_spent', 0)
            })

        return json.dumps({
            "type": "task_list",
            "text_summary": text_summary_content,
            "structured_tasks": structured_list,
            "filter_used": filter_string
        })
    except Exception as e:
        logger.error(f"[show_tasks_tool] ERROR: {e} for filter '{filter_string}'", exc_info=True)
        return json.dumps({
            "type": "error",
            "error_type": "ToolExecutionError",
            "text_summary": f"❌ Error showing tasks: {e}",
            "suggested_user_action": "An unexpected error occurred while trying to show tasks. Please try a different filter or simplify your request."
        })
    

@tool
def search_tasks(query: str) -> str:
    """
    SEARCHES for tasks by keyword in description, category, project, or tags.
    Use this ONLY when user wants to FIND specific tasks by keyword.
    Input: search query. Example: "client meeting"
    Returns: JSON string with 'text_summary' and 'structured_tasks', or 'No tasks found' message.
    """
    try:
        # FIX: Strip whitespace from the query for consistent 'query_used' and matching
        clean_query = query.strip()
        tasks = task_manager.search_tasks(clean_query) # task_manager.search_tasks will also strip and lowercase

        if not tasks:
            return json.dumps({
                "type": "search_results",
                "text_summary": f"🔍 No tasks found matching: {clean_query}",
                "structured_tasks": [],
                "query_used": clean_query
            })

        text_summary_lines = [f"\n🔍 Search Results for '{clean_query}' ({len(tasks)} found)", "=" * 50]
        structured_task_list = []

        for i, task in enumerate(tasks, 1):
            status = "✓" if task.get('completed') else "○"
            task_line = f"{status} {i}. {task['description']} (ID: {task.get('id','unknown')[:8]}...)\n"
            task_line += f"   Category: {task.get('category', 'General')} | Deadline: {display_date_nicely(task.get('deadline'))}\n\n"
            text_summary_lines.append(task_line)
            structured_task_list.append({
                "id": task.get('id', 'unknown'),
                "description": task.get('description', 'N/A'),
                "deadline": task.get('deadline', 'N/A'),
                "deadline_display": display_date_nicely(task.get('deadline')),
                "importance": task.get('importance', 'Medium'),
                "effort": task.get('effort', 'N/A'),
                "category": task.get('category', 'General'),
                "tags": task.get('tags', []),
                "completed": task.get('completed', False),
                "time_spent": task.get('time_spent', 0)
            })

        return json.dumps({
            "type": "search_results",
            "text_summary": "\n".join(text_summary_lines),
            "structured_tasks": structured_task_list,
            "query_used": clean_query
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "text_summary": f"❌ Error searching tasks: {str(e)}"
        })

@tool
def track_time(params_string: str) -> str:
    """
    TRACKS time spent on a specific task. Use this ONLY when user wants to RECORD time.
    Input: JSON with task_id and minutes
    Example: {"task_id": "abc123", "minutes": 30}
    Input format: {"task_id": "task_id", "minutes": number}
    Returns: Success message with updated time
    """
    try:
        params = json.loads(params_string)
        task_id = params.get("task_id")
        minutes = params.get("minutes")
        
        if not task_id or minutes is None:
            return "❌ Error: task_id and minutes required"
        
        return task_manager.track_time(task_id, int(minutes))
    except json.JSONDecodeError:
        return "❌ Error: Invalid JSON format for track_time input"
    except Exception as e:
        return f"❌ Error tracking time: {str(e)}"

@tool
def get_analytics(period: str = "week") -> str:
    """
    PROVIDES productivity analytics and insights. Use this ONLY when user asks for STATISTICS or ANALYTICS.
    Input: time period (day/week/month/year)
    Returns: Analytics report with completion stats and time tracking
    """
    try:
        analytics_data = task_manager.get_analytics(period) # Renamed to analytics_data to avoid conflict
        
        output = f"\n📊 Analytics for last {period}\n"
        output += "=" * 50 + "\n"
        output += f"Tasks completed: {analytics_data['tasks_completed']}\n"
        output += f"Overdue tasks: {analytics_data['overdue_tasks']}\n"
        output += f"Average completion time: {analytics_data['average_completion_hours']} hours\n\n"
        
        if analytics_data.get('completion_by_category'): # Use .get() for safety
            output += "Completion by category:\n"
            for category, count in analytics_data['completion_by_category'].items():
                output += f"  • {category}: {count}\n"
            output += "\n" # Add a newline after this section if it exists
        
        # --- START OF UNCOMMENTED AND MODIFIED SECTION ---
        if analytics_data.get('time_spent_by_category'): # Use .get() for safety
            output += "Time spent by category:\n"
            if analytics_data['time_spent_by_category']: # Check if it's not empty
                for category, time_str in analytics_data['time_spent_by_category'].items():
                    output += f"  • {category}: {time_str}\n"
                output += "\n" # Add a newline after this section if it exists
            else:
                output += "  No time tracked by category for this period.\n\n"
        # --- END OF UNCOMMENTED AND MODIFIED SECTION ---
        
        return output
    except Exception as e:
        logger.error(f"Error in get_analytics tool: {str(e)}", exc_info=True) # Add exc_info for traceback
        return f"❌ Error getting analytics: {str(e)}"

@tool
def delete_tasks_by_ids(task_ids_string: str) -> str:
    """
    DELETES tasks from the task list using their unique Task IDs.
    Input: A comma-separated string of Task IDs (full or partial starting segment is acceptable if unique).
    Example: "c1c05f8d,149df98f,abc123ef"
    Returns: A success message indicating how many tasks were deleted, or an error message if IDs are not found.
    """
    raw_identifiers = [ident.strip() for ident in task_ids_string.split(',') if ident.strip()]
    
    if not raw_identifiers:
        return "❌ Error: No Task IDs provided."

    ids_to_delete_resolved = set()
    not_found_identifiers = []
    ambiguous_identifiers = [] # For partial IDs that might match multiple tasks (though _find_task_by_id returns first)

    for ident_str in raw_identifiers:
        task = task_manager._find_task_by_id(ident_str) # _find_task_by_id handles full and partial IDs
        if task:
            # If ident_str was partial, ensure we are adding the full ID
            ids_to_delete_resolved.add(task['id'])
        else:
            not_found_identifiers.append(ident_str)

    if not_found_identifiers:
        return (f"❌ Tasks not found for the following identifiers: {', '.join(not_found_identifiers)}. "
                f"No tasks were deleted. Please ensure the IDs are correct. You can use 'show_tasks all' to verify IDs.")

    if not ids_to_delete_resolved:
        # This case should ideally be caught by the above, but as a safeguard.
        return "ℹ️ No tasks matched the provided IDs for deletion."

    original_task_count = len(task_manager.tasks)
    # Filter out tasks whose IDs are in the set of resolved IDs to delete
    task_manager.tasks = [
        task for task in task_manager.tasks if task['id'] not in ids_to_delete_resolved
    ]
    deleted_count = original_task_count - len(task_manager.tasks)
            
    if deleted_count > 0:
        task_manager.save_tasks()
        return f"✅ Successfully deleted {deleted_count} tasks."
    else:
        # This might happen if the tasks were already deleted by another concurrent process or if logic error
        return "ℹ️ No tasks were deleted. They might have already been removed or there was an issue."


@tool
def mark_task_complete(task_identifier: str) -> str:
    """
    MARKS a task as complete. Use this ONLY when user wants to COMPLETE a task.
    Input: task index (1-based from show_tasks output) or task ID
    Example: "3" or "abc12345"
    Returns: Success message
    """
    try:
        # First try as index
        try:
            task_idx = int(task_identifier.strip()) - 1
            if 0 <= task_idx < len(task_manager.tasks):
                task = task_manager.tasks[task_idx]
            else:
                task = None
        except ValueError:
            # Try as ID
            task = task_manager._find_task_by_id(task_identifier.strip())
        
        if not task:
            return f"❌ Task not found: {task_identifier}"
        
        if task.get('completed'):
            return f"ℹ️ Task '{task['description']}' is already marked as complete"
        
        task['completed'] = True
        task['completed_at'] = datetime.now().isoformat()
        task_manager.save_tasks()
        return f"✅ Task '{task['description']}' marked as complete"
    except Exception as e:
        return f"❌ Error marking task complete: {str(e)}"

# Note: create_template, use_template, export_tasks, import_tasks tools are commented out as per instructions
# Instead, here are placeholders that would be commented out:

# @tool
# def create_template(...):
#     ...

# @tool
# def use_template(...):
#     ...

# @tool
# def export_tasks(...):
#     ...

# @tool
# def import_tasks(...):
#     ...

# Smart suggestion tools
@tool
def what_should_i_work_on(query: str = "") -> str:
    """
    PROVIDES AI-powered task suggestions.
    Returns: JSON string with 'text_summary', 'structured_tasks' (recommended tasks), and 'insights'.
    """
    try:
        suggestions = task_manager.get_smart_suggestions()
        
        text_summary_lines = ["\n🤖 Smart Task Suggestions", "=" * 50]
        recommended_tasks_structured = []

        if suggestions.get("next_task"):
            task = suggestions["next_task"]
            text_summary_lines.append(f"🎯 Next recommended task: {task['description']}")
            text_summary_lines.append(f"   ID: {task.get('id','unknown')[:8]}...")
            text_summary_lines.append(f"   Deadline: {display_date_nicely(task.get('deadline'))}")
            text_summary_lines.append(f"   Importance: {task.get('importance')}")
            text_summary_lines.append(f"   Optimal time: {suggestions.get('optimal_time', 'N/A')}\n")
            recommended_tasks_structured.append({ # For UI, we might want to display this as a special card
                "id": task.get('id', 'unknown'),
                "description": task.get('description', 'N/A'),
                "deadline": task.get('deadline', 'N/A'),
                "deadline_display": display_date_nicely(task.get('deadline')),
                "importance": task.get('importance', 'Medium'),
                "effort": task.get('effort', 'N/A'),
                "category": task.get('category', 'General'),
                "tags": task.get('tags', []),
                "completed": task.get('completed', False),
                "time_spent": task.get('time_spent', 0),
                "is_next_recommendation": True, # Flag for UI
                "optimal_time": suggestions.get('optimal_time', 'N/A')
            })
        
        insights = suggestions.get("insights", [])
        if insights:
            text_summary_lines.append("💡 Insights:")
            for insight in insights:
                text_summary_lines.append(f"   • {insight}")
        
        if not recommended_tasks_structured and not insights:
             text_summary_lines.append("No specific suggestions or insights at this time. All caught up or add more tasks!")


        return json.dumps({
            "type": "smart_suggestions",
            "text_summary": "\n".join(text_summary_lines),
            "structured_tasks": recommended_tasks_structured, # Could be empty or one task
            "insights": insights
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "text_summary": f"❌ Error getting suggestions: {str(e)}"
        })

@tool
def what_should_i_do_next(query: str = "") -> str:
    """
    Alias for what_should_i_work_on. Provides smart task recommendations.
    Use when user asks what to do next.
    """
    return what_should_i_work_on(query)

@tool
def get_suggestions(query: str = "") -> str:
    """
    Alias for what_should_i_work_on. Provides smart task recommendations.
    Use when user asks for suggestions.
    """
    return what_should_i_work_on(query)


@tool
def get_suggested_task_order(filter_criteria_json: str) -> str:
    """
    Retrieves tasks matching the specified filter criteria and orders them by a smart suggestion priority.
    Input: A JSON string specifying filter criteria.
    Returns: JSON string with 'text_summary' and 'structured_tasks', ordered by suggestion priority.
    """
    try:
        filters = json.loads(filter_criteria_json)

        adapted_filters = {}
        now = datetime.now()
        if filters.get("deadline_today"):
            today_str = now.strftime('%Y-%m-%d')
            adapted_filters["deadline_start"] = today_str
            adapted_filters["deadline_end"] = today_str
        if filters.get("deadline_this_week"):
            start_of_week = now.date() - timedelta(days=now.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            adapted_filters["deadline_start"] = start_of_week.strftime('%Y-%m-%d')
            adapted_filters["deadline_end"] = end_of_week.strftime('%Y-%m-%d')
        
        for key in ["category", "importance", "project", "completed"]:
            if key in filters:
                adapted_filters[key] = filters[key]
        
        if not adapted_filters:
            return json.dumps({
                "type": "error",
                "text_summary": "❌ No valid filter criteria provided for ordering. Please specify filters like 'deadline_today', 'category', etc."
            })

        candidate_tasks = task_manager.filter_tasks_advanced(adapted_filters)
        
        if not candidate_tasks:
            return json.dumps({
                "type": "suggested_order",
                "text_summary": f"📋 No tasks found matching your criteria for suggested order: {filter_criteria_json}",
                "structured_tasks": [],
                "filter_used": filter_criteria_json
            })

        # MODIFIED SORTING KEY:
        # Use task_manager to call the new method with consider_time_of_day_for_effort=False
        sorted_tasks = sorted(candidate_tasks, key=lambda t: task_manager._calculate_task_priority(t, consider_time_of_day_for_effort=False), reverse=True)

        text_summary_lines = [f"\n📋 Suggested Task Order ({len(sorted_tasks)} tasks found matching {filter_criteria_json})", "=" * 50]
        structured_task_list = []

        for i, task_item in enumerate(sorted_tasks, 1):
            status = "✓" if task_item.get('completed') else "○"
            task_line = f"{status} {i}. {task_item['description']}\n"
            task_line += f"   ID: {task_item.get('id','unknown')[:8]}... | Deadline: {display_date_nicely(task_item.get('deadline'))}\n"
            task_line += f"   Category: {task_item.get('category', 'General')} | "
            task_line += f"Importance: {task_item.get('importance', 'N/A')} | Effort: {task_item.get('effort', 'N/A')}\n\n"
            text_summary_lines.append(task_line)
            structured_task_list.append({
                "id": task_item.get('id', 'unknown'),
                "description": task_item.get('description', 'N/A'),
                "deadline": task_item.get('deadline', 'N/A'),
                "deadline_display": display_date_nicely(task_item.get('deadline')),
                "importance": task_item.get('importance', 'Medium'),
                "effort": task_item.get('effort', 'N/A'),
                "category": task_item.get('category', 'General'),
                "tags": task_item.get('tags', []),
                "completed": task_item.get('completed', False),
                "time_spent": task_item.get('time_spent', 0)
            })
        
        return json.dumps({
            "type": "suggested_order",
            "text_summary": "\n".join(text_summary_lines),
            "structured_tasks": structured_task_list,
            "filter_used": filter_criteria_json
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "text_summary": f"❌ Error getting suggested task order: {str(e)}"
        })


@tool
def delete_task_flexible(identifier_string: str) -> str:
    """
    DELETES a task. Input can be a Task ID (full or partial) or the task's name/description.
    If a name matches multiple tasks (either exactly or partially), it will ask for clarification.
    Example: "abc123ef" OR "Review Q1 Financials" OR "Call John"
    Returns: Success message, error, or a request for clarification if multiple tasks match a name.
    """
    task_by_id = task_manager._find_task_by_id(identifier_string)
    if task_by_id:
        task_id_to_delete = task_by_id['id']
        task_description = task_by_id['description']
        task_manager.tasks = [t for t in task_manager.tasks if t['id'] != task_id_to_delete]
        task_manager.save_tasks()
        return f"✅ Successfully deleted task: '{task_description}' (ID: {task_id_to_delete[:8]}...)."

    # 2. If not a valid ID (or not found by ID), try as name
    matching_tasks = task_manager.find_tasks_by_name(identifier_string, exact_match=True) # Start with exact name match

    if not matching_tasks:
        # If no exact match, try partial name match
        matching_tasks = task_manager.find_tasks_by_name(identifier_string, exact_match=False) 
        if not matching_tasks:
            return f"❌ No task found matching ID or name: '{identifier_string}'."

    if len(matching_tasks) == 1:
        task_to_delete = matching_tasks[0]
        task_manager.tasks = [t for t in task_manager.tasks if t['id'] != task_to_delete['id']]
        task_manager.save_tasks()
        return f"✅ Successfully deleted task: '{task_to_delete['description']}' (ID: {task_to_delete['id'][:8]}...)."

    if len(matching_tasks) > 1:
        output = f"❓ Found multiple tasks matching '{identifier_string}':\n"
        for i, task_item in enumerate(matching_tasks, 1):
            output += (f"{i}. {task_item['description']} (ID: {task_item['id'][:8]}..., "
                       f"Due: {display_date_nicely(task_item['deadline'])}, "
                       f"Category: {task_item.get('category', 'N/A')})\n")
        output += "Please specify which one to delete by providing its full ID or the number from this list."
        # This response needs to be handled by the agent to re-prompt the user.
        # The agent should then call delete_task_flexible again with the more specific identifier.
        return output
    
    return f"❌ Unexpected error trying to delete task: '{identifier_string}'."

@tool
def add_bulk_tasks_tool(tasks_json_string: str) -> str:
    """
    ADDS multiple tasks to the task list in bulk.
    Input: A JSON string representing a LIST of task objects.
           Each task object MUST have 'description' and 'deadline'.
           Optional fields per task: 'importance', 'effort', 'category', 'project', 'tags', 'recurring'.
           Example: '[{"description": "Task A", "deadline": "tomorrow", "importance": "High"}, {"description": "Task B", "deadline": "next Monday"}]'
    Returns: A summary message indicating how many tasks succeeded or failed, along with individual status messages.
             Tasks with past due dates will be rejected.
    """
    logger.debug(f"[add_bulk_tasks_tool] Received raw input string: {repr(tasks_json_string)}")
    
    processed_string = tasks_json_string.strip()
    # Check if the string is wrapped in an extra layer of single quotes
    if processed_string.startswith("'") and processed_string.endswith("'") and len(processed_string) > 1:
        processed_string = processed_string[1:-1]
        logger.debug(f"[add_bulk_tasks_tool] Stripped outer single quotes, now: {repr(processed_string)}")
    # Also check for double quotes, just in case, though less likely from LLM for this error
    elif processed_string.startswith('"') and processed_string.endswith('"') and len(processed_string) > 1:
        try:
            inner_content_check = json.loads(processed_string)
            if isinstance(inner_content_check, str):
                 processed_string = inner_content_check
                 logger.debug(f"[add_bulk_tasks_tool] Stripped outer double quotes, inner was string, now: {repr(processed_string)}") # CHANGED
        except json.JSONDecodeError:
            pass

    try:
        tasks_data = json.loads(processed_string)
        logger.debug(f"[add_bulk_tasks_tool] Parsed with json.loads: {type(tasks_data)}") # CHANGED
    except json.JSONDecodeError as e1:
        logger.debug(f"[add_bulk_tasks_tool] json.loads failed ({e1}), trying ast.literal_eval on: {repr(processed_string)}") # CHANGED
        try:
            tasks_data = ast.literal_eval(processed_string)
            logger.debug(f"[add_bulk_tasks_tool] Parsed with ast.literal_eval: {type(tasks_data)}") # CHANGED
        except (ValueError, SyntaxError, TypeError) as e2:
            logger.debug(f"[add_bulk_tasks_tool] ast.literal_eval also failed ({e2}).") # CHANGED
            return f"❌ Error: Input was not valid JSON nor a parsable Python literal for tasks. Original JSON error: {e1}. Processed input was: {processed_string[:200]}"

    if not isinstance(tasks_data, list):
        logger.debug(f"[add_bulk_tasks_tool] Parsed data is not a list. Type: {type(tasks_data)}, Value: {str(tasks_data)[:200]}") # CHANGED
        return "❌ Error: Parsed input did not result in a list of task objects."
    if not tasks_data:
        return "ℹ️ No tasks provided in the list to add."

    results = task_manager.add_multiple_tasks(tasks_data)
    
    summary_message = f"Bulk Add Summary: ✅ {results['succeeded']} tasks added successfully. ⚠️ {results['failed']} tasks failed."
    
    if results['messages']:
        detailed_report = "\nDetails:\n" + "\n".join(results['messages'][:5]) 
        if len(results['messages']) > 5:
            detailed_report += f"\n...and {len(results['messages']) - 5} more status messages."
        summary_message += detailed_report
        
    return summary_message


tools = [
    Tool.from_function(
        func=add_task,
        name="add_task",
        description="ADDS a new task to the task list. Use this ONLY when user wants to CREATE a new task."
    ), 
    Tool.from_function(
        func=show_tasks,
        name="show_tasks",
        description="SHOWS tasks with optional filtering. Use this ONLY when user wants to SEE their tasks. Input: Optional filter string (e.g., \"all\", \"today\", \"this_week\", \"overdue\", \"completed\", \"incomplete\", \"high priority\") OR a JSON filter string for advanced filtering. JSON examples: {\"category\": \"Work\", \"filter_due_this_week\": true}, {\"importance\": \"High\", \"deadline_start\": \"YYYY-MM-DD\", \"deadline_end\": \"YYYY-MM-DD\"}. Supported boolean date flags: \"filter_due_today\", \"filter_due_this_week\"."
    ),
    Tool.from_function(
        func=search_tasks,
        name="search_tasks",
        description="SEARCHES for tasks by keyword in description, category, project, or tags."
    ),
    Tool.from_function(
        func=track_time,
        name="track_time",
        description="TRACKS time spent on a specific task. Use this ONLY when user wants to RECORD time."
    ),
    Tool.from_function(
        func=get_analytics,
        name="get_analytics",
        description="PROVIDES productivity analytics and insights. Use this ONLY when user asks for STATISTICS or ANALYTICS."
    ),
    Tool.from_function(
        func=mark_task_complete,
        name="mark_task_complete",
        description="MARKS a task as complete. Use this ONLY when user wants to COMPLETE a task."
    ),
    Tool.from_function(
        func=what_should_i_work_on,
        name="what_should_i_work_on",
        description="PROVIDES AI-powered task suggestions on what to work on next. Use when the user asks 'what should I work on?', 'what should I do next?', or asks for 'suggestions'."
    ),
    Tool.from_function(
        func=get_suggested_task_order,
        name="get_suggested_task_order",
        description="Retrieves tasks matching the specified filter criteria and orders them by a smart suggestion priority."
    ),
    Tool.from_function(
        func=delete_task_flexible,
        name="delete_task_flexible",
        description="DELETES a task. Input can be a Task ID (full or partial) or the task's name/description."
    ),
    Tool.from_function(
        func=add_bulk_tasks_tool,
        name="add_bulk_tasks_tool",
        description="ADDS multiple tasks to the task list in bulk."
    )
]

# Get the standard ReAct prompt
# react_prompt = hub.pull("hwchase17/react")
react_chat_prompt = hub.pull("hwchase17/react-chat")

# Create our custom instructions
custom_instructions = """You are a smart task management assistant. Your job is to help users manage their tasks efficiently.

CRITICAL FORMATTING RULES - YOU MUST FOLLOW THESE EXACTLY:
1. ALWAYS follow this exact format for your responses:
   
   Thought: Your reasoning here
   Action: tool_name
   Action Input: tool_input
   
   Observation: tool will provide this
   
   Final Answer: Your response to the user

2. NEVER use multiple Thought: lines in a row
3. NEVER use Action: without Action Input:
4. ALWAYS end with Final Answer: after getting tool results
5. If you get format errors, simplify your response and try again

# --- CRITICAL BEHAVIORAL GUIDELINES ---
# 1. ACCURACY & RELEVANCE:
#    - Always strive to understand the user's intent. If ambiguous, ask for clarification.
#    - Provide clear, helpful, and concise responses in your Final Answer.
# 2. NON-REPETITION & EFFICIENCY:
#    - If a tool call was successful (e.g., task added) or an error was already reported (e.g. "task not found", "duplicate task"),
#      DO NOT repeat the exact same Action and Action Input. Think about the next logical step or ask for clarification.
#    - If you've already shown tasks with a specific filter in this interaction, DO NOT show them again with the same filter unless explicitly asked to refresh.
#    - If the user asks to see tasks (e.g., "show all tasks", "show high priority tasks") and you, based on chat_history, believe you have *recently* shown this exact information in a previous turn,
#      your Thought should acknowledge this. Your Final Answer should then be something like: "I recently showed you [type of tasks, e.g., 'high priority tasks']. Would you like me to display them again or perhaps use a different filter?"
#      DO NOT simply state "No tasks found" if you are skipping the tool call due to recent display. Only say "No tasks found" if the tool was actually called and returned no results.
# 3. ERROR HANDLING & USER GUIDANCE:
#    - If a tool returns an error (e.g., "Cannot add task... deadline is in the past", "Task not found", "Invalid JSON")
#      or an observation indicates a problem:
#        - DO NOT immediately retry the exact same Action with the exact same Action Input.
#        - Your Thought should acknowledge the error.
#        - Your Final Answer should clearly explain the error to the user and suggest what they should do
#          (e.g., "The deadline you provided is in the past. Please provide a future date.",
#          "I couldn't find a task with that ID. Would you like to see all tasks to find the correct ID?").
#        - If the error is due to invalid input from the user (like a past date), ask the user to provide corrected input.
#          DO NOT attempt to "correct" the input yourself and retry the same failing action.
#        - If a tool's Observation is a JSON error object containing 'suggested_user_action',
#          your Final Answer should use this suggestion to guide the user.
# 4. TOOL USAGE PROTOCOL:
#    - ONLY use ONE tool per Thought/Action/Observation cycle unless a multi-step process is *explicitly and clearly* stated by the user for a single request.

# --- SPECIFIC SCENARIO HANDLING & ADVANCED INTERACTIONS ---
# 1. Multi-Step Deletion/Completion with Clarification:
#    - If the user asks to delete/complete a task by name (e.g., "delete Task B") and a tool
#      (like `search_tasks` or `delete_task_flexible` itself) returns multiple matches, your Final Answer must list these matches
#      (with IDs/numbers) and ask for clarification.
#    - On the user's NEXT turn, if they provide a clarifier (e.g., "delete number 2" or "delete ID xyz"):
#        - Your Thought should recognize this is a follow-up. EXPLICITLY REFER TO the 'Previous conversation history' (chat_history)
#          AND your own previous 'Observation' or 'Final Answer' in the agent_scratchpad to identify which task "number 2"
#          (or the ID) corresponds to from the list you provided in the previous turn.
#        - Carefully map the user's number or ID reference to the correct task from the previous list shown.
#        - Then, directly call `delete_task_flexible` or `mark_task_complete` with the resolved specific ID.
#        - DO NOT re-run `search_tasks` for the original ambiguous name if the user has provided a specific identifier from your list.
# 2. Viewing Specific Task Details (e.g., "show Task X" or "details of Task X"):
#    - First, use `search_tasks` with "Task X".
#    - If `search_tasks` returns one or more tasks, your Final Answer should present these tasks.
#      The UI will use the `structured_tasks` from the `search_tasks` Observation to display cards.
#    - DO NOT call `show_tasks` tool again after `search_tasks` has already found the specific task(s).
# 3. Adding Multiple Tasks (Bulk vs. Sequential):
#    - If the user asks to add multiple tasks in one go, attempt to parse them into a list of JSON objects and use the 'add_bulk_tasks_tool'.
#    - If parsing is too complex or ambiguous, inform the user you will add them one by one using 'add_task' or ask for clarification.
# 4. Handling Multiple Actions in One Request (e.g., "Delete Task A and Task B"):
#    - Address one task at a time.
#    - For "Delete Task A and Task B":
#        1. First, try to resolve and delete "Task A" (using search_tasks then delete_task_flexible, handling ambiguity for "Task A" if it arises).
#        2. After successfully deleting "Task A" (or confirming it doesn't exist), THEN in a subsequent Thought/Action cycle (if agent iterations allow, or prompt user to confirm next step), address "Task B".
#        3. If the first part (e.g., deleting "Task A") requires user clarification, resolve that fully before moving to "Task B".
#        4. Your Final Answer for the turn should reflect the action taken for the *first* part and any pending clarifications.

# TOOL USAGE GUIDELINES:
# - add_task:
#   - Use for creating single new tasks.
#   - Input MUST be a JSON string. Example: {{"description": "Review report", "deadline": "Friday", "importance": "High"}}
#   - If the user provides a simple string like 'Task description by/on/due deadline_text', you MUST convert this into the required JSON format before calling the tool.
#     For example, "Review financial report by next Friday" becomes {{"description": "Review financial report", "deadline": "next Friday"}}.
#   - **EFFORT ESTIMATION: If the user does NOT specify an effort for the task, you MUST try to estimate a reasonable effort
#     (e.g., "30 minutes", "1 hour", "2 hours", "4 hours", "1 day") based on the task description and common sense.
#     Include this estimated effort in the "effort" field of the JSON input. If you are very unsure, you can omit it,
#     and a default will be applied by the system, but aim to provide an estimate.**
#   - **DEADLINE HANDLING & INFERENCE:**  
#     - If the user does not specify a deadline, infer a reasonable one (e.g., 'today', 'tomorrow', 'next week') based on task context and urgency.
#     - **Crucially, for terms like 'ASAP', 'now', 'immediately', 'urgent', or 'by end of day' (EOD), you MUST interpret these as 'today' when populating the 'deadline' field in the JSON input for the `add_task` tool.**
#     - Ensure the 'deadline' value in the JSON is a string that the system can parse (e.g., "today", "tomorrow", "next Monday", "YYYY-MM-DD"). Do not use the unparsed terms like "ASAP" directly in the JSON. 
# - show_tasks: (Description remains largely the same, but emphasize it's for *seeing* tasks)
#   - Use this ONLY when user wants to SEE their tasks. Input: Optional filter string. Examples: "all", "today", "this_week", "overdue",
#     "completed", "incomplete", "high priority", "medium priority", "low priority",
#     or a JSON filter string for advanced filtering (e.g., {{"category": "Work", "importance": "High"}}).
      or a JSON filter string for advanced filtering.
#     Examples of JSON filters:
#       - {{"category": "Work", "importance": "High"}}
#       - {{"category": "Personal", "filter_due_this_week": true}}
#       - {{"importance": "Low", "filter_due_today": true}}
#       - {{"completed": false, "deadline_start": "2024-01-01", "deadline_end": "2024-01-31"}}
#     Valid boolean flags for relative date ranges: "filter_due_today", "filter_due_this_week".
#     Other valid JSON keys for filtering: "category", "importance", "project", "tags", "max_effort", "completed", "deadline_start", "deadline_end".
#     If "deadline_start" and "deadline_end" are provided, they take precedence over boolean date flags.
# - search_tasks: (Description remains largely the same)
#   - SEARCHES for tasks by keyword in description, category, project, or tags.
# - track_time: (Input format emphasis)
#   - TRACKS time spent on a specific task. Input format: {{"task_id": "task_id_value", "minutes": number_value}}
# - get_analytics: (Description remains largely the same)
#   - PROVIDES productivity analytics and insights.
# - what_should_i_work_on: (Description remains largely the same)
#   - PROVIDES AI-powered task suggestions on what to work on next.
# - get_suggested_task_order: (Input format emphasis)
#   - Retrieves tasks matching filter criteria and orders them by smart priority.
#     Input should be a JSON string with filters like {{"deadline_today": true, "category": "Work"}}.
# - mark_task_complete: (ID preference)
#   - MARKS a task as complete. Input: Task ID (preferred) or a 1-based index from the *last relevant 'show_tasks' or 'search_tasks' list*. Using IDs is safer.
# - delete_task_flexible: (Clarification process emphasis)
#   - DELETES a task by Task ID (full or partial) or name.
#   - If a name matches multiple tasks, the tool's Observation will list them. Your Final Answer must present these options to the user and ask for clarification (e.g., by ID).
#   - On the user's next turn, if they provide a specific ID, use that ID directly with this tool.
# - add_bulk_tasks_tool: (Input format emphasis)
#   - ADDS multiple tasks at once. Input MUST be a JSON string representing a LIST of task objects.
#   - Each object needs 'description' and 'deadline'. Optional: 'importance', 'effort', 'category'.
#   - Example: '[{{"description": "Task A", "deadline": "tomorrow"}}, {{"description": "Task B", "deadline": "next week", "importance": "High"}}]'
#   - If the user provides multiple tasks in natural language (e.g., "Add: 1. Task X by Monday. 2. Task Y by Tuesday."),
#     you MUST parse this into the required JSON list format for this tool. If you cannot reliably parse it,
#     inform the user and ask them to provide tasks one by one or in the correct format.

# IMPORTANT GUIDELINES FOR FINAL ANSWERS:
# - When a tool (like show_tasks, search_tasks, get_suggested_task_order, what_should_i_work_on)
#   provides an Observation in JSON format that includes a 'structured_tasks' list (even if empty):
#     - Your 'Final Answer' to the user MUST be a VERY BRIEF, natural language introductory sentence.
#       Examples: "Here are the tasks I found:", "Okay, here are your tasks for today:", "Here's what I recommend:", "I couldn't find any tasks matching that."
#     - The UI will display the detailed tasks as cards using the 'structured_tasks' data from the Observation.
#     - DO NOT list the tasks, their details, or any part of the 'text_summary' from the Observation in your Final Answer text. The cards handle the details.
#     - If 'structured_tasks' is empty in the Observation, your Final Answer should clearly state this concisely.
#       Example: "I couldn't find any tasks matching your criteria." or "There are no tasks due today."
# - For other tools (like add_task, mark_task_complete, delete_task_flexible, get_analytics) that return a simple success, error string, or textual report in their Observation:
#     - Your 'Final Answer' should clearly convey this result or report to the user.
#       Example: "Successfully added task 'XYZ'.", "Task 'ABC' marked as complete.", "Here are your analytics for the week: ...", "Could not find task '123'."

# RESPONSE PROCESS:
# 1. Think about what the user wants.
# 2. Choose the right tool if needed.
# 3. Formulate the Action Input for the tool according to its specific guidelines (especially JSON formats).
# 4. Use the tool.
# 5. Wait for the observation.
# 6. Provide a clear final answer based on the observation and Final Answer guidelines.
# 7. If a task ID is needed and not provided, guide the user to find it using show_tasks or search_tasks.

# Remember: Keep your responses simple and follow the format EXACTLY.
# """
if hasattr(react_chat_prompt, 'messages'):
    # Find the SystemMessage or create one if not present
    system_message_found = False
    for msg_template in react_chat_prompt.messages:
        if msg_template.role == "system": # Langchain's way of denoting system message templates
            # Prepend custom_instructions to existing system message content
            original_system_content = msg_template.prompt.template
            msg_template.prompt.template = custom_instructions + "\n\n" + original_system_content
            system_message_found = True
            break
    if not system_message_found:
        # If no system message, we might need to adapt or insert one.
        # For simplicity, if the prompt is a single string template, we'll prepend there.
        # This part might need adjustment based on the exact structure of "hwchase17/react-chat"
        # For now, let's assume it has a system message or we can modify its main template string.
        # A simpler way if the prompt is a single string:
        # react_chat_prompt.template = custom_instructions + "\n\n" + react_chat_prompt.template
        # Fallback: If complex structure, this might be an oversimplification.
        # The goal is to make `custom_instructions` the primary system guidance.
        # Let's assume react_chat_prompt.template is the main string for now for simplicity of example.
        # A more robust way is to modify the ChatPromptTemplate's messages list.
        # For "hwchase17/react-chat", it's a ChatPromptTemplate.
        # The first message is usually a SystemMessage.
        try:
            # This is a common way to update the system message content
            react_chat_prompt.messages[0].prompt.template = custom_instructions + "\n\n" + react_chat_prompt.messages[0].prompt.template
        except (AttributeError, IndexError, TypeError) as e:
            print(f"Warning: Could not directly prepend custom_instructions to react_chat_prompt.messages[0]. Error: {e}. Will try to prepend to .template if available.")
            if hasattr(react_chat_prompt, 'template'):
                 react_chat_prompt.template = custom_instructions + "\n\n" + react_chat_prompt.template
            else:
                raise ValueError("Cannot determine how to prepend custom_instructions to the pulled prompt. Please inspect the prompt structure.")

else: # If it's a string-based prompt (less likely for "react-chat" but as a fallback)
    react_chat_prompt.template = custom_instructions + "\n\n" + react_chat_prompt.template


print("---- FINAL PROMPT TEMPLATE (react-chat based) ----")
# print(react_chat_prompt.template) # This might be long
print(f"Prompt input variables: {react_chat_prompt.input_variables}") # Should include 'chat_history'

# Create the agent with the modified prompt
agent = create_react_agent(llm, tools, react_chat_prompt)


# Create the Agent Executor with better configuration
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=AGENT_MAX_ITERATIONS, # MODIFIED
    early_stopping_method=AGENT_EARLY_STOPPING_METHOD, # MODIFIED
    return_intermediate_steps=False,
    max_execution_time=60,
)

# Helper functions for date handling
def is_due_today(task_deadline_iso):
    """Checks if a task is due today."""
    if not task_deadline_iso:
        return False
    try:
        deadline_date = datetime.strptime(task_deadline_iso, '%Y-%m-%d').date()
        return deadline_date == datetime.now().date()
    except ValueError:
        return False

def is_due_this_week(task_deadline_iso):
    """Checks if a task is due this week."""
    if not task_deadline_iso:
        return False
    try:
        deadline_date = datetime.strptime(task_deadline_iso, '%Y-%m-%d').date()
        today = datetime.now().date()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        return start_of_week <= deadline_date <= end_of_week
    except ValueError:
        return False

def is_overdue(task_deadline_iso):
    """Checks if a task is overdue."""
    if not task_deadline_iso:
        return False
    try:
        deadline_date = datetime.strptime(task_deadline_iso, '%Y-%m-%d').date()
        return deadline_date < datetime.now().date()
    except ValueError:
        return False

# Main execution flow with better error handling
if __name__ == "__main__":
    print("🤖 Welcome to the Enhanced Task Prioritizer Agent! 🚀")
    print("\n✨ KEY FEATURES:")
    print("• Smart Organization: Categories, projects, and tags")
    print("• Time Management: Track time spent on tasks")
    print("• Recurring Tasks: Set up automated recurring tasks")
    print("• Intelligent Search: Find tasks quickly")
    print("• Analytics: Track your productivity")
    print("• Templates: Reuse common task patterns")
    print("• AI Assistant: Get smart suggestions on what to work on")
    
    print("\n📝 EXAMPLE COMMANDS:")
    print("• 'Add a task to review client proposal by Friday'")
    print("• 'Show me all work tasks'")
    print("• 'What should I work on next?'")
    print("• 'Track 30 minutes on task abc123'")
    print("• 'Create a template called Weekly Review'")
    print("• 'Export my tasks to markdown'")
    
    print("\n💡 TIP: Use natural language - I'll understand what you need!")
    print("\nType 'exit' to quit.\n")
    
    while True:
        user_input = input(">>> You: ").strip()
        if user_input.lower() == 'exit':
            print("\n👋 Thanks for using Task Prioritizer! Your tasks have been saved.")
            break
        if not user_input:
            continue
        
        try:
            # Reset action tracker for new request
            action_tracker.reset()
            
            # Generate current execution ID
            current_exec_id = str(uuid.uuid4())
            
            # Set execution ID for this request
            task_manager.current_execution_id = current_exec_id
            
            # Clear any cached state specific to a single agent interaction/execution
            task_manager.last_show_filter = None
            task_manager.last_show_execution_id = None # Specific for show_tasks tool
            task_manager.last_add_task_input = None
            task_manager.last_add_execution_id = None # Specific for add_task tool
            
            # Execute the agent with timeout handling
            try:
                response = agent_executor.invoke(
                    {"input": user_input},
                    config={
                        "callbacks": [action_tracker],
                        "metadata": {"execution_id": current_exec_id}
                    }
                )
            except TimeoutError:
                print("\n⏱️ Request timed out. Please try a simpler request.")
                continue
            
            # Check if agent should have stopped earlier
            if action_tracker.should_stop():
                print("\n⚠️ Agent appeared to be stuck or hit a limit. Here's what I understood:")
                # Provide a fallback response based on the user's intent
                if "add" in user_input.lower() and "task" in user_input.lower():
                    output = "I can help you add a task. Please provide: description and deadline."
                elif "show" in user_input.lower() or "list" in user_input.lower():
                    output = "To show tasks, try: 'Show all tasks' or 'Show today's tasks'"
                elif "what" in user_input.lower() and "next" in user_input.lower():
                    output = "To get recommendations, try: 'What should I work on?'"
                else:
                    output = "I had trouble understanding that. Could you please rephrase?"
            else:
                # Extract clean output
                output = response.get('output', '')
                
                # Clean up any format artifacts
                if 'Final Answer:' in output:
                    output = output.split('Final Answer:')[-1].strip()
                
                # Remove any ANSI escape codes
                output = re.sub(r'\x1b\[[0-9;]*[mK]', '', output)
            
            print(f"\n<<< Assistant: {output}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Thanks for using Task Prioritizer!")
            break
        except Exception as e:
            print(f"\n❌ Error in main loop: {str(e)}")
            print("Please try rephrasing your request or use a simpler command.\n")