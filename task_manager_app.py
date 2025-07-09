# -*- coding: utf-8 -*-
"""
Created on Fri May 16 18:33:48 2025

@author: merto
"""

import streamlit as st
import time
import re
import os
import sys
import json
from datetime import datetime
from task_prioritizer_agent import action_tracker

# Import your existing task manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_prioritizer_agent import agent_executor

# Page config
st.set_page_config(
    page_title="AI Task Assistant",
    page_icon="✅",
    layout="centered"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
    /* Chat message containers */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        animation: slide-in 0.3s ease-out;
        color: #222;  /* dark readable text */
    }

    .chat-message.user {
        background-color: #e1f5fe; /* Light blue */
        border-bottom-right-radius: 0.2rem;
    }

    .chat-message.assistant {
        background-color: #f1f8e9; /* Light green */
        border-bottom-left-radius: 0.2rem;
    }

    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }

    .chat-message .message {
        width: 80%;
        padding-left: 1rem;
    }

    /* Typing indicator */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background-color: #eeeeee;
        border-radius: 1rem;
        margin: 0.5rem 0;
    }

    .typing-indicator span {
        height: 8px;
        width: 8px;
        background-color: #333;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
        animation: blink 1.3s infinite;
    }

    .task-list {
        padding: 0.5rem;
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    /* Task Cards */
    .task-card {
        border-left: 3px solid #4CAF50;
        margin: 8px 0;
        padding: 10px;
        border-radius: 4px;
        background-color: #ffffff;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .task-card:hover {
        transform: translateX(5px);
    }
    .task-card.high-priority {
        border-left-color: #e53935;
    }
    .task-card.medium-priority {
        border-left-color: #fb8c00;
    }
    .task-card.low-priority {
        border-left-color: #1e88e5;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 0.5rem;
    }
    .badge-priority-high {
        background-color: #e53935;
        color: white;
    }
    .badge-priority-medium {
        background-color: #fb8c00;
        color: white;
    }
    .badge-priority-low {
        background-color: #1e88e5;
        color: white;
    }
    .badge-category-default {
        background-color: #9e9e9e;
        color: white;
    }
    .badge-category-work {
        background-color: #42a5f5;
        color: white;
    }
    .badge-category-personal {
        background-color: #66bb6a;
        color: white;
    }
    .badge-deadline {
        background-color: #8d6e63;
        color: white;
    }

    /* Overdue styling */
    .task-card.overdue {
        background-color: #ffebee;
        border-left: 4px solid #c62828;
        box-shadow: 0 0 8px rgba(198, 40, 40, 0.2);
    }

    .overdue-indicator {
        background-color: #c62828;
        color: white;
        font-size: 0.75rem;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        animation: pulse-warning 2s infinite;
    }

    /* App header */
    .app-header {
        background: linear-gradient(90deg, #bbdefb 0%, #c8e6c9 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(120, 120, 120, 0.4);
        border-radius: 3px;
    }

    /* Animation for slide in */
    @keyframes slide-in {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>

""", unsafe_allow_html=True)

# Mapping for category icons
category_icons = {
    "Work": "💼",
    "Personal": "🏠",
    "General": "📝",
    "Calls": "📞",
    "Meetings": "🗓️",
    "Emails": "📧",
    "Default": "📌" # Default icon
}

# Mapping for category badge CSS classes
category_badge_classes = {
    "Work": "badge-category-work",
    "Personal": "badge-category-personal",
    "General": "badge-category-general",
    "Calls": "badge-category-calls",
    "Meetings": "badge-category-meetings",
    "Emails": "badge-category-emails",
    "Default": "badge-category-default" # Default class
}


# App branding header
st.markdown("""
    <div class="app-header">
        <span style="font-size: 2rem;"></span>
        <h1>Task Genius Agent</h1>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you manage your tasks today? 👋", "emoji": "🤖", "animation_class": ""}
    ]
else:
    # Ensure all existing messages have the animation_class field
    for msg in st.session_state.messages:
        if "animation_class" not in msg:
            msg["animation_class"] = ""

if 'input_field_value' not in st.session_state:
    st.session_state.input_field_value = ""
    
# Session state for last structured tasks from a TOOL's observation
#if 'last_tool_structured_tasks' not in st.session_state:
    st.session_state.last_tool_structured_tasks = None
#if 'last_tool_filter_for_structured_tasks' not in st.session_state:
    st.session_state.last_tool_filter_for_structured_tasks = None
#if 'last_tool_insights' not in st.session_state:
    st.session_state.last_tool_insights = None
# To track if the current assistant message corresponds to the last tool call that produced structured tasks
#if 'assistant_message_is_for_last_tool_list' not in st.session_state:
    st.session_state.assistant_message_is_for_last_tool_list = False

# Helper function to add emojis based on message content
def add_emoji_to_message(message, role):
    animation_class = ""
    
    if role == "user":
        return {"role": role, "content": message, "emoji": "👤", "animation_class": animation_class}
    
    emoji = "🤖"
    if re.search(r'successfully added task|added.+task', message.lower()):
        emoji = "✅"
        animation_class = "success-flash"
    elif re.search(r'similar task already exists|already exists', message.lower()):
        emoji = "⚠️"
        animation_class = "warning-flash"
    elif re.search(r'tasks found|all tasks', message.lower()):
        emoji = "📋"
    elif re.search(r'recommend|suggest', message.lower()):
        emoji = "💡"
    elif re.search(r'error|not found|invalid', message.lower()):
        emoji = "❌"
        animation_class = "error-flash"
    elif re.search(r'analytics|statistics|report', message.lower()):
        emoji = "📊"
    elif re.search(r'tracked|minutes|time spent', message.lower()):
        emoji = "⏱️"
    
    return {"role": role, "content": message, "emoji": emoji, "animation_class": animation_class}

# Helper function to parse effort string to minutes (simplified)
def parse_effort_to_minutes_for_ui(effort_str: str) -> int:
    if not effort_str or not isinstance(effort_str, str):
        return 0
    
    effort_str = effort_str.lower()
    total_minutes = 0
    
    # Parse days (assuming 8 hours per day as per agent logic)
    days_match = re.search(r'(\d+)\s*(?:days?|d)', effort_str)
    if days_match:
        total_minutes += int(days_match.group(1)) * 8 * 60
    
    # Parse hours
    hours_match = re.search(r'(\d+)\s*(?:hours?|hrs?|hr?|h)', effort_str)
    if hours_match:
        total_minutes += int(hours_match.group(1)) * 60
    
    # Parse minutes
    mins_match = re.search(r'(\d+)\s*(?:minutes?|mins?|min?|m)', effort_str)
    if mins_match:
        total_minutes += int(mins_match.group(1))
    
    # If no specific unit found but numbers are present, assume hours if a larger number, or minutes if small
    if total_minutes == 0:
        try:
            # Find all numbers, take the first one
            num_val_match = re.search(r'(\d+)', effort_str)
            if num_val_match:
                num_val = int(num_val_match.group(1))
                # Simple heuristic: if number is > 24, unlikely to be hours for a single effort entry,
                # but could be minutes. If <=24, could be hours.
                # This is very basic; agent's parsing is more robust.
                if num_val > 3 and num_val <= 24 : # e.g. "2", "3" likely hours
                    total_minutes = num_val * 60
                elif num_val <= 180: # e.g. "30", "90" likely minutes
                    total_minutes = num_val
                # else, it's ambiguous without units, leave as 0 or handle as default
        except:
            pass # Could not parse as number
            
    return total_minutes

# Helper function to get effort badge class
def get_effort_badge_class(effort_str: str) -> str:
    minutes = parse_effort_to_minutes_for_ui(effort_str)
    if minutes == 0: # Unparsed or zero effort
        return "badge-effort-default"
    elif minutes <= 120:  # Up to 2 hours (inclusive)
        return "badge-effort-low"
    elif minutes <= 180:  # > 2 hours and up to 3 hours (inclusive)
        return "badge-effort-medium"
    else:  # More than 3 hours
        return "badge-effort-high"

def render_structured_task_cards(tasks_data: list):
    if not tasks_data:
        st.markdown("<p>No tasks to display for this view.</p>", unsafe_allow_html=True)
        return

    for task in tasks_data:
        priority_class = "high-priority" if task.get('importance', '').lower() == "high" else \
                         "medium-priority" if task.get('importance', '').lower() == "medium" else \
                         "low-priority"
        priority_badge_class = f"badge-priority-{task.get('importance', 'medium').lower()}"
        
        task_category = task.get('category', 'Default') # Use 'Default' if category is missing
        cat_icon = category_icons.get(task_category, category_icons["Default"])
        cat_badge_class = category_badge_classes.get(task_category, category_badge_classes["Default"])

        effort_value = task.get("effort", "N/A")
        effort_badge_class = get_effort_badge_class(effort_value)

        # Check if task is overdue
        is_task_overdue = False
        deadline_str = task.get('deadline')
        if deadline_str and deadline_str != 'N/A':
            try:
                deadline_date = datetime.strptime(deadline_str, '%Y-%m-%d').date()
                is_task_overdue = deadline_date < datetime.now().date()
            except ValueError:
                pass

        # Start building the card HTML
        overdue_class = " overdue" if is_task_overdue else ""
        card_html_parts = [
            f'<div class="task-card {priority_class}{overdue_class}" data-id="{task.get("id", "unknown")}">',
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">',
            f'<div>',
            f'<span class="badge {priority_badge_class}">{task.get("importance", "N/A")}</span>',
            f'<span class="badge {cat_badge_class}">{cat_icon} {task.get("category", "N/A")}</span>',
            f'<span class="badge {effort_badge_class}">Effort: {effort_value}</span>',
            f'</div>',
            f'<div style="display: flex; align-items: center;">',
        ]
        
        # Add overdue indicator next to the deadline if task is overdue
        if is_task_overdue:
            card_html_parts.append(
                f'<span class="overdue-indicator"><span class="overdue-icon">⚠️</span>OVERDUE</span>'
            )
        
        card_html_parts.extend([
            f'<span class="badge badge-deadline">Due: {task.get("deadline_display", "N/A")}</span>',
            f'</div>',
            f'</div>',
            f'<div style="font-weight: bold; margin-bottom: 5px;">{task.get("description", "N/A")}</div>',
            f'<div style="font-size: 0.8rem; color: #aaa;">ID: {task.get("id", "unknown")[:8]}...</div>'
        ])

        if task.get('is_next_recommendation'):
            card_html_parts.append(
                f'<div style="font-size: 0.8rem; color: #4CAF50; margin-top: 5px;">'
                f'✨ Optimal time: {task.get("optimal_time", "N/A")}'
                f'</div>' 
            )
        
        card_html_parts.append('</div>') 
        
        st.markdown("".join(card_html_parts), unsafe_allow_html=True)


# Function to display chat messages with animations and enhanced formatting
def display_all_messages():
    for msg_idx, msg in enumerate(st.session_state.messages):
        role_class = msg["role"]
        animation_class = msg.get("animation_class", "")
        emoji = msg["emoji"]
        content = msg["content"] # This is the agent's Final Answer string

        avatar_url_seed = "user" if role_class == "user" else "assistant"
        avatar_html = f"""<img src="https://api.dicebear.com/7.x/bottts/svg?seed={avatar_url_seed}" alt="{role_class.capitalize()} Avatar">"""

        display_content = str(content)
        # `cleaned_display_content` will be the variable passed to st.markdown
        cleaned_display_content = display_content 

        # If the agent's output (content) contains literal \uXXXX sequences,
        # they need to be converted to actual Unicode characters.
        if '\\u' in display_content: 
            try:
                import codecs
                cleaned_display_content = codecs.decode(display_content, 'unicode_escape')
            except Exception:
                cleaned_display_content = display_content 

        main_content_html = f"""
        <div class="chat-message {role_class} {animation_class}">
            <div class="avatar">{avatar_html}</div>
            <div class="message">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{emoji}</span>
                {cleaned_display_content}
        """
        st.markdown(main_content_html, unsafe_allow_html=True)

        if role_class == "assistant":
            if msg.get("structured_tasks_to_display"):
                st.markdown("""<div class="task-cards-container" style="margin-top: 10px;">""", unsafe_allow_html=True)
                render_structured_task_cards(msg["structured_tasks_to_display"])
                st.markdown("""</div>""", unsafe_allow_html=True)

            if msg.get("insights_to_display"):
                st.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
                st.markdown("<h5>💡 Insights:</h5>", unsafe_allow_html=True)
                insights_html = "<ul style='padding-left: 20px; margin-bottom: 0;'>"
                for insight_item in msg["insights_to_display"]:
                    insights_html += f"<li>{insight_item}</li>"
                insights_html += "</ul>"
                st.markdown(insights_html, unsafe_allow_html=True)
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)


# Function to generate contextual suggestions
def generate_suggestions(last_message):
    suggestions = []
    
    # Check if user has actually started interacting
    user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
    if not user_messages:
        return []  # No suggestions until user starts talking
    
    # Get task statistics for more intelligent suggestions
    try:
        from task_prioritizer_agent import task_manager as tm
        total_tasks = len([t for t in tm.tasks if not t.get('completed')])
        overdue_tasks = len([t for t in tm.tasks if not t.get('completed') and t.get('deadline') and 
                           datetime.strptime(t['deadline'], '%Y-%m-%d').date() < datetime.now().date()])
        today_tasks = len([t for t in tm.tasks if not t.get('completed') and t.get('deadline') and 
                         datetime.strptime(t['deadline'], '%Y-%m-%d').date() == datetime.now().date()])
        high_priority_tasks = len([t for t in tm.tasks if not t.get('completed') and t.get('importance') == 'High'])
    except:
        total_tasks = 0
        overdue_tasks = 0
        today_tasks = 0
        high_priority_tasks = 0
    
    # Always provide contextual suggestions based on conversation state
    if st.session_state.messages:
        # Look at the last few messages to understand context
        recent_messages = st.session_state.messages[-3:]  # Last 3 messages
        context = " ".join([msg["content"].lower() for msg in recent_messages])
        last_assistant_msg = next((msg["content"].lower() for msg in reversed(st.session_state.messages) if msg["role"] == "assistant"), "")
        
        # Dynamic suggestions based on context and task state
        if "error" in context or "not found" in context or "invalid" in context:
            suggestions = [
                "Show all my tasks",
                "What should I work on next?",
                "Show today's tasks" if today_tasks > 0 else "Add a new task"
            ]
        elif "successfully added" in last_assistant_msg:
            suggestions = [
                "Show all my tasks",
                "Add another task",
                "What should I work on next?" if total_tasks > 1 else "Show my new task"
            ]
        elif "marked as complete" in last_assistant_msg:
            suggestions = [
                f"Show remaining {total_tasks} tasks" if total_tasks > 0 else "All tasks completed!",
                "Track time on completed task",
                "What should I work on next?" if total_tasks > 0 else "Add a new task"
            ]
        elif any(phrase in context for phrase in ["show", "tasks found", "here are"]):
            # After showing tasks, offer different actions
            if overdue_tasks > 0:
                suggestions.append(f"Focus on {overdue_tasks} overdue tasks")
            if high_priority_tasks > 0:
                suggestions.append("Show high priority tasks")
            suggestions.extend([
                "What should I work on next?",
                "Add a new task",
                "Get productivity analytics"
            ])
        elif "track" in context or "time" in context:
            suggestions = [
                "Show time tracking report",
                "Track more time",
                "Show completed tasks"
            ]
        else:
            # Smart default suggestions based on current state
            suggestions = []
            
            # Priority-based suggestions
            if overdue_tasks > 0:
                suggestions.append(f"Show {overdue_tasks} overdue tasks")
            elif today_tasks > 0:
                suggestions.append(f"Show {today_tasks} tasks due today")
            elif high_priority_tasks > 0:
                suggestions.append("Show high priority tasks")
            
            # Always include these helpful options
            suggestions.extend([
                "What should I work on next?" if total_tasks > 0 else "Add your first task",
                "Show this week's tasks" if total_tasks > 0 else "Create a weekly plan"
            ])
    
    # Time-based contextual additions
    current_hour = datetime.now().hour
    day_of_week = datetime.now().strftime('%A')
    
    if current_hour < 10:  # Morning
        if "Show today's tasks" not in suggestions and today_tasks > 0:
            suggestions.insert(0, "Show today's tasks")
    elif current_hour >= 16:  # Late afternoon
        if not any("overdue" in s for s in suggestions) and overdue_tasks > 0:
            suggestions.insert(0, "Review overdue tasks")
    
    if day_of_week == 'Monday' and "Show this week's tasks" not in suggestions:
        suggestions.append("Plan this week's tasks")
    elif day_of_week == 'Friday' and "Get productivity analytics" not in suggestions:
        suggestions.append("Get weekly analytics")
    
    # Remove duplicates and empty suggestions
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s and s not in seen and s != "All tasks completed!" or (s == "All tasks completed!" and total_tasks == 0):
            seen.add(s)
            unique_suggestions.append(s)
    
    return unique_suggestions[:3]  # Limit to 3 suggestions

# Display task progress summary if we have enough history
def display_progress_summary():
    try:
        from task_prioritizer_agent import task_manager as agent_tm
        all_tasks = agent_tm.tasks 
        if not all_tasks:
            return 

        total_task_count = len(all_tasks)
        completed_task_count = sum(1 for task in all_tasks if task.get('completed'))
        
        progress_percent = (completed_task_count / total_task_count) * 100 if total_task_count > 0 else 0
        
        analytics_summary = agent_tm.get_analytics(period="all_time") 
        overdue_count = analytics_summary.get('overdue_tasks', 0)

    except ImportError:
        st.error("Could not load task manager for progress summary.")
        return
    except Exception as e:
        return

    if total_task_count > 0:
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-title">Task Progress ({completed_task_count} / {total_task_count} Completed)</div>
            <div class="progress-bar">
                <div class="progress-value" style="width: {progress_percent:.2f}%;"></div>
            </div>
            <div class="progress-stats">
                <div>Completed: {progress_percent:.0f}%</div>
                <div>Overdue: {overdue_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Main Script Logic ---

submitted_text_for_processing = None
if st.session_state.input_field_value:
    submitted_text_for_processing = st.session_state.input_field_value
    st.session_state.input_field_value = ""

if submitted_text_for_processing:
    user_message_obj = add_emoji_to_message(submitted_text_for_processing, "user")
    st.session_state.messages.append(user_message_obj)

display_all_messages()

# Show typing indicator only when processing
if submitted_text_for_processing:
    typing_indicator_placeholder = st.empty()
    typing_html = """
    <div class="typing-indicator">
        <span></span>
        <span></span>
        <span></span>
    </div>
    """
    typing_indicator_placeholder.markdown(typing_html, unsafe_allow_html=True)
    
    try:
        response = agent_executor.invoke(
            {"input": submitted_text_for_processing},
            config={"callbacks": [action_tracker]}
        )
        assistant_response_content = response.get('output', 'I could not process that command.')
    
        structured_tasks_for_this_message = None
        insights_for_this_message = None
    
        if action_tracker.last_tool_result:
            try:
                tool_data = json.loads(action_tracker.last_tool_result)
                if isinstance(tool_data, dict) and "type" in tool_data:
                    if tool_data["type"] in ["task_list", "search_results", "suggested_order", "smart_suggestions"]:
                        if "structured_tasks" in tool_data:
                            structured_tasks_for_this_message = tool_data["structured_tasks"]
                        if tool_data["type"] == "smart_suggestions" and "insights" in tool_data:
                            insights_for_this_message = tool_data["insights"]
            except (json.JSONDecodeError, TypeError):
                pass 
            
            action_tracker.last_tool_result = None 
    
    except Exception as e:
        assistant_response_content = f"Sorry, I encountered an error: {str(e)}"
        structured_tasks_for_this_message = None 
        insights_for_this_message = None
        if hasattr(action_tracker, 'last_tool_result'): 
            action_tracker.last_tool_result = None
    
    time.sleep(1.5) 
    typing_indicator_placeholder.empty()
    
    assistant_message_obj = add_emoji_to_message(assistant_response_content, "assistant")
    
    if structured_tasks_for_this_message is not None:
        assistant_message_obj["structured_tasks_to_display"] = structured_tasks_for_this_message
    if insights_for_this_message is not None:
        assistant_message_obj["insights_to_display"] = insights_for_this_message
    
    st.session_state.messages.append(assistant_message_obj)
    st.rerun()

# Always show contextual suggestions (persistent and proactive)
# This is now ALWAYS shown, not just after processing
suggestions = generate_suggestions("")  # Generate based on conversation state

# Check if user has actually interacted (more than just the initial greeting)
user_has_interacted = any(msg["role"] == "user" for msg in st.session_state.messages)

# Place suggestions after messages but before input - only after user interaction
if suggestions and user_has_interacted:  # Show only after user starts conversation
    # Add custom styling for suggestion area
    st.markdown("""
    <style>
    .suggestion-area {
        background-color: rgba(59, 66, 83, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton > button[kind="secondary"] {
        background-color: rgba(59, 66, 83, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.9rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: rgba(69, 76, 93, 0.9);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create suggestion area with fade-in animation
    with st.container():
        st.markdown('<div class="suggestion-area">', unsafe_allow_html=True)
        
        # Header
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("**💡 Suggested next actions**")
        
        # Suggestions
        cols = st.columns(3)
        for idx, suggestion in enumerate(suggestions):
            with cols[idx % 3]:
                if st.button(
                    suggestion, 
                    key=f"suggestion_{idx}",
                    type="secondary",
                    use_container_width=True
                ):
                    st.session_state.input_field_value = suggestion
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

display_progress_summary()

# Input field at the bottom
with st.container():
    st.text_input(
        "UserInputCommand",
        placeholder="Type your command (e.g., 'Add task: Review documents by Friday')",
        key="input_field_value",
        label_visibility="collapsed"
    )

with st.expander("💬 Example Commands"):
    st.markdown("""
    Try these commands:
    
    - **Add a task**: "Add task: Review project proposal by next Friday 📝"
    - **View tasks**: "Show me all my tasks 📋" or "Show today's tasks"
    - **Get recommendations**: "What should I work on next? 🤔"
    - **Track time**: "Track 30 minutes on task [ID] ⏱️"
    - **Add task with details**: "Add task: Prepare presentation by Monday, importance: High, category: Work 🗂️"
    
    The assistant understands natural language, so feel free to phrase your requests conversationally!
    """)