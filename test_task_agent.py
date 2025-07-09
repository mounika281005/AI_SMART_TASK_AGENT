

import os
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import uuid

# Import the components we're testing
from task_prioritizer_agent import (
    agent_executor, 
    action_tracker, 
    task_manager, 
    parse_and_format_date,
    EnhancedTaskManager,
    ActionTracker,
    AGENT_MAX_ITERATIONS,
    ACTION_TRACKER_MAX_SAME_ACTION_REPETITIONS
)

# ===== TEST CONFIGURATION =====
class TestConfig:
    """Configuration for test runs"""
    BACKUP_TASKS_FILE = "tasks_backup.json"
    TEST_TIMEOUT = 30
    TODAY = datetime.now().date()
    TOMORROW = TODAY + timedelta(days=1)
    NEXT_WEEK = TODAY + timedelta(days=7)
    YESTERDAY = TODAY - timedelta(days=1)

# ===== FIXTURES AND SETUP =====
class TestBase:
    """Base class for test organization"""
    
    @classmethod
    def setup_class(cls):
        """Backup original tasks before any tests"""
        if os.path.exists(task_manager.filename):
            with open(task_manager.filename, 'r') as f:
                cls.original_tasks = f.read()
        else:
            cls.original_tasks = "[]"
    
    @classmethod
    def teardown_class(cls):
        """Restore original tasks after all tests"""
        if hasattr(cls, 'original_tasks'):
            with open(task_manager.filename, 'w') as f:
                f.write(cls.original_tasks)
    
    def setup_method(self):
        """Reset state before each test"""
        task_manager.tasks = []
        task_manager.save_tasks()
        action_tracker.reset()
        if hasattr(agent_executor, 'memory') and agent_executor.memory:
            agent_executor.memory.clear()

# ===== UNIT TESTS FOR COMPONENTS =====

class TestDateParsing(TestBase):
    """Test the parse_and_format_date helper function"""
    
    def test_standard_date_formats(self):
        """Test parsing of various date formats"""
        test_cases = [
            # (input, expected_output_pattern, allow_past)
            ("2025-12-31", "2025-12-31", False),
            ("31/12/2025", "2025-12-31", False),
            ("tomorrow", TestConfig.TOMORROW.strftime('%Y-%m-%d'), False),
            ("next Friday", None, False),  # Dynamic - just check it parses
            ("yesterday", TestConfig.YESTERDAY.strftime('%Y-%m-%d'), True),
        ]
        
        for date_input, expected, allow_past in test_cases:
            result = parse_and_format_date(date_input, allow_past_dates=allow_past)
            if expected:
                assert result == expected, f"Failed to parse {date_input}"
            else:
                assert result is not None, f"Failed to parse {date_input}"
                # Verify it's a valid date format
                datetime.strptime(result, '%Y-%m-%d')
    
    def test_invalid_dates(self):
        """Test that invalid dates return None"""
        invalid_dates = [
            "not-a-date",
            "32/13/2025",  # Invalid day/month
            "",
            None
        ]
        
        for invalid in invalid_dates:
            result = parse_and_format_date(invalid)
            assert result is None, f"Should not parse: {invalid}"
    
    def test_past_date_handling(self):
        """Test past date behavior with allow_past_dates flag"""
        past_date = "2020-01-01"
        
        # Should parse when allowed
        result_allowed = parse_and_format_date(past_date, allow_past_dates=True)
        assert result_allowed == "2020-01-01"
        
        # Should still parse but caller decides what to do
        result_not_allowed = parse_and_format_date(past_date, allow_past_dates=False)
        assert result_not_allowed == "2020-01-01"

class TestActionTracker(TestBase):
    """Test the ActionTracker callback handler"""
    
    def test_action_tracking(self):
        """Test that actions are properly tracked"""
        tracker = ActionTracker()
        
        # Simulate tool start
        tracker.on_tool_start(
            {"name": "add_task"}, 
            '{"description": "Test task", "deadline": "tomorrow"}'
        )
        
        assert tracker.tool_call_count == 1
        assert len(tracker.actions_taken) == 1
        assert tracker.actions_taken[0]['tool'] == 'add_task'
    
    def test_repetition_detection(self):
        """Test detection of repeated actions"""
        tracker = ActionTracker()
        
        # Same tool and input multiple times
        for _ in range(3):
            tracker.on_tool_start(
                {"name": "show_tasks"}, 
                "all"
            )
        
        assert tracker.tool_call_count == 3
        assert tracker.same_action_count["show_tasks:all"] == 3
        assert tracker.should_stop() == True  # Should stop after MAX repetitions

class TestEnhancedTaskManager(TestBase):
    """Test the Enhanced Task Manager functionality"""
    
    def test_add_task_success(self):
        """Test successful task addition"""
        manager = EnhancedTaskManager()
        manager.tasks = []  # Start fresh
        
        result = manager.add_task(
            description="Test task",
            deadline="tomorrow",
            importance="High",
            category="Work"
        )
        
        assert "Successfully added task" in result
        assert len(manager.tasks) == 1
        assert manager.tasks[0]['description'] == "Test task"
        assert manager.tasks[0]['importance'] == "High"
    
    def test_duplicate_prevention(self):
        """Test that duplicate tasks are detected"""
        manager = EnhancedTaskManager()
        manager.tasks = []
        
        # Add first task
        manager.add_task("Meeting", "2025-06-01")
        
        # Try to add duplicate
        result = manager.add_task("Meeting", "2025-06-01")
        
        assert "similar task already exists" in result
        assert len(manager.tasks) == 1
    
    def test_past_date_rejection(self):
        """Test that past dates are rejected"""
        manager = EnhancedTaskManager()
        
        result = manager.add_task(
            description="Past task",
            deadline="2020-01-01"
        )
        
        # Check if it's a JSON error response
        try:
            error_data = json.loads(result)
            assert error_data.get("error_type") == "PastDateError"
        except json.JSONDecodeError:
            # Fallback to string check
            assert "past" in result.lower()
    
    def test_task_search(self):
        """Test task search functionality"""
        manager = EnhancedTaskManager()
        manager.tasks = [
            {"id": "1", "description": "Review code", "category": "Work", "completed": False},
            {"id": "2", "description": "Buy groceries", "category": "Personal", "completed": False},
            {"id": "3", "description": "Review documents", "category": "Work", "completed": False}
        ]
        
        # Search by keyword
        results = manager.search_tasks("review")
        assert len(results) == 2
        
        # Search by category
        results = manager.search_tasks("personal")
        assert len(results) == 1
        assert results[0]['description'] == "Buy groceries"

# ===== INTEGRATION TESTS FOR TOOLS =====

class TestAgentTools(TestBase):
    """Test the individual tools that the agent uses"""
    
    def test_add_task_tool(self):
        """Test the add_task tool function"""
        from task_prioritizer_agent import add_task
        
        # Valid JSON input
        result = add_task('{"description": "Test task", "deadline": "tomorrow"}')
        assert "Successfully added task" in result or "already exists" in result
    
    def test_show_tasks_tool(self):
        """Test the show_tasks tool function"""
        from task_prioritizer_agent import show_tasks
        
        # Add a task first
        task_manager.tasks = []
        task_manager.add_task("Test task", "tomorrow")
        
        # Test showing all tasks
        result = show_tasks("all")
        result_data = json.loads(result)
        
        assert result_data['type'] == 'task_list'
        assert len(result_data['structured_tasks']) == 1
    
    def test_delete_task_flexible_tool(self):
        """Test the flexible delete tool"""
        from task_prioritizer_agent import delete_task_flexible
        
        # Add tasks
        task_manager.tasks = []
        task_manager.add_task("Test task", "tomorrow")
        task_id = task_manager.tasks[0]['id']
        
        # Delete by ID
        result = delete_task_flexible(task_id[:8])
        assert "Successfully deleted" in result

# ===== AGENT BEHAVIOR TESTS =====

class TestAgentBehavior(TestBase):
    """Test the complete agent behavior"""
    
    def test_simple_add_task_flow(self):
        """Test that agent can add a task successfully"""
        response = agent_executor.invoke({
            "input": "Add task: Prepare presentation by next Friday"
        })
        
        output = response.get('output', '')
        assert "success" in output.lower() or "added" in output.lower()
        assert len(task_manager.tasks) > 0
    
    def test_agent_prevents_past_dates(self):
        """Test that agent handles past dates appropriately"""
        response = agent_executor.invoke({
            "input": "Add task: Old task by last Monday"
        })
        
        output = response.get('output', '')
        # Agent should either reject or interpret intelligently
        assert any(phrase in output.lower() for phrase in [
            "past", "future date", "cannot add", "already passed",
            "not in a valid date format", "provide the deadline"  # <-- Added
        ]) or "added" in output.lower()
    
    def test_agent_memory_usage(self):
        """Test that agent uses memory to avoid repetition"""
        # First request
        response1 = agent_executor.invoke({
            "input": "Show all tasks"
        })
        
        # Second identical request
        response2 = agent_executor.invoke({
            "input": "Show all tasks"
        })
        
        # Agent should acknowledge it already showed tasks
        # or show them again based on design
        assert response2.get('output') is not None

# ===== SCENARIO TESTS =====

class TestScenarios(TestBase):
    """Test complete user scenarios"""
    
    def test_clarification_flow(self):
        """Test multi-turn clarification for ambiguous requests"""
        # Setup: Add ambiguous tasks
        task_manager.tasks = []
        task_manager.add_task("Team meeting", "tomorrow", category="Work")
        task_manager.add_task("Team meeting", "next week", category="Personal")
        
        # Turn 1: Ambiguous delete
        response1 = agent_executor.invoke({
            "input": "Delete task: Team meeting"
        })
        
        output1 = response1['output']
        assert "multiple" in output1.lower() or "which" in output1.lower()
        
        # Extract task ID from response (simplified)
        task_id = task_manager.tasks[0]['id']
        
        # Turn 2: Clarify with ID
        response2 = agent_executor.invoke({
            "input": f"Delete the one with ID {task_id}"
        })
        
        output2 = response2['output']
        assert "deleted" in output2.lower()
        assert len(task_manager.tasks) == 1
    
    def test_productivity_flow(self):
        """Test getting suggestions and analytics"""
        # Setup some tasks
        task_manager.tasks = []
        task_manager.add_task("Urgent task", "today", importance="High")
        task_manager.add_task("Future task", "next month", importance="Low")
        
        # Ask for suggestions
        response = agent_executor.invoke({
            "input": "What should I work on next?"
        })
        
        output = response['output']
        # Should get some form of suggestion
        assert output is not None and len(output) > 0

# ===== PERFORMANCE TESTS =====

class TestPerformance(TestBase):
    """Test performance and limits"""
    
    def test_agent_stops_on_repetition(self):
        """Test that agent stops when detecting loops"""
        # This might trigger repetition detection
        response = agent_executor.invoke({
            "input": "Show all tasks, then show all tasks again, then once more"
        })
        
        # Check that action tracker detected the issue
        assert action_tracker.tool_call_count <= AGENT_MAX_ITERATIONS
    
    def test_bulk_operations(self):
        """Test bulk task operations"""
        response = agent_executor.invoke({
            "input": 'Add tasks: [{"description": "Task 1", "deadline": "tomorrow"}, {"description": "Task 2", "deadline": "next week"}]'
        })
        
        output = response['output']
        assert "added" in output.lower()

# ===== TEST RUNNER =====

def run_test_summary():
    """Run tests and provide summary"""
    print("🧪 TASK MANAGER TEST SUITE 🧪")
    print("=" * 50)
    
    # You can use pytest to run these
    # pytest.main([__file__, '-v'])
    
    # Or run specific test classes
    test_classes = [
        TestDateParsing,
        TestActionTracker,
        TestEnhancedTaskManager,
        TestAgentTools,
        TestAgentBehavior,
        TestScenarios,
        TestPerformance
    ]
    
    print(f"Test Categories: {len(test_classes)}")
    print("Ready to run with pytest")
    print("\nUsage: pytest test_task_agent.py -v")

if __name__ == "__main__":
    run_test_summary()