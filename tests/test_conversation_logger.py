import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.append('..')
from optillm.conversation_logger import ConversationLogger


class TestConversationLogger(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.logger_enabled = ConversationLogger(self.temp_dir, enabled=True)
        self.logger_disabled = ConversationLogger(self.temp_dir, enabled=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temp directory
        for file in self.temp_dir.glob("*"):
            file.unlink()
        self.temp_dir.rmdir()
    
    def test_logger_initialization_and_disabled_state(self):
        """Test logger initialization and disabled logger behavior"""
        # Test enabled logger
        self.assertTrue(self.logger_enabled.enabled)
        self.assertEqual(self.logger_enabled.log_dir, self.temp_dir)
        self.assertTrue(self.temp_dir.exists())
        
        # Test disabled logger
        self.assertFalse(self.logger_disabled.enabled)
        
        # Disabled logger should return empty string and perform no operations
        request_id = self.logger_disabled.start_conversation({}, "test", "model")
        self.assertEqual(request_id, "")
        
        # Other methods should not raise errors but do nothing
        self.logger_disabled.log_provider_call("req1", {}, {})
        self.logger_disabled.log_final_response("req1", {})
        self.logger_disabled.log_error("req1", "error")
        self.logger_disabled.finalize_conversation("req1")
    
    def test_conversation_lifecycle(self):
        """Test complete conversation lifecycle: start, log calls, errors, finalize"""
        client_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
            "temperature": 0.7
        }
        
        # Start conversation
        request_id = self.logger_enabled.start_conversation(
            client_request=client_request,
            approach="moa",
            model="gpt-4o-mini"
        )
        
        # Should return a valid request ID
        self.assertIsInstance(request_id, str)
        self.assertTrue(request_id.startswith("req_"))
        self.assertEqual(len(request_id), 12)  # "req_" + 8 hex chars
        
        # Should create an active entry
        self.assertIn(request_id, self.logger_enabled.active_entries)
        entry = self.logger_enabled.active_entries[request_id]
        self.assertEqual(entry.request_id, request_id)
        self.assertEqual(entry.approach, "moa")
        self.assertEqual(entry.model, "gpt-4o-mini")
        
        # Log multiple provider calls
        provider_request = {"model": "test", "messages": []}
        provider_response = {"choices": [{"message": {"content": "response"}}]}
        
        self.logger_enabled.log_provider_call(request_id, provider_request, provider_response)
        self.logger_enabled.log_provider_call(request_id, provider_request, provider_response)
        
        # Check calls were logged
        entry = self.logger_enabled.active_entries[request_id]
        self.assertEqual(len(entry.provider_calls), 2)
        self.assertEqual(entry.provider_calls[0]["call_number"], 1)
        self.assertEqual(entry.provider_calls[1]["call_number"], 2)
        
        # Log final response
        final_response = {"choices": [{"message": {"content": "final"}}]}
        self.logger_enabled.log_final_response(request_id, final_response)
        
        # Log error
        error_msg = "Test error message"
        self.logger_enabled.log_error(request_id, error_msg)
        
        # Check entries were updated
        entry = self.logger_enabled.active_entries[request_id]
        self.assertEqual(entry.error, error_msg)
        
        # Finalize the conversation
        self.logger_enabled.finalize_conversation(request_id)
        
        # Should no longer be in active entries
        self.assertNotIn(request_id, self.logger_enabled.active_entries)
        
        # Should have written to file
        log_files = list(self.temp_dir.glob("conversations_*.jsonl"))
        self.assertEqual(len(log_files), 1)
        
        # Read and verify log content
        with open(log_files[0], 'r', encoding='utf-8') as f:
            log_line = f.read().strip()
        
        log_entry = json.loads(log_line)
        
        # Verify structure
        self.assertEqual(log_entry["request_id"], request_id)
        self.assertEqual(log_entry["approach"], "moa")
        self.assertEqual(log_entry["model"], "gpt-4o-mini")
        self.assertEqual(log_entry["client_request"], client_request)
        self.assertEqual(len(log_entry["provider_calls"]), 2)
        self.assertEqual(log_entry["final_response"]["choices"][0]["message"]["content"], "final")
        self.assertIsInstance(log_entry["total_duration_ms"], int)
        self.assertEqual(log_entry["error"], error_msg)
    
    def test_multiple_conversations_and_log_files(self):
        """Test handling multiple concurrent conversations and log file naming"""
        with patch('optillm.conversation_logger.datetime') as mock_datetime:
            # Mock datetime.now to return a specific date
            mock_datetime.now.return_value = datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # Test log file naming
            log_path = self.logger_enabled._get_log_file_path()
            expected_filename = "conversations_2025-01-27.jsonl"
            self.assertEqual(log_path.name, expected_filename)
            self.assertEqual(log_path.parent, self.temp_dir)
        
        # Start multiple conversations
        request_id1 = self.logger_enabled.start_conversation({}, "moa", "model1")
        request_id2 = self.logger_enabled.start_conversation({}, "none", "model2")
        
        # Should be different IDs
        self.assertNotEqual(request_id1, request_id2)
        
        # Both should be active
        self.assertIn(request_id1, self.logger_enabled.active_entries)
        self.assertIn(request_id2, self.logger_enabled.active_entries)
        
        # Log different data to each
        self.logger_enabled.log_provider_call(request_id1, {"req": "1"}, {"resp": "1"})
        self.logger_enabled.log_provider_call(request_id2, {"req": "2"}, {"resp": "2"})
        
        # Finalize both
        self.logger_enabled.finalize_conversation(request_id1)
        self.logger_enabled.finalize_conversation(request_id2)
        
        # Should have 2 log entries in the file
        log_files = list(self.temp_dir.glob("conversations_*.jsonl"))
        self.assertEqual(len(log_files), 1)
        
        with open(log_files[0], 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        
        self.assertEqual(len(lines), 2)
        
        # Verify both entries
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])
        
        self.assertEqual(entry1["approach"], "moa")
        self.assertEqual(entry2["approach"], "none")
    
    def test_invalid_request_id_and_stats(self):
        """Test handling of invalid request IDs and logger statistics"""
        # Invalid request IDs should not raise errors but do nothing
        self.logger_enabled.log_provider_call("invalid_id", {}, {})
        self.logger_enabled.log_final_response("invalid_id", {})
        self.logger_enabled.log_error("invalid_id", "error")
        self.logger_enabled.finalize_conversation("invalid_id")
        
        # Test disabled logger stats
        stats = self.logger_disabled.get_stats()
        expected_disabled_stats = {
            "enabled": False,
            "log_dir": str(self.temp_dir),
            "active_conversations": 0
        }
        self.assertEqual(stats, expected_disabled_stats)
        
        # Test enabled logger stats with active conversations
        request_id1 = self.logger_enabled.start_conversation({}, "test", "model")
        self.logger_enabled.start_conversation({}, "test", "model")
        
        stats = self.logger_enabled.get_stats()
        
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["log_dir"], str(self.temp_dir))
        self.assertEqual(stats["active_conversations"], 2)
        self.assertEqual(stats["log_files_count"], 0)  # No finalized conversations yet
        self.assertEqual(stats["total_entries_approximate"], 0)
        
        # Finalize one and check stats again
        self.logger_enabled.finalize_conversation(request_id1)
        stats = self.logger_enabled.get_stats()
        
        self.assertEqual(stats["active_conversations"], 1)
        self.assertEqual(stats["log_files_count"], 1)
        self.assertEqual(stats["total_entries_approximate"], 1)


if __name__ == '__main__':
    unittest.main()