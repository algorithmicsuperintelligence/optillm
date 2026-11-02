"""
Offline unit tests for `developer` role messages.

OpenAI clients send `developer` messages in place of `system` messages for
o-series and newer models. optillm used to drop them when it split a request
into system prompt and conversation, so the instructions never reached the
approach. Uses the Flask test client with a mocked approach (no network, no model).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from optillm import server


CAPTURED = {}


def fake_chat_with_mcts(system_prompt, initial_query, client, model,
                        num_simulations, exploration_weight, depth,
                        request_config=None, request_id=None):
    CAPTURED['system_prompt'] = system_prompt
    CAPTURED['initial_query'] = initial_query
    return "ok", 1


class TestParseConversationDeveloperRole(unittest.TestCase):

    def test_developer_message_becomes_system_prompt(self):
        messages = [
            {"role": "developer", "content": "Answer in French."},
            {"role": "user", "content": "Hello"},
        ]
        system_prompt, initial_query, approach = server.parse_conversation(messages)
        self.assertEqual(system_prompt, "Answer in French.")
        self.assertEqual(initial_query, "User: Hello")
        self.assertIsNone(approach)

    def test_developer_message_with_content_parts(self):
        messages = [
            {"role": "developer", "content": [{"type": "text", "text": "Answer in French."}]},
            {"role": "user", "content": "Hello"},
        ]
        system_prompt, _, _ = server.parse_conversation(messages)
        self.assertEqual(system_prompt, "Answer in French.")

    def test_developer_message_can_select_approach(self):
        messages = [
            {"role": "developer", "content": "<optillm_approach>re2</optillm_approach> Be brief."},
            {"role": "user", "content": "Hello"},
        ]
        system_prompt, _, approach = server.parse_conversation(messages)
        self.assertEqual(system_prompt, "Be brief.")
        self.assertEqual(approach, "re2")

    def test_system_message_unchanged(self):
        messages = [
            {"role": "system", "content": "Answer in French."},
            {"role": "user", "content": "Hello"},
        ]
        system_prompt, initial_query, _ = server.parse_conversation(messages)
        self.assertEqual(system_prompt, "Answer in French.")
        self.assertEqual(initial_query, "User: Hello")


class TestDeveloperRoleReachesApproach(unittest.TestCase):

    def setUp(self):
        CAPTURED.clear()
        self._originals = {
            'get_config': server.get_config,
            'chat_with_mcts': server.chat_with_mcts,
            'config': dict(server.server_config),
        }
        server.get_config = lambda: (object(), 'optillm')
        server.chat_with_mcts = fake_chat_with_mcts
        server.server_config['optillm_api_key'] = ''
        server.app.config['TESTING'] = True
        self.client = server.app.test_client()

    def tearDown(self):
        server.get_config = self._originals['get_config']
        server.chat_with_mcts = self._originals['chat_with_mcts']
        server.server_config.clear()
        server.server_config.update(self._originals['config'])

    def test_developer_message_is_passed_to_approach(self):
        resp = self.client.post(
            '/v1/chat/completions',
            json={"model": "gpt-4o-mini",
                  "messages": [
                      {"role": "developer", "content": "Answer in French."},
                      {"role": "user", "content": "Hello"},
                  ],
                  "optillm_approach": "mcts"})

        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        self.assertEqual(CAPTURED['system_prompt'], "Answer in French.")
        self.assertEqual(CAPTURED['initial_query'], "User: Hello")


if __name__ == '__main__':
    unittest.main()
