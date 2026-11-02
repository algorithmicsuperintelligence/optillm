"""Offline provider tests through Flask and the real SDK's HTTP serialization."""
import json
import os
import sys
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optillm import server
from optillm.plugins.proxy.client import ProxyClient, list_provider_models


class TestMiniMaxProvider(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'MINIMAX_API_KEY': 'fixture-key'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = patch.dict(server.server_config, {
            'base_url': '', 'optillm_api_key': '', 'approach': 'none',
            'model': server.default_model(), 'ssl_verify': True, 'ssl_cert_path': '',
        })
        self.config.start()
        self.addCleanup(self.config.stop)
        self.requests = []
        self.transport = httpx.MockTransport(self.respond)
        self.http = httpx.Client(transport=self.transport)
        self.addCleanup(self.http.close)
        self.http_patch = patch('httpx.HTTPTransport.handle_request', side_effect=self.respond)
        self.http_patch.start()
        self.addCleanup(self.http_patch.stop)
        self.models_patch = patch('optillm.plugins.proxy.client.list_provider_models', return_value=None)
        self.models_patch.start()
        self.addCleanup(self.models_patch.stop)
        self.web = server.app.test_client()

    def respond(self, request):
        self.requests.append(request)
        if request.method == 'GET':
            models = ['MiniMax-M2.7', 'MiniMax-M3', 'speech-2.8-hd', 'custom-model']
            if request.url.host == 'openrouter.ai':
                models = ['minimax/minimax-m2.7', 'minimax/minimax-m3',
                          'minimax/minimax-m3.1-flash:free', 'other/model']
            return httpx.Response(200, json={'object': 'list', 'data': [
                {'id': model, 'object': 'model', 'created': 0, 'owned_by': 'fixture',
                 'context_length': 12345, 'pricing': {'prompt': 'fixture-price'}}
                for model in models
            ]})
        body = json.loads(request.content)
        return httpx.Response(200, json={
            'id': 'chatcmpl-fixture', 'object': 'chat.completion', 'created': 0,
            'model': body['model'],
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        })

    def post(self, payload=None, path='/v1/chat/completions', headers=None):
        data = {'messages': [{'role': 'user', 'content': 'Hello'}]}
        data.update(payload or {})
        response = self.web.post(path, json=data, headers=headers or {})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return json.loads(self.requests[-1].content)

    def test_startup_default_without_model_override(self):
        code = """
import json
from unittest.mock import patch
import httpx
from optillm import server
requests = []
def respond(request):
    requests.append(json.loads(request.content))
    return httpx.Response(200, json={'id': 'fixture', 'model': requests[-1]['model'],
        'object': 'chat.completion', 'created': 0, 'choices': [
            {'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})
with patch('httpx.HTTPTransport.handle_request', side_effect=respond):
    response = server.app.test_client().post('/v1/chat/completions',
        json={'messages': [{'role': 'user', 'content': 'Hello'}]})
assert response.status_code == 200, response.get_data(as_text=True)
assert requests[-1]['model'] == 'MiniMax-M3.1-Flash', requests
"""
        result = subprocess.run([sys.executable, '-c', code], env={'MINIMAX_API_KEY': 'fixture-key'},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_regions_and_catalog(self):
        for region, host in [('global', 'api.minimax.io'), ('cn', 'api.minimaxi.com')]:
            with self.subTest(region=region), patch.dict(os.environ, {'MINIMAX_API_REGION': region}):
                body = self.post()
                self.assertEqual(body['model'], 'MiniMax-M3.1-Flash')
                self.assertEqual(self.requests[-1].url.host, host)
                self.assertEqual(self.requests[-1].url.path, '/v1/chat/completions')
                self.assertEqual(self.requests[-1].headers['authorization'], 'Bearer fixture-key')
                for key in ('thinking', 'reasoning_effort', 'service_tier', 'reasoning_split'):
                    self.assertNotIn(key, body)
                models = self.web.get('/v1/models').get_json()['data']
                self.assertEqual([m['id'] for m in models], list(server.MINIMAX_MODELS))

    def test_cli_and_saved_model_precedence(self):
        for argv, env, expected in [
            (['optillm'], {}, 'MiniMax-M3.1-Flash'),
            (['optillm'], {'OPTILLM_MODEL': 'MiniMax-M2.7'}, 'MiniMax-M2.7'),
            (['optillm', '--model', 'custom-model'], {'OPTILLM_MODEL': 'MiniMax-M3'}, 'custom-model'),
        ]:
            with self.subTest(expected=expected), patch.dict(os.environ, env), patch.object(sys, 'argv', argv):
                args = server.parse_args()
                server.server_config['model'] = args.model
                self.assertEqual(self.post()['model'], expected)

    def test_explicit_models_and_temperature_are_preserved(self):
        for model in ('MiniMax-M3', 'MiniMax-M2.7', 'MiniMax-M2.7-highspeed', 'custom-model', 'none-MiniMax-M3.1-Flash'):
            for temperature in (0, 1.5, 2):
                with self.subTest(model=model, temperature=temperature):
                    body = self.post({'model': model, 'temperature': temperature})
                    self.assertEqual(body['model'], model.removeprefix('none-'))
                    self.assertEqual(body['temperature'], temperature)

    def test_custom_endpoint_is_not_given_direct_api_rules(self):
        server.server_config['base_url'] = 'https://custom.example/v1'
        body = self.post({'model': 'custom-model', 'temperature': 0, 'service_tier': 'priority',
                          'extra_body': {'thinking': {'type': 'disabled'}}})
        self.assertEqual(self.requests[-1].url.host, 'custom.example')
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(body['temperature'], 0)
        self.assertEqual(body['model'], 'custom-model')
        models = self.web.get('/v1/models').get_json()['data']
        self.assertIn('MiniMax-M2.7', [m['id'] for m in models])

    def test_openrouter_catalog_and_saved_request_keep_gateway_contract(self):
        server.server_config['base_url'] = 'https://openrouter.ai/api/v1'
        models = self.web.get('/v1/models').get_json()['data']
        self.assertEqual([m['id'] for m in models], [
            'minimax/minimax-m3.1-flash:free', 'minimax/minimax-m3', 'other/model'])
        self.assertEqual(models[0]['context_length'], 12345)
        self.assertEqual(models[0]['pricing'], {'prompt': 'fixture-price'})
        for model in ('minimax/minimax-m2.7', 'minimax/minimax-m3.1-flash:free'):
            body = self.post({'model': model, 'extra_body': {'reasoning': {'effort': 'high'}}})
            self.assertEqual(body['model'], model)
            self.assertEqual(body['reasoning'], {'effort': 'high'})
            self.assertNotIn('thinking', body)
            self.assertEqual(self.requests[-1].url.path, '/api/v1/chat/completions')

    def test_effort_thinking_and_token_limit_serialization(self):
        for effort in ('low', 'medium', 'high', 'xhigh', 'max'):
            with self.subTest(effort=effort):
                body = self.post({'reasoning_effort': effort, 'thinking': {'type': 'adaptive'},
                                  'reasoning_split': True, 'max_completion_tokens': 524288})
                self.assertEqual(body['reasoning_effort'], effort)
                self.assertEqual(body['thinking'], {'type': 'adaptive'})
                self.assertTrue(body['reasoning_split'])
                self.assertEqual(body['max_completion_tokens'], 524288)
                self.assertNotIn('extra_body', body)
        body = self.post({'model': 'MiniMax-M3', 'thinking': {'type': 'disabled'},
                          'reasoning_effort': 'high', 'service_tier': 'priority', 'reasoning_split': False})
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(body['service_tier'], 'priority')
        self.assertFalse(body['reasoning_split'])

    def test_flash_rejects_unsupported_options_without_sending(self):
        for options in ({'thinking': {'type': 'disabled'}}, {'reasoning_effort': 'none'},
                        {'service_tier': 'standard'}, {'reasoning_split': False},
                        {'extra_body': {'thinking': {'type': 'disabled'}}},
                        {'extra_body': {'service_tier': 'priority'}},
                        {'extra_body': {'reasoning_effort': 'none'}}):
            with self.subTest(options=options):
                self.requests.clear()
                response = self.web.post('/v1/chat/completions', json={
                    'messages': [{'role': 'user', 'content': 'Hello'}], **options})
                self.assertEqual(response.status_code, 500)
                self.assertIn('MiniMax-M3.1-Flash', response.get_json()['error'])
                self.assertEqual(self.requests, [])

    def test_multimodal_and_tool_history_survive_transport(self):
        messages = [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Describe'},
                {'type': 'image_url', 'image_url': {'url': 'https://example.test/image.png'}},
                {'type': 'video_url', 'video_url': {'url': 'mm_file://fixture'}},
            ]},
            {'role': 'assistant', 'content': '<think>keep</think>', 'reasoning_content': 'keep too',
             'tool_calls': [{'id': 'call_fixture', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_fixture', 'content': 'found'},
        ]
        body = self.post({'messages': messages})
        self.assertEqual(body['messages'], messages)
        server.server_config['base_url'] = 'https://openrouter.ai/api/v1'
        body = self.post({'model': 'minimax/minimax-m3.1-flash:free', 'messages': messages})
        self.assertEqual(body['messages'], messages)

    def test_legacy_completion_and_streaming_routes(self):
        body = self.post({'prompt': 'Hello'}, path='/v1/completions')
        self.assertEqual(body['model'], 'MiniMax-M3.1-Flash')
        response = self.web.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'[DONE]', response.data)
        self.assertNotIn('stream', json.loads(self.requests[-1].content))

    def test_higher_priority_provider_and_bearer_override(self):
        with patch.dict(os.environ, {'CEREBRAS_API_KEY': 'fixture-other'}):
            with patch.object(server, 'Cerebras', return_value=OpenAI(
                    api_key='fixture-other', base_url='https://other.example/v1', http_client=self.http)):
                self.assertEqual(server.default_model(), 'gpt-4o-mini')
                body = self.post({'model': 'other-model', 'temperature': 0})
                self.assertEqual(body['temperature'], 0)
                self.assertEqual(self.requests[-1].url.host, 'other.example')
        with patch.dict(os.environ, {'OPTILLM_API_KEY': 'fixture-local'}), patch.dict(
                sys.modules, {'optillm.inference': MagicMock()}):
            self.assertEqual(server.default_model(), 'gpt-4o-mini')
            self.assertEqual(server.get_config()[1], 'fixture-local')
        self.post({'model': 'other-model', 'temperature': 0}, headers={'Authorization': 'Bearer sk-fixture'})
        self.assertEqual(self.requests[-1].url.host, 'api.openai.com')

    def test_lower_priority_provider_keys(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'fixture-other', 'AZURE_OPENAI_API_KEY': 'fixture-azure'}):
            self.post()
            self.assertEqual(self.requests[-1].url.host, 'api.minimax.io')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'fixture-other'}, clear=True):
            self.assertEqual(server.default_model(), 'gpt-4o-mini')
            self.post({'model': 'other-model'})
            self.assertEqual(self.requests[-1].url.host, 'api.openai.com')

    def test_proxy_catalog_preserves_aliases_and_nontext(self):
        config = {'providers': [{'name': 'minimax', 'base_url': 'https://api.minimax.io/v1',
                                'api_key': 'fixture-key', 'model_map': {'saved': 'MiniMax-M2.7'}}]}
        with patch('optillm.plugins.proxy.client.OpenAI', side_effect=lambda **kwargs: OpenAI(http_client=self.http, **kwargs)):
            models = list_provider_models(config)
        ids = [model['id'] for model in models]
        self.assertEqual(ids[:2], list(server.MINIMAX_MODELS))
        self.assertNotIn('MiniMax-M2.7', ids)
        self.assertIn('speech-2.8-hd', ids)
        self.assertIn('custom-model', ids)
        self.assertIn('saved', ids)

    def test_proxy_fallback_serializes_direct_options(self):
        from optillm.plugins import proxy_plugin
        direct = OpenAI(api_key='fixture-key', base_url='https://api.minimax.io/v1', http_client=self.http)
        with patch.object(proxy_plugin.ProxyConfig, 'load', return_value={'providers': []}):
            proxy_plugin.run('', 'Hello', direct, 'MiniMax-M3.1-Flash',
                             request_config={'thinking': {'type': 'adaptive'}, 'reasoning_effort': 'max'})
        body = json.loads(self.requests[-1].content)
        self.assertEqual(body['thinking'], {'type': 'adaptive'})
        self.assertEqual(body['reasoning_effort'], 'max')

    def test_proxy_transport_preserves_explicit_model_map(self):
        config = {'providers': [{'name': 'minimax', 'base_url': 'https://api.minimaxi.com/v1',
                                'api_key': 'fixture-key', 'model_map': {'saved': 'MiniMax-M3'}}],
                  'routing': {'health_check': {'enabled': False}}}
        with patch('optillm.plugins.proxy.client.OpenAI', side_effect=lambda **kwargs: OpenAI(http_client=self.http, **kwargs)):
            client = ProxyClient(config)
            client.chat.completions.create(model='saved', messages=[{'role': 'user', 'content': 'Hi'}],
                                           thinking={'type': 'disabled'})
        body = json.loads(self.requests[-1].content)
        self.assertEqual(body['model'], 'MiniMax-M3')
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(self.requests[-1].url.host, 'api.minimaxi.com')


if __name__ == '__main__':
    unittest.main()
