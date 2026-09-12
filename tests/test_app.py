import sys
import os
import json
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app import route_inference, route_inference_stream, async_sync_firestore, FIRESTORE_COLLECTION

class TestObsygnalInference(unittest.TestCase):
    @patch("requests.post")
    def test_route_inference_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps({"response": "Obsygnal operational "}),
            json.dumps({"response": "response from Qwen 2.5."})
        ]
        mock_post.return_value = mock_response

        result = route_inference("Hello Obsygnal AI")
        self.assertEqual(result, "Obsygnal operational response from Qwen 2.5.")
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "qwen2.5:1.5b")
        self.assertEqual(kwargs["json"]["prompt"], "Hello Obsygnal AI")
        self.assertTrue(kwargs["json"]["stream"])
        self.assertIn("options", kwargs["json"])
        self.assertEqual(kwargs["json"]["options"]["num_thread"], 2)
        self.assertEqual(kwargs["json"]["options"]["num_ctx"], 1024)

    @patch("requests.post")
    def test_route_inference_stream_generator(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps({"response": "Token1 "}),
            json.dumps({"response": "Token2"})
        ]
        mock_post.return_value = mock_response

        tokens = list(route_inference_stream("Stream prompt"))
        self.assertEqual(tokens, ["Token1 ", "Token2"])

    @patch("requests.post")
    def test_route_inference_normalization(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps({"response": "Normalized route response."})
        ]
        mock_post.return_value = mock_response

        result = route_inference("Ping", endpoint="http://127.0.0")
        self.assertEqual(result, "Normalized route response.")
        target_url, _ = mock_post.call_args[0][0], mock_post.call_args[1]
        self.assertEqual(target_url, "http://127.0.0:11434/api/generate")

    @patch("requests.post")
    def test_route_inference_failure_handling(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection to 127.0.0.1:11434 refused")

        result = route_inference("Test fail")
        self.assertIn("Inference engine routing failure", result)
        self.assertIn("Connection to 127.0.0.1:11434 refused", result)

    @patch("src.app.get_firestore_client")
    def test_firestore_envelope_ttl(self, mock_get_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_doc = MagicMock()

        mock_get_client.return_value = mock_client
        mock_client.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_doc

        test_user = "test-user-12345"
        test_role = "user"
        test_content = "Store this prompt"

        # Execute async write task synchronously inside mock
        with patch("src.app.db_executor.submit", side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)):
            async_sync_firestore(test_user, test_role, test_content)

        mock_client.collection.assert_called_with(FIRESTORE_COLLECTION)
        mock_doc.set.assert_called_once()
        written_data = mock_doc.set.call_args[0][0]

        self.assertEqual(written_data["user_id"], test_user)
        self.assertEqual(written_data["role"], test_role)
        self.assertEqual(written_data["content"], test_content)
        self.assertIn("updated_at", written_data)
        self.assertIn("expire_at", written_data)

        # Validate 7-day TTL delta
        delta = written_data["expire_at"] - written_data["updated_at"]
        self.assertAlmostEqual(delta.total_seconds(), timedelta(days=7).total_seconds(), delta=5)

if __name__ == "__main__":
    unittest.main()
