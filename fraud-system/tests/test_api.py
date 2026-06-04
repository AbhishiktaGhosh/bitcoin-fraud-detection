import os
import sys
import unittest
from fastapi.testclient import TestClient

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from api.main import app

class TestFraudDetectionAPI(unittest.TestCase):
    def setUp(self):
        self.client_ctx = TestClient(app)
        self.client = self.client_ctx.__enter__()
        
    def tearDown(self):
        self.client_ctx.__exit__(None, None, None)
        
    def test_health_endpoint(self):
        response = self.client.get("/health")
        # Since it runs on startup, it might be 200 or 503 if graph is not loaded,
        # but the JSON response must have a 'status' key.
        self.assertIn(response.status_code, [200, 503])
        data = response.json()
        self.assertIn("status", data)
        
    def test_model_info_endpoint(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gnn_models", data)
        self.assertIn("p99_latency_sla", data)
        
    def test_predict_endpoint_missing_payload(self):
        # Empty payload should return 400
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)
        
    def test_predict_custom_input_validation(self):
        # Custom features request
        payload = {
            "features": {f"trans_feat_{i}": 0.1 for i in range(93)}
        }
        # Missing scaler will return 400 or 200 depending on training state
        response = self.client.post("/predict", json=payload)
        self.assertIn(response.status_code, [200, 400])

if __name__ == "__main__":
    unittest.main()
