from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls.tmpdir.name) / "test.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"

        from fastapi.testclient import TestClient
        from app.main import app

        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.tmpdir.cleanup()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["data_mode"], "demo")

    def test_overview(self):
        response = self.client.get("/api/v1/metrics/overview")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["households"], 1500)
        self.assertEqual(body["adults"], 1500)
        self.assertEqual(body["matched_pairs"], 1500)
        self.assertGreaterEqual(body["adoption_rate"], 0)
        self.assertLessEqual(body["adoption_rate"], 1)

    def test_internet_groups(self):
        response = self.client.get("/api/v1/metrics/groups", params={"dimension": "internet"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["dimension"], "internet")
        self.assertEqual(len(body["groups"]), 2)
        self.assertEqual(sum(g["n"] for g in body["groups"]), 1500)

    def test_income_quartiles(self):
        response = self.client.get("/api/v1/metrics/groups", params={"dimension": "income_quartile"})
        self.assertEqual(response.status_code, 200)
        groups = response.json()["groups"]
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(0 <= g["adoption_rate"] <= 1 for g in groups))

    def test_validation(self):
        response = self.client.get("/api/v1/validation")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "pass")
        self.assertGreaterEqual(len(body["checks"]), 5)
        self.assertTrue(all(c["status"] == "pass" for c in body["checks"]))

    def test_household_detail(self):
        response = self.client.get("/api/v1/households/100001")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["household"]["fid22"], 100001)
        self.assertIn("adult", body)

    def test_household_not_found(self):
        response = self.client.get("/api/v1/households/999999")
        self.assertEqual(response.status_code, 404)

    def test_invalid_dimension(self):
        response = self.client.get("/api/v1/metrics/groups", params={"dimension": "unknown"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
