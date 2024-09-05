import unittest
from flask import Flask
from app import RecommendationApp


class TestRecommendationApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the Flask app and test client
        cls.app = RecommendationApp().app
        cls.client = cls.app.test_client()

    def test_recommend_known_user(self):
        # Test recommendation for a known user
        user_id = 90341
        response = self.client.get(f'/recommend/{user_id}')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("recommendations", data)
        self.assertIsInstance(data['recommendations'], list)
        self.assertGreater(len(data['recommendations']), 0)

    def test_recommend_unknown_user(self):
        # Test recommendation for an unknown user (user_id not in the dataset)
        user_id = 999999
        response = self.client.get(f'/recommend/{user_id}')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("recommendations", data)
        self.assertGreater(len(data['recommendations']), 0)

    def test_invalid_user_id(self):
        # Test invalid user_id (non-integer)
        response = self.client.get('/recommend/abc')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Invalid user_id. It should be an integer.")

    # def test_recommendation_with_validations(self):
    #     # Test valid user_id but invalid num_recommendations (exceeds limit)
    #     response = self.client.get('/recommend/123?num_recommendations=200')
    #     self.assertEqual(response.status_code, 400)  # Should fail due to num_recommendations > 100
    #     data = response.get_json()
    #     self.assertIn("error", data)
    #     self.assertIn("details", data)

    def test_get_user_sample(self):
        # Test user sample endpoint
        response = self.client.get('/user_sample')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)


if __name__ == '__main__':
    unittest.main()
