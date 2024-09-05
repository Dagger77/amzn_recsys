from flask import Flask, request, jsonify
import numpy as np
import pickle
import random


class RecommendationApp:
    def __init__(self):
        # Load pre-trained LightFM model, mappings, known interactions and top items
        self.lfm_model = pickle.load(open('model\\lightfm_model.pkl', 'rb'))

        self.user_id_mapping = pickle.load(open('model\\model_user_id_map.pkl', 'rb'))
        self.item_id_mapping = pickle.load(open('model\\model_item_id_map.pkl', 'rb'))
        self.item_id_inverse_mapping = {v: k for k, v in self.item_id_mapping.items()}

        self.user_name_mapping = pickle.load(open('model\\user_to_name_map.pkl', 'rb'))
        self.item_name_mapping = pickle.load(open('model\\item_to_name_map.pkl', 'rb'))

        self.known_interactions = pickle.load(open('model\\user_item_interactions.pkl', 'rb'))
        self.top_items = pickle.load(open('model\\top_20_items.pkl', 'rb'))

        # Initialize Flask app
        self.app = Flask(__name__)
        self.add_routes()

    def recommend_items(self, user_id, num_recommendations=5):
        try:
            user_id = int(user_id)
            model_user_id = self.user_id_mapping.get(user_id, None)

            if model_user_id is None:
                # If the user is unknown, return the precomputed top global items
                print(f"Unknown user {user_id}. Returning top global items.")
                return self.top_items[:num_recommendations]

            n_items = len(self.item_id_mapping)
            scores = self.lfm_model.predict(model_user_id, np.arange(n_items))

            # Excluding items that users already interacted with
            known_items = self.known_interactions.get(user_id, set())
            known_item_indices = np.array(
                [self.item_id_mapping[item] for item in known_items if item in self.item_id_mapping])
            mask = np.ones(n_items, dtype=bool)
            mask[known_item_indices] = False

            filtered_scores = scores[mask]
            filtered_item_indices = np.arange(n_items)[mask]

            top_filtered_indices = np.argsort(-filtered_scores)[:num_recommendations]
            top_item_ids = [self.item_id_inverse_mapping[filtered_item_indices[i]] for i in top_filtered_indices]

            return top_item_ids
        except Exception as e:
            print(f"Error during recommendation for user {user_id}: {e}")
            return []

    def add_routes(self):
        # Define the recommendation route
        @self.app.route('/recommend/<user_id>', methods=['GET'])
        def recommend(user_id):
            recommendations = self.recommend_items(user_id)

            user_name = self.user_name_mapping.get(user_id, "Unknown User")

            recommended_item_names = [
                {"item_id": item_id, "item_name": self.item_name_mapping.get(item_id, "Unknown Item")}
                for item_id in recommendations
            ]

            return jsonify({
                "user_id": user_id,
                "user_name": user_name,
                'recommendations': recommendations,
                "recommendations_ext": recommended_item_names
            })

        # Define a route to get a sample of model's user_id: userName pairs
        @self.app.route('/user_sample', methods=['GET'])
        def get_user_sample():
            # Get a sample of user_id: userName pairs
            sample_size = 10
            user_sample = random.sample(list(self.user_name_mapping.items()), sample_size)
            return jsonify(user_sample)

    def run(self):
        self.app.run(debug=True)


if __name__ == '__main__':
    recommendation_app = RecommendationApp()
    recommendation_app.run()
