from flask import jsonify
import random


def recommend_handler(app, recommendation_app):
    @app.route('/recommend/<user_id>', methods=['GET'])
    def recommend(user_id):
        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({"error": "Invalid user_id. It should be an integer."}), 400

        recommendations = recommendation_app.recommend_items(user_id)

        if isinstance(recommendations, tuple):
            return jsonify(recommendations[0]), recommendations[1]

        user_name = recommendation_app.user_name_mapping.get(user_id, "Unknown User")
        recommended_item_names = [
            {"item_id": item_id, "item_name": recommendation_app.item_name_mapping.get(item_id, "Unknown Item")}
            for item_id in recommendations
        ]

        return jsonify({
            "user_id": user_id,
            "user_name": user_name,
            'recommendations': recommendations,
            "recommendations_ext": recommended_item_names
        })


def user_sample_handler(app, recommendation_app):
    @app.route('/user_sample', methods=['GET'])
    def get_user_sample():
        try:
            sample_size = 10
            user_sample = random.sample(list(recommendation_app.user_name_mapping.items()), sample_size)
            return jsonify(user_sample)
        except Exception as e:
            print(f"Error retrieving user sample: {e}")
            return {"error": "An error occurred while retrieving the user sample."}, 500
