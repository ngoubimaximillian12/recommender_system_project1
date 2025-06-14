import pandas as pd

from recommender_system_project.models.neural_network import NeuralCollaborativeFiltering

import os
def main():
    print("Starting NCF recommender system demonstration...")

    # 1. Create dummy data
    # In a real scenario, you would load your MovieLens or Amazon dataset here
    data = {
        'userId': [1, 1, 2, 2, 3, 3, 1, 2, 3],
        'movieId': [101, 102, 101, 103, 102, 103, 103, 102, 101],
        'rating': [5, 3, 4, 5, 2, 4, 3, 5, 4]
    }
    ratings_df = pd.DataFrame(data)
    print("\n--- Raw Ratings Data ---")
    print(ratings_df)

    # 2. Initialize the NCF model
    ncf_recommender = NeuralCollaborativeFiltering(model_type='ncf', embedding_dim=16, hidden_layers=[32, 16])

    # 3. Train the model
    print("\n--- Training NCF Model ---")
    # Use a small number of epochs for demonstration purposes
    ncf_recommender.train(ratings_df, epochs=5, batch_size=2, validation_split=0.2)

    # 4. Make predictions
    print("\n--- Making Predictions ---")
    # Example: Predict rating for user 1 on movie 103
    # Ensure user_ids and item_ids are in the original format, not encoded
    user_to_predict = [1, 2]
    item_to_predict = [103, 101]
    predicted_ratings = ncf_recommender.predict(user_to_predict, item_to_predict)
    print(f"Predicted normalized rating for User {user_to_predict[0]} on Movie {item_to_predict[0]}: {predicted_ratings[0]:.4f}")
    print(f"Predicted normalized rating for User {user_to_predict[1]} on Movie {item_to_predict[1]}: {predicted_ratings[1]:.4f}")

    # You can convert normalized ratings back to original scale if max_rating and min_rating are known
    min_rating = 1
    max_rating = 5
    original_scale_pred_1 = predicted_ratings[0] * (max_rating - min_rating) + min_rating
    original_scale_pred_2 = predicted_ratings[1] * (max_rating - min_rating) + min_rating
    print(f"Predicted original scale rating for User {user_to_predict[0]} on Movie {item_to_predict[0]}: {original_scale_pred_1:.2f}")
    print(f"Predicted original scale rating for User {user_to_predict[1]} on Movie {item_to_predict[1]}: {original_scale_pred_2:.2f}")

    # 5. Save and Load Model
    print("\n--- Saving and Loading Model ---")
    model_save_path = 'saved_models/my_ncf_model'
    ncf_recommender.save_model(path=model_save_path)

    # Create a new instance and load the model
    loaded_recommender = NeuralCollaborativeFiltering()
    loaded_recommender.load_model(path=model_save_path)

    # Verify predictions after loading
    print("\n--- Verifying Predictions with Loaded Model ---")
    loaded_predictions = loaded_recommender.predict(user_to_predict, item_to_predict)
    print(f"Loaded model prediction for User {user_to_predict[0]} on Movie {item_to_predict[0]}: {loaded_predictions[0]:.4f}")
    print(f"Loaded model prediction for User {user_to_predict[1]} on Movie {item_to_predict[1]}: {loaded_predictions[1]:.4f}")

    print("\nDemonstration complete.")

if __name__ == "__main__":
    main()
