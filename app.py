from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import save_model
from PIL import Image
import json
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
from flask_pymongo import PyMongo
from pymongo import MongoClient



app = Flask(__name__)

# Load the pre-trained Keras model
model_path = "trained_model.h5"
model = load_model(model_path)

# Assuming the model expects images of shape (300, 300)
img_height, img_width = 300, 300

df = pd.read_csv('bins.csv')

app.config['MONGO_URI'] = 'mongodb+srv://gnits-trashcent:sudeep@weight.dhdf8nq.mongodb.net/trashcent'
# mongo = pymongo(app)
# collection = mongo.weights

client = MongoClient('mongodb+srv://gnits-trashcent:sudeep@weight.dhdf8nq.mongodb.net/trashcent')
db = client['trashcent']
collection = db['weights']

# Assuming labels are loaded from a file
with open('labels.txt', 'r') as f:
    labels = f.read().splitlines()
    
@app.route('/nigga', methods=['GET'])
def lawda():
    return jsonify({"yes" : "nigga"})
    

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        # print(image_file)
        img = Image.open(image_file)
        # print(img)
        img = img.resize((img_height, img_width)) 


        # Load and preprocess the image
        # img = image./zload_img(image_file, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Perform inference
        predictions = model.predict(img_array)

        # Postprocess output data
        max_probability = np.max(predictions[0], axis=-1)
        predicted_class = labels[np.argmax(predictions[0], axis=-1)]

        result = {
            'maximum_probability': max_probability.item(),
            'predicted_class': predicted_class,
            'class_probabilities': {label: round(prob * 100, 2) for label, prob in zip(labels, predictions[0])}
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/locate', methods=['POST'])
def nearest_dustbins():
    try:
        # Get user's location coordinates and radius from the request
        # print(request.data)
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0  # Radius of the Earth in kilometers

            # Convert latitude and longitude from degrees to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

            # Calculate the differences between latitudes and longitudes
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            # Haversine formula
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            # Distance in kilometers
            distance = R * c
            return distance
        
        data = json.loads(request.data)

        user_lat = float(data['lat'])
        user_lon = float(data['lon'])
        radius = float(data['radius'])
        # print(f'lol : {user_lat} {user_lon} {radius}')

        # Calculate distances to all dustbins
        df['distance'] = df.apply(lambda row: haversine(user_lat, user_lon, row['Latitude'], row['Longitude']), axis=1)

        # Filter dustbins within the specified radius
        
        # print(type(df['distance']))
        
        nearby_dustbins = df[df['distance'].astype(float) <= radius]

        nearby_dustbins = nearby_dustbins.sort_values(by='distance')
        # Return the list of nearby dustbins as JSON
        result = nearby_dustbins[['Latitude', 'Longitude', 'distance', 'Assessor Address']].to_dict(orient='records')
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/weights', methods=['POST'])
def store_data():
    try:
        # Get parameters from the POST request
        data = json.loads(request.data)
        
        weight = data['weight']
        pickup_id = data['pickup_id']
        category = data['category']

        # Store data in MongoDB
        data = {
            'weight': weight,
            'pickup_id': pickup_id,
            'category': category
        }

        # Insert the data into the MongoDB collection
        collection.insert_one(data)

        return jsonify({'message': 'Data stored successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='10.10.8.83', port=3000)



