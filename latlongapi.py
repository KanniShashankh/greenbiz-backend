import json
from flask import Flask, request, jsonify
from math import radians, sin, cos, sqrt, atan2
import pandas as pd

app = Flask(__name__)

# Load your CSV file with dustbin coordinates
df = pd.read_csv('bins.csv')

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


@app.route('/locate', methods=['POST'])
def nearest_dustbins():
    try:
        # Get user's location coordinates and radius from the request
        # print(request.data)
        data = json.loads(request.data)

        user_lat = float(data['lat'])
        user_lon = float(data['lon'])
        radius = float(data['radius'])
        print(f'lol : {user_lat} {user_lon} {radius}')

        # Calculate distances to all dustbins
        df['distance'] = df.apply(lambda row: haversine(user_lat, user_lon, row['Latitude'], row['Longitude']), axis=1)

        # Filter dustbins within the specified radius
        nearby_dustbins = df[df['distance'] <= radius]

        nearby_dustbins = df.sort_values(by='distance')
        # Return the list of nearby dustbins as JSON
        result = nearby_dustbins[['Latitude', 'Longitude', 'distance']].to_dict(orient='records')
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(port=3001)
