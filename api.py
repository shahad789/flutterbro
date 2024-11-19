from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import lightgbm as lgb
import unicodedata

app = Flask(__name__)
CORS(app)

# I 
property_type_mapping = {
    'apartment': 0.0,
    'شقة': 0.0,
    'villa': 0.99,
    'فيلا': 0.99,
    'floor': 0.49,
    'دور': 0.49
}

def normalize_text(text):
    if isinstance(text, str):
        return unicodedata.normalize('NFKC', text).strip().lower()
    return text

english_to_arabic_city_map = {
    'riyadh': 'الرياض',
    'jeddah': 'جدة',
    'dammam': 'الدمام',
    'khobar': 'الخبر'
}

train_data = pd.read_csv('train_90.csv')
befpreprocess_data = pd.read_csv('befpreprocess.csv')

befpreprocess_data['city'] = befpreprocess_data['city'].apply(normalize_text)
befpreprocess_data['district'] = befpreprocess_data['district'].apply(normalize_text)

city_mapping = {city: idx / len(befpreprocess_data['city'].unique()) for idx, city in enumerate(befpreprocess_data['city'].unique())}
district_mapping = {district: idx / len(befpreprocess_data['district'].unique()) for idx, district in enumerate(befpreprocess_data['district'].unique())}

X_train = train_data.drop(columns=['price'])
y_train = train_data['price']
lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'objective': 'regression',
    'metric': ['mae', 'rmse'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'force_row_wise': True,
    'max_bin': 550,
    'subsample_for_bin': 200000,
    'min_child_samples': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
}

model = lgb.train(params, lgb_train, num_boost_round=200)

@app.route('/districts', methods=['GET'])
def get_districts():
    city_name = normalize_text(request.args.get('city', ''))

    if city_name not in befpreprocess_data['city'].unique():
        print(f"Received city: '{city_name}' not found in city data.")
        return jsonify({'error': 'تم اختيار مدينة غير صحيحة.'}), 400

    districts = list(set(befpreprocess_data[befpreprocess_data['city'] == city_name]['district'].tolist()))
    districts = [normalize_text(district) for district in districts]

    cleaned_districts = [district.replace("حي ", "").strip() for district in districts]

    print(f"Available districts for '{city_name}': {cleaned_districts}")
    for district in cleaned_districts:
        encoded_value = district_mapping.get(district, 'N/A')
        print(f"District: '{district}' -> Encoded Value: {encoded_value}")

    return jsonify({'districts': cleaned_districts})

def preprocess_input(data):
    city = normalize_text(data.get('city', ''))
    district = normalize_text(data.get('district', ''))
    property_type = normalize_text(data.get('property_type', ''))

    print(f"Preprocessing input: city='{city}', district='{district}', property_type='{property_type}'")

    # Print the encoded mappings before checking
    print("City Mapping:", city_mapping)
    print("District Mapping:", district_mapping)
    print("Property Type Mapping:", property_type_mapping)

    if city not in city_mapping:
        print(f"City '{city}' not found in city_mapping.")
        return "Invalid city value.", None
    if district not in district_mapping:
        print(f"District '{district}' not found in district_mapping.")
        return "Invalid district value.", None
    if property_type not in property_type_mapping:
        print(f"Property type '{property_type}' not found in property_type_mapping.")
        return "Invalid property type value. Choose from: شقة, فيلا, دور", None

    try:
        city_encoded = city_mapping[city]
        district_encoded = district_mapping[district]
        property_type_encoded = property_type_mapping[property_type]
        area = float(data['area'])
        rooms = int(data['rooms'])
        bathrooms = int(data['bathrooms'])

        print(f"Encoded Values: city='{city}' -> {city_encoded}, district='{district}' -> {district_encoded}, property_type='{property_type}' -> {property_type_encoded}")
    except ValueError as e:
        print(f"Error in input conversion: {e}")
        return "Invalid input format. Ensure numeric values for area, rooms, and bathrooms.", None

    input_data = pd.DataFrame([{
        'city': city_encoded,
        'district': district_encoded,
        'property_type': property_type_encoded,
        'area': area,
        'rooms': rooms,
        'bathrooms': bathrooms,
    }])

    print(f"Encoded input data:\n{input_data}")
    return None, input_data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    error, input_data = preprocess_input(data)
    if error:
        return jsonify({'error': error}), 400

    print("Making prediction with input data:")
    print(input_data)

    prediction = model.predict(input_data)
    predicted_price = round(prediction[0], 2)

    print(f"Predicted price: {predicted_price}")
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=6000)
