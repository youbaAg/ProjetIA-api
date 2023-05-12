from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)



# Charger le modèle entraîné à partir du fichier
model = joblib.load('model.joblib')

# Charger les colonnes à partir du fichier
with open('columns.txt', 'r') as f:
    columns = f.read().splitlines()
    
# Endpoint pour faire des prédictions
@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données de test
    test_data = request.json
    
    # Transformer les données en DataFrame
    test_df = pd.DataFrame(test_data, index=[0])
    
    # Encoder les données en utilisant les colonnes d'encodage one-hot
    test_df = pd.get_dummies(test_df)
    test_df = test_df.reindex(columns=columns, fill_value=0)
    
    # Faire la prédiction
    prediction = model.predict(test_df)
    
    # Retourner la prédiction en format JSON
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)