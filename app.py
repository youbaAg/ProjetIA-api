from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return """
    <h1>Bienvenue sur mon API</h1>
    <p>Cette API est utilisée pour prédire des données en utilisant un modèle ML entraîné.</p>
    """
    
# Charger le modèle entraîné à partir du fichier
model = joblib.load('model.joblib')


# Charger les colonnes à partir du fichier
with open('columns.txt', 'r') as f:
    columns = f.read().splitlines()

# Endpoint pour faire des prédictions
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Récupérer les données de test à partir du corps de la demande JSON
        test_data = request.json
    else:
        # Récupérer les données de test à partir des paramètres de l'URL
        test_data = {
            'price': request.args.get('price'),
            'color': request.args.get('color'),
            'brand': request.args.get('brand'),
            'category': request.args.get('category')
        }

    # Transformer les données en DataFrame
    test_df = pd.DataFrame(test_data, index=[0])

    # Encoder les données en utilisant les colonnes d'encodage one-hot
    test_df = pd.get_dummies(test_df)
    test_df = test_df.reindex(columns=columns, fill_value=0)

    # Faire la prédiction
    prediction = model.predict(test_df)

    # Retourner la prédiction en format JSON
    return jsonify({'prediction': prediction[0]})
