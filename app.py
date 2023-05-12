from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return """
    <h1>Bienvenue sur notre API Fast Fashion predictions</h1>
    <p>Notre objectif est de changer la façon dont l'industrie fast fashion fonctionne en évitant la surproduction de vêtements et en promouvant une approche plus durable. 

    <p>Un modèle prédictif est utilisé pour déterminer si un article sera en rupture de stock ou s'il restera en stock. 
    Grâce à cette prédiction, nous pouvons ajuster notre production pour éviter les surproductions inutiles. 
    Cela permet de réduire les déchets et les émissions de CO2 tout en créant des vêtements de qualité que les clients adorent</p>

    <p>Vous pouvez effectuer des prédictions en utilisant l'endpoint <code>/predict</code>. Pour cela, vous devez fournir les paramètres suivants:</p>
    <ul>
        <li><code>price</code>: le prix du produit (requis)</li>
        <li><code>color</code>: la couleur du produit (requis)</li>
        <li><code>brand</code>: la marque du produit (requis)</li>
        <li><code>category</code>: la catégorie du produit (requis)</li>
    </ul>
    <p>Vous pouvez fournir les paramètres soit dans le corps de la demande JSON (si vous utilisez une méthode POST), soit dans les paramètres de l'URL (si vous utilisez une méthode GET).</p>
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

    # Vérifier si les données de test sont fournies
    if not all(test_data.values()):
        # Si les données ne sont pas fournies, retourner une erreur 400 (Bad Request)
        return jsonify({'error': 'Les données de test sont manquantes price, color , brand, category.'}), 400

    # Transformer les données en DataFrame
    test_df = pd.DataFrame(test_data, index=[0])

    # Encoder les données en utilisant les colonnes d'encodage one-hot
    test_df = pd.get_dummies(test_df)
    test_df = test_df.reindex(columns=columns, fill_value=0)

    # Faire la prédiction
    prediction = model.predict(test_df)

    # Retourner la prédiction en format JSON
    return jsonify({'prediction': prediction[0]})
