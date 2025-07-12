
import pandas as pd
import openai
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import hashlib
from datetime import datetime
import streamlit as st  # ✅ Assure-toi que cette ligne est présente

# ... les autres imports


# === Chargement sécurisé des variables d'environnement === #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# === Configuration de l'API OpenAI === #
if not OPENAI_API_KEY:
    st.error("❌ Clé API OpenAI manquante. Vérifie le fichier .env")
    st.stop()
openai.api_key = OPENAI_API_KEY

# === Configuration de l'application Streamlit === #
st.set_page_config(page_title="Agent IA sécurisé", layout="wide")

# === Authentification utilisateur === #
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False
if not st.session_state.authentifie:
    mdp = st.text_input("🔐 Entrez le mot de passe :", type="password")
    if mdp == APP_PASSWORD:
        st.session_state.authentifie = True
        st.success("✅ Authentification réussie")
    else:
        st.stop()

st.title("📊 Agent IA Sécurisé - Visualisation Intelligente avec Seaborn")

# === Upload CSV === #
uploaded_file = st.file_uploader("Téléverse un fichier CSV (max 200 Mo)", type=["csv"])

@st.cache_data(show_spinner=False)
def hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

# === Générer des suggestions de graphiques avec l'IA === #
def generer_suggestions(df):
    sample = {
        "extrait": df.head(5).to_dict(),
        "stats": df.describe(include='all').fillna("").to_dict()
    }

    prompt = f"""
Tu es un expert en visualisation de données.
Voici un extrait et des statistiques :
{json.dumps(sample, indent=2)}

Propose une LISTE JSON de graphiques à créer avec :
- type (barplot, lineplot, scatter, pie)
- colonnes (liste)
- objectif (texte court)
Seulement une liste JSON.
    """

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu aides à créer des dashboards visuels utiles."},
                {"role": "user", "content": prompt}
            ]
        )
        contenu = response.choices[0].message.content
        return json.loads(contenu)

    except json.JSONDecodeError:
        st.error("❌ Erreur : le format JSON retourné par GPT est invalide.")
        return []
    except openai.OpenAIError as e:
        st.error(f"❌ Erreur OpenAI : Code d’erreur : {e}")
        return []
    except Exception as e:
        st.error(f"❌ Autre erreur : {e}")
        return []

# === Génération des graphiques avec Seaborn === #
def afficher_graphique(df, suggestion):
    try:
        type_graph = suggestion["type"]
        cols = suggestion["colonnes"]

        plt.figure(figsize=(8, 5))
        if type_graph == "barplot":
            sns.barplot(data=df, x=cols[0], y=cols[1])
        elif type_graph == "lineplot":
            sns.lineplot(data=df, x=cols[0], y=cols[1])
        elif type_graph == "scatter":
            sns.scatterplot(data=df, x=cols[0], y=cols[1])
        elif type_graph == "pie":
            pie_data = df.groupby(cols[0])[cols[1]].sum()
            plt.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
            plt.axis("equal")
        else:
            st.warning(f"Type de graphique non reconnu : {type_graph}")
            return

        plt.title(suggestion["objectif"])
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.error(f"❌ Erreur dans le graphique : {e}")

# === Traitement du fichier uploadé === #
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Conversion automatique des dates
        colonnes_dates = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        for col in colonnes_dates:
            try:
                df[col] = pd.to_datetime(df[col])
                df["mois"] = df[col].dt.month
                df["annee"] = df[col].dt.year
                df["mois_annee"] = df[col].dt.to_period("M").astype(str)
                st.success(f"✅ Colonne de date détectée et transformée : {col}")
                break
            except Exception:
                continue

        # Validation des données
        if df.shape[1] < 2:
            st.error("❌ Le fichier doit contenir au moins 2 colonnes")
            st.stop()

        if not any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
            st.error("❌ Il faut au moins une colonne numérique")
            st.stop()

        st.success("✅ Fichier chargé avec succès")
        st.write("📄 Aperçu des données :", df.head(20))

        cle_cache = hash_dataframe(df)
        if f"suggestions_{cle_cache}" not in st.session_state:
            with st.spinner("🧠 Analyse IA en cours..."):
                suggestions = generer_suggestions(df)
                st.session_state[f"suggestions_{cle_cache}"] = suggestions
        else:
            suggestions = st.session_state[f"suggestions_{cle_cache}"]

        if suggestions:
            st.subheader("📌 Suggestions de graphiques par l’IA")
            for i, s in enumerate(suggestions):
                with st.expander(f"📈 Graphique {i+1} : {s['objectif']}"):
                    st.json(s)
                    afficher_graphique(df, s)
        else:
            st.warning("Aucune suggestion trouvée.")

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement : {e}")