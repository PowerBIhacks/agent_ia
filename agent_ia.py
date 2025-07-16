import pandas as pd
import json
import sqlalchemy
import chardet
import os
import tempfile
import chardet
import pandas as pd
import streamlit as st
import sqlalchemy
import pymysql
import psycopg2
import sqlite3
import requests
import json
from google.cloud import bigquery
from google.oauth2 import service_account
from pymongo import MongoClient
from ftplib import FTP
import paramiko  # pour SFTP
import os
import re
import ast
import io
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import hashlib
import streamlit as st
import cohere
from PIL import Image
import plotly.express as px
import plotly.io as pio  # nécessaire pour write_image
import seaborn as sns
import matplotlib.pyplot as plt
import chardet


# --- CONFIG STREAMLIT --- #
st.set_page_config(page_title="Agent IA sécurisé", layout="wide", page_icon="📊")
st.markdown("""
<style>
    body {
        background-color: #f7f9fb;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .reportview-container .main .block-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .stButton>button {
        color: white;
        background-color: #0072C6;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
    }
    .stTextInput>div>div>input {
        border-radius: 6px;
        padding: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


df = None
    
def verifier_utilisateur(email, password):
    try:
        # Chemin relatif vers users.json dans le même dossier que ce script
        chemin_utilisateur = os.path.join(os.path.dirname(__file__), "users.json")
        with open(chemin_utilisateur, "r") as f:
            users = json.load(f)
        for user in users:
            if user.get("email") == email and user.get("password") == password:
                return True
        return False
    except FileNotFoundError:
        st.error("❌ Fichier users.json introuvable.")
        return False
    except Exception as e:
        st.error(f"❌ Erreur de lecture users.json : {e}")
        return False

# --- Interface connexion ---
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.title("🔐 Connexion à l’Agent IA sécurisé")
    email = st.text_input("Email")
    pwd = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if verifier_utilisateur(email, pwd):
            st.success("✅ Connexion réussie.")
            st.session_state.authentifie = True
            st.session_state.utilisateur = email
        else:
            st.error("❌ Identifiants incorrects.")
    st.stop()


API_KEY="iE0yYFbKh7cigvuHlvdGm3E4AMtdXVq0HhcG1TRH"
client = cohere.Client(API_KEY)

def generer_suggestions(df):
    # ... ton code qui utilise client.generate() ...
    response = client.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=700,
        temperature=0.4,
    )
    # etc.


# --- FONCTIONS UTILITAIRES --- #

def hash_dataframe(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def convertir_json_compatible(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):  # gère NaN, NaT
        return None
    elif isinstance(obj, dict):
        return {k: convertir_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_json_compatible(i) for i in obj]
    else:
        return obj

def sauvegarder_session(email, resume, suggestions):
    try:
        conn = sqlite3.connect("sessions.db")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (email TEXT, resume TEXT, suggestions TEXT)''')
        c.execute("INSERT INTO sessions (email, resume, suggestions) VALUES (?, ?, ?)",
                  (email, resume, json.dumps(suggestions)))
        conn.commit()
        conn.close()
    except Exception as e:
        st.warning(f"Erreur de sauvegarde : {e}")

# --- DETECTION DU DOMAINE --- #

def detecter_domaine(df):
    texte = " ".join(df.columns).lower()
    if any(kw in texte for kw in ["produit", "vente", "prix", "quantite", "ca","fournisseur"]):
        return "ventes"
    elif any(kw in texte for kw in ["note", "moyenne", "eleve", "classe","matière","niveau d'étude"]):
        return "scolaire"
    elif any(kw in texte for kw in ["employe", "salaire", "poste", "absences","présence","domaine"]):
        return "rh"
    elif any(kw in texte for kw in ["patient", "diagnostic", "soin", "traitement","maladie"]):
        return "sante"
    elif any(kw in texte for kw in ["depense", "revenu", "budget", "finance"]):
        return "finance"
    elif any(kw in texte for kw in ["livraison", "stock", "produit", "entrepot"]):
        return "logistique"
    elif any(kw in texte for kw in ["projet", "avancement", "delai", "tache"]):
        return "projets"
    elif any(kw in texte for kw in ["trafic", "visite", "clic", "conversion","site web"]):
        return "webmarketing"
    elif any(kw in texte for kw in ["incident", "support", "ticket", "resolution"]):
        return "it_support"
    elif any(kw in texte for kw in ["pourboir","prix par personne","heure de repas","nombre de personne","chambre"]):
        return "hotel"
    else:
        return "autre"

# --- AFFICHAGE KPIS --- #


def afficher_kpis(df, domaine):
    st.subheader("📌 Indicateurs clés")

    # Indicateurs généraux
    if "prix" in (c.lower() for c in df.columns):
        st.metric("💰 Total Prix de vente", f"{df[[c for c in df.columns if 'prix' in c.lower()][0]].sum():,.0f} FCFA")
    if any("licence" in c.lower() for c in df.columns):
        st.metric("📦 Total Licences vendues", f"{df[[c for c in df.columns if 'licence' in c.lower()][0]].sum():,.0f}")

    # --- DOMAINES --- #

    if domaine == "ventes":
        prix_col = next((col for col in df.columns if 'prix' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        quant_col = next((col for col in df.columns if 'quant' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        client_col = next((col for col in df.columns if 'client' in col.lower()), None)
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        produit_col = next((col for col in df.columns if 'produit' in col.lower()), None)
        categorie_col = next((col for col in df.columns if 'catégorie' in col.lower() or 'categorie' in col.lower()), None)

        if prix_col and quant_col:
            df["CA"] = df[prix_col] * df[quant_col]
            if date_col:
                df["Année"] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True).dt.year

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 CA total", f"{df['CA'].sum():,.0f} FCFA")
            col2.metric("📦 Quantité vendue", f"{df[quant_col].sum():,.0f}")
            col3.metric("💵 CA moyen", f"{df['CA'].mean():,.0f} FCFA")
            if client_col:
                col4.metric("👥 Clients uniques", f"{df[client_col].nunique():,}")

            graphs = []
        
            if "Année" in df.columns:
                    # Somme du CA par année
                ca_annee = df.groupby("Année")["CA"].mean().reset_index()
                fig1 = px.line(ca_annee, x="Année", y="CA", title="📈 CA par Année")
                graphs.append(fig1)
                
                    # Moyenne du CA par année
                ca_annee_mean = df.groupby("Année")["CA"].mean().reset_index()
                fig_mean = px.line(ca_annee_mean, x="Année", y="CA", title="📉 CA moyen par Année")
                graphs.append(fig_mean)

            if produit_col:
                quant_produit = df.groupby(produit_col)[quant_col].sum().reset_index().sort_values(by=quant_col, ascending=False)
                fig2 = px.bar(quant_produit, x=produit_col, y=quant_col, title="📦 Quantité vendue par produit")
                graphs.append(fig2)

                ca_produit = df.groupby(produit_col)["CA"].sum().reset_index().sort_values(by="CA", ascending=False)
                fig3 = px.bar(ca_produit, x=produit_col, y="CA", title="💰 CA par produit")
                graphs.append(fig3)

            # Vérifie que les colonnes nécessaires existent
            categorie_col = next((col for col in df.columns if 'catégorie' in col.lower() or 'categorie' in col.lower()), None)

            if "CA" in df.columns and categorie_col:
                ca_par_categorie = df.groupby(categorie_col)["CA"].sum().reset_index().sort_values(by="CA", ascending=False)
                fig_ca_categorie = px.pie(
                    ca_par_categorie,
                    names=categorie_col,
                    values="CA",
                    title="💼 Chiffre d’affaires par catégorie de produit"
                )
                st.plotly_chart(fig_ca_categorie, use_container_width=True)
            else:
                pass


            # Affichage 3 par ligne
            for i in range(0, len(graphs), 3):
                cols = st.columns(3)
                for j, fig in enumerate(graphs[i:i+3]):
                    with cols[j]:
                        st.plotly_chart(fig, use_container_width=True)

        else:
            pass

    elif domaine == "scolaire":
        note_col = next((col for col in df.columns if 'note' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        classe_col = next((col for col in df.columns if 'classe' in col.lower()), None)
        matiere_col = next((col for col in df.columns if 'matiere' in col.lower() or 'matière' in col.lower()), None)

        if note_col:
            st.metric("🎓 Moyenne générale", f"{df[note_col].mean():.2f}")
        if classe_col:
            st.metric("🏫 Nombre de classes", f"{df[classe_col].nunique():,}")

        figs = []
        if matiere_col and note_col:
            moyennes = df.groupby(matiere_col)[note_col].mean().reset_index()
            figs.append(px.bar(moyennes, x=matiere_col, y=note_col, title="📘 Moyenne par matière"))

        if classe_col and note_col:
            moyenne_classe = df.groupby(classe_col)[note_col].mean().reset_index()
            figs.append(px.bar(moyenne_classe, x=classe_col, y=note_col, title="🏫 Moyenne par classe"))

        for i in range(0, len(figs), 3):
            cols = st.columns(3)
            for j, fig in enumerate(figs[i:i+3]):
                with cols[j]:
                    st.plotly_chart(fig, use_container_width=True)

    elif domaine == "finance":
        montant_col = next((col for col in df.columns if 'montant' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        type_col = next((col for col in df.columns if 'type' in col.lower()), None)
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        categorie_col = next((col for col in df.columns if 'categorie' in col.lower() or 'catégorie' in col.lower()), None)
        
        if not montant_col:
            #st.info("Colonne 'montant' nécessaire pour afficher les indicateurs financiers.")
            return
        
        # Conversion date et extraction année/mois
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[f"{date_col}_année"] = df[date_col].dt.year
            df[f"{date_col}_mois"] = df[date_col].dt.month

        # KPIs financiers
        total_montants = df[montant_col].sum()
        moyenne_montants = df[montant_col].mean()
        nb_operations = df.shape[0]
        st.metric("💸 Total des montants", f"{total_montants:,.0f} FCFA")
        st.metric("📊 Montant moyen", f"{moyenne_montants:,.0f} FCFA")
        st.metric("📋 Nombre d'opérations", f"{nb_operations:,}")

        graphs = []

        # Répartition par type (pie chart)
        if type_col:
            repartition_type = df.groupby(type_col)[montant_col].sum().reset_index()
            fig_type = px.pie(repartition_type, names=type_col, values=montant_col, title="💼 Répartition des montants par type")
            graphs.append(fig_type)

        # Répartition par catégorie (bar chart)
        if categorie_col:
            repartition_categorie = df.groupby(categorie_col)[montant_col].sum().reset_index()
            fig_categorie = px.bar(repartition_categorie, x=categorie_col, y=montant_col, title="📊 Montants par catégorie")
            graphs.append(fig_categorie)

        # Evolution du montant total par année (line chart)
        annees_col = f"{date_col}_année"
        if date_col and annees_col in df.columns:
            evolution_annee_sum = df.groupby(annees_col)[montant_col].sum().reset_index()
            fig_evo_sum = px.line(evolution_annee_sum, x=annees_col, y=montant_col, title="📈 Evolution du total des montants par année")
            graphs.append(fig_evo_sum)

            # Evolution montant moyen par année (line chart)
            evolution_annee_mean = df.groupby(annees_col)[montant_col].mean().reset_index()
            fig_evo_mean = px.line(evolution_annee_mean, x=annees_col, y=montant_col, title="📉 Evolution du montant moyen par année")
            graphs.append(fig_evo_mean)

        # Affichage graphiques 3 par ligne
        for i in range(0, len(graphs), 3):
            cols = st.columns(3)
            for j, fig in enumerate(graphs[i:i+3]):
                cols[j].plotly_chart(fig, use_container_width=True)

    elif domaine == "logistique":
        stock_col = next((col for col in df.columns if 'stock' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        entrepot_col = next((col for col in df.columns if 'entrepot' in col.lower()), None)
        if stock_col:
            st.metric("📦 Stock total", f"{df[stock_col].sum():,.0f}")
        if entrepot_col and stock_col:
            stock_entrepot = df.groupby(entrepot_col)[stock_col].sum().reset_index()
            fig = px.bar(stock_entrepot, x=entrepot_col, y=stock_col, title="Stock par entrepôt")
            st.plotly_chart(fig, use_container_width=True)

    elif domaine == "webmarketing":
        clics_col = next((col for col in df.columns if 'clic' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        campagne_col = next((col for col in df.columns if 'campagne' in col.lower()), None)
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        impressions_col = next((col for col in df.columns if 'impression' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        clics_uniques_col = next((col for col in df.columns if 'clic' in col.lower() and 'unique' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        ville_col = next((col for col in df.columns if 'ville' in col.lower()), None)

        if not clics_col:
            st.info("Colonne 'clics' nécessaire pour afficher les indicateurs webmarketing.")
            return

        # Conversion date et extraction année/mois
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[f"{date_col}_année"] = df[date_col].dt.year
            df[f"{date_col}_mois"] = df[date_col].dt.month

        # KPIs webmarketing
        total_clics = df[clics_col].sum()
        moyenne_clics = df[clics_col].mean()
        total_impressions = df[impressions_col].sum() if impressions_col else None
        total_clics_uniques = df[clics_uniques_col].sum() if clics_uniques_col else None
        nb_jours = df.shape[0]

        st.metric("🖱 Total des clics", f"{total_clics:,}")
        st.metric("📊 Clics moyens par enregistrement", f"{moyenne_clics:.2f}")
        if total_impressions is not None:
            st.metric("👁 Total des impressions", f"{total_impressions:,}")
        if total_clics_uniques is not None:
            st.metric("🆕 Clics uniques totaux", f"{total_clics_uniques:,}")
        st.metric("📅 Nombre d'enregistrements", f"{nb_jours:,}")

        graphs = []

        # Clics par campagne (bar chart)
        if campagne_col:
            clics_campagne = df.groupby(campagne_col)[clics_col].sum().reset_index()
            fig_campagne = px.bar(clics_campagne, x=campagne_col, y=clics_col, title="🛠️ Clics par campagne")
            graphs.append(fig_campagne)

        # Clics par ville (bar chart)
        if ville_col:
            clics_ville = df.groupby(ville_col)[clics_col].sum().reset_index()
            fig_ville = px.bar(clics_ville, x=ville_col, y=clics_col, title="🌍 Clics par ville")
            graphs.append(fig_ville)

        # Evolution des clics par année (line chart)
        annees_col = f"{date_col}_année"
        if date_col and annees_col in df.columns:
            evolution_annee_clics = df.groupby(annees_col)[clics_col].sum().reset_index()
            fig_evo_clics = px.line(evolution_annee_clics, x=annees_col, y=clics_col, title="📈 Evolution des clics par année")
            graphs.append(fig_evo_clics)

            # Evolution des impressions par année (line chart)
            if impressions_col:
                evolution_annee_impr = df.groupby(annees_col)[impressions_col].sum().reset_index()
                fig_evo_impr = px.line(evolution_annee_impr, x=annees_col, y=impressions_col, title="📊 Evolution des impressions par année")
                graphs.append(fig_evo_impr)

        # Affichage 3 graphiques par ligne
        for i in range(0, len(graphs), 3):
            cols = st.columns(3)
            for j, fig in enumerate(graphs[i:i+3]):
                cols[j].plotly_chart(fig, use_container_width=True)


    elif domaine == "projets":
        st.subheader("📌 Suivi des Projets")

        taches_col = next((col for col in df.columns if 'tâche' in col.lower() or 'tache' in col.lower()), None)
        avancement_col = next((col for col in df.columns if 'avancement' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        statut_col = next((col for col in df.columns if 'statut' in col.lower()), None)
        budget_col = next((col for col in df.columns if 'budget' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        responsable_col = next((col for col in df.columns if 'responsable' in col.lower()), None)
        date_col = next((col for col in df.columns if 'début' in col.lower() or 'debut' in col.lower()), None)

        st.metric("📋 Nombre de projets", f"{df.shape[0]:,}")
        if taches_col:
            st.metric("🧩 Nombre total de tâches", f"{df[taches_col].sum():,}")
        if avancement_col:
            st.metric("📈 Taux d’avancement moyen", f"{df[avancement_col].mean():.0f}%")
        if budget_col:
            st.metric("💰 Budget total estimé", f"{df[budget_col].sum():,.0f} FCFA")

        graphs = []

        # Répartition des statuts
        if statut_col:
            status_counts = df[statut_col].value_counts().reset_index()
            status_counts.columns = [statut_col, "Nombre"]
            fig_statut = px.pie(status_counts, names=statut_col, values="Nombre", title="📊 Répartition des statuts de projet")
            graphs.append(fig_statut)

        # Avancement moyen par responsable
        if responsable_col and avancement_col:
            avancement_resp = df.groupby(responsable_col)[avancement_col].mean().reset_index()
            fig_av_resp = px.bar(avancement_resp, x=responsable_col, y=avancement_col, title="👤 Avancement moyen par responsable")
            graphs.append(fig_av_resp)

        # Budget par projet
        nom_projet_col = next((col for col in df.columns if 'nom' in col.lower() and 'projet' in col.lower()), None)
        if nom_projet_col and budget_col:
            fig_budget = px.bar(df, x=nom_projet_col, y=budget_col, title="💸 Budget par projet")
            graphs.append(fig_budget)

        # Évolution des projets dans le temps
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df["Année"] = df[date_col].dt.year
            projets_par_annee = df["Année"].value_counts().reset_index()
            projets_par_annee.columns = ["Année", "Nombre de projets"]
            fig_temps = px.line(projets_par_annee.sort_values("Année"), x="Année", y="Nombre de projets", title="📆 Projets lancés par année")
            graphs.append(fig_temps)

        # Affichage des graphiques : 3 par ligne
        for i in range(0, len(graphs), 3):
            cols = st.columns(3)
            for j, fig in enumerate(graphs[i:i+3]):
                cols[j].plotly_chart(fig, use_container_width=True)


    elif domaine == "it_support":
        st.subheader("📌 Support IT")

        statut_col = next((col for col in df.columns if 'statut' in col.lower()), None)
        technicien_col = next((col for col in df.columns if 'technicien' in col.lower()), None)
        probleme_col = next((col for col in df.columns if 'problème' in col.lower() or 'probleme' in col.lower()), None)
        duree_col = next((col for col in df.columns if 'durée' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)

        st.metric("🛠 Tickets enregistrés", f"{df.shape[0]:,}")
        if statut_col:
            st.metric("✅ Types de statuts", f"{df[statut_col].nunique():,}")
        if duree_col:
            st.metric("⏱ Durée moyenne de résolution", f"{df[duree_col].mean():.1f} h")

        graphs = []

        # Répartition des statuts
        if statut_col:
            repartition_statut = df[statut_col].value_counts().reset_index()
            repartition_statut.columns = [statut_col, "Nombre"]
            fig_statut = px.pie(repartition_statut, names=statut_col, values="Nombre", title="🧾 Répartition des statuts de tickets")
            graphs.append(fig_statut)

        # Problèmes les plus fréquents
        if probleme_col:
            probleme_freq = df[probleme_col].value_counts().reset_index()
            probleme_freq.columns = [probleme_col, "Nombre"]
            fig_problemes = px.bar(probleme_freq, x=probleme_col, y="Nombre", title="📌 Types de problèmes fréquents")
            graphs.append(fig_problemes)

        # Tickets par technicien
        if technicien_col:
            tech_stats = df[technicien_col].value_counts().reset_index()
            tech_stats.columns = [technicien_col, "Nombre"]
            fig_techs = px.bar(tech_stats, x=technicien_col, y="Nombre", title="👨‍💻 Tickets par technicien")
            graphs.append(fig_techs)

        # Évolution des tickets par date
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df["Jour"] = df[date_col].dt.date
            evolution = df["Jour"].value_counts().sort_index().reset_index()
            evolution.columns = ["Date", "Nombre de tickets"]
            fig_evolution = px.line(evolution, x="Date", y="Nombre de tickets", title="📅 Évolution des tickets dans le temps")
            graphs.append(fig_evolution)

        # Affichage 3 graphiques par ligne
        for i in range(0, len(graphs), 3):
            cols = st.columns(3)
            for j, fig in enumerate(graphs[i:i+3]):
                cols[j].plotly_chart(fig, use_container_width=True)


    else:
        st.info("Aucun KPI spécifique détecté. Voici quelques statistiques générales :")
        st.metric("📊 Nombre de lignes", f"{df.shape[0]:,}")
        st.metric("📁 Nombre de colonnes", f"{df.shape[1]:,}")
        st.metric("🔍 Valeurs manquantes", f"{df.isnull().sum().sum():,}")



# --- CREATION DE FIGURES --- #

def creer_figure(type_graph, df, cols):
    try:
        if not cols or len(cols) < 1:
            return None

        col1 = cols[0]
        col2 = cols[1] if len(cols) > 1 else None

        # Règle : ne jamais croiser Prix et Quantité ou Prix et CA
        if (col1 and col2) and (
            ("prix" in col1.lower() and "quant" in col2.lower()) or
            ("quant" in col1.lower() and "prix" in col2.lower()) or
            ("prix" in col1.lower() and "ca" in col2.lower()) or
            ("ca" in col1.lower() and "prix" in col2.lower())
        ):
            return None

        # Comparer les ventes par client ou par catégorie → barplot ou pie
        if type_graph == "barplot" and col1 and col2:
            if "client" in col1.lower() or "produit" in col1.lower() or "categorie" in col1.lower():
                bar_data = df.groupby(col1)[col2].sum().reset_index()
                return px.bar(bar_data, x=col1, y=col2)
            else:
                return px.bar(df, x=col1, y=col2)

        elif type_graph == "pie" and col1 and col2:
            pie_data = df.groupby(col1)[col2].sum().reset_index()
            return px.pie(pie_data, names=col1, values=col2)

        elif type_graph == "lineplot" and col1 and col2:
            if "année" in col1.lower() or "mois" in col1.lower() or pd.api.types.is_datetime64_any_dtype(df[col1]):
                return px.line(df, x=col1, y=col2)

        elif type_graph == "scatter" and col1 and col2:
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                return px.scatter(df, x=col1, y=col2)

        elif type_graph == "boxplot" and col1 and col2:
            if pd.api.types.is_numeric_dtype(df[col2]):
                return px.box(df, x=col1, y=col2)

        return None
    except Exception as e:
        return None


st.subheader("📊 Résultats de l’analyse")
# --- AFFICHAGE GRAPHIQUES --- #

def afficher_graphique(df, suggestion, container):
    type_graph = suggestion["type"]
    cols = suggestion.get("colonnes") or suggestion.get("columns")

    if not cols or not all(col in df.columns for col in cols):
        container.warning("Colonnes manquantes ou incorrectes.")
        return

    # Interdire graphiques combinant Prix & Quantité ensemble
    colonnes_prix = [c for c in cols if "prix" in c.lower()]
    colonnes_quant = [c for c in cols if "quant" in c.lower() or "licence" in c.lower()]

    if colonnes_prix and colonnes_quant:
        # On ignore ces graphes
        return

    # Si la colonne est un identifiant client ou similaire, faire un countplot (bar chart des occurrences)
    colonnes_clients = [c for c in cols if "client" in c.lower() or "id" in c.lower()]
    if colonnes_clients and type_graph != "countplot":
        # Override : créer un bar chart des comptes
        client_col = colonnes_clients[0]
        freq = df[client_col].value_counts().reset_index()
        freq.columns = [client_col, "count"]
        fig = px.bar(freq, x=client_col, y="count", title=f"Répartition des {client_col}")
        container.plotly_chart(fig, use_container_width=True)
        return

    # Sinon utiliser la fonction habituelle pour les graphes autorisés
    fig = creer_figure(type_graph, df, cols)
    if fig:
        container.plotly_chart(fig, use_container_width=True)
    else:
        pass

# --- GENERER SUGGESTIONS VIA COHERE --- #


def generer_suggestions(df):
    extrait_dict = df.head(200).to_dict()
    stats_dict = df.describe(include='all').fillna("").to_dict()
    sample = {
        "extrait": convertir_json_compatible(extrait_dict),
        "stats": convertir_json_compatible(stats_dict)
    }
    prompt = f"""
Tu es un expert en visualisation de données.
Voici un extrait et des statistiques :
{json.dumps(sample, indent=2)}

Propose une LISTE JSON de graphiques à créer avec :
- type (barplot, lineplot, pie, boxplot)
- colonnes (liste)
- objectif (texte court)

CONTRAINTES:
- Les graphiques 'lineplot' ne doivent être proposés que si une colonne date, année ou mois est utilisée.
- Sinon, ne pas proposer de lineplot.

Seulement une liste JSON.
"""
    try:
        response = client.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=700,
            temperature=0.4,
        )
        contenu = response.generations[0].text.strip()
        try:
            return json.loads(contenu)
        except:
            match = re.search(r'\[.*\]', contenu, re.DOTALL)
            if match:
                suggestions = ast.literal_eval(match.group(0))
                if isinstance(suggestions, list):
                    return suggestions
        st.warning("⚠ Réponse de Cohere non exploitable comme JSON.")
        return []
    except Exception as e:
        st.error(f"❌ Erreur Cohere : {e}")
        return []

# --- GENERER RESUME --- #

def generer_resume(df):
    extrait_csv = df.head(200).to_csv(index=False)
    prompt = f"""
Tu es un expert en data analytics.

Voici les 200 premières lignes :
{extrait_csv}

Et les statistiques globales :
{df.describe(include='all').fillna('').to_string()}

Donne un résumé clair en français :
- La structure générale des données
- Les valeurs manquantes
- Les tendances visibles
- Les éventuelles anomalies
- Les propositions pour la prise de de décision

Sois concis et précis.
"""
    try:
        response = client.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=700,
            temperature=0.4,
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Erreur Cohere : {e}"

# --- FONCTION DE SAUVEGARDE --- #

def sauvegarder_session(utilisateur, resume, suggestions):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    nom_fichier = os.path.join(log_dir, f"{utilisateur}_historique.txt")
    with open(nom_fichier, "a", encoding="utf-8") as f:
        f.write(f"\n=== Analyse du {datetime.now()} ===\n")
        f.write(f"Résumé :\n{resume}\n")
        f.write("Suggestions :\n")
        for sugg in suggestions:
            f.write(f"- {sugg.get('objectif', 'Sans description')}\n")

# --- APPEL DANS LE FLUX PRINCIPAL --- #

# Supposons que df_to_use est ta DataFrame sur laquelle tu analyses
ECHANTILLON_IA = 200

if "utilisateur" in st.session_state and df is not None:
    suggestions = generer_suggestions(df.head(ECHANTILLON_IA))
    resume = generer_resume(df.head(ECHANTILLON_IA))

    st.subheader("Résumé de l'analyse")
    st.write(resume)

    st.subheader("Suggestions de graphiques")
    for s in suggestions:
        colonnes = tuple(s.get("colonnes", []))  # 🔁 Transformation sûre
        st.write(f"- {s.get('objectif', 'Sans description')} (Colonnes : {colonnes})")

        # Si tu veux appliquer un groupby ou autre ici :
        # grouped = df.groupby(colonnes)  # <- uniquement si nécessaire

    # Sauvegarde dans le fichier log
    sauvegarder_session(st.session_state.utilisateur, resume, suggestions)

# --- INTERFACE PRINCIPALE --- #

logo_path = "C:/Users/GINOV-PC/Desktop/archive/power.png"
if os.path.exists(logo_path):
    image = Image.open(logo_path)
    st.image(image, width=150)


st.title("🔍 Analyse intelligente des données")

ECHANTILLON_IA = 2000  # Ta variable d'échantillon IA, à définir plus haut dans ton script


@st.cache_data(show_spinner="📊 Chargement du fichier en cours...", max_entries=10)
def charger_fichier(uploaded_file):
    raw_data = uploaded_file.read()
    encodage_detecte = chardet.detect(raw_data)
    uploaded_file.seek(0)


# --- CONNECTEURS EN SIDEBAR ---

# Fonction utilitaire pour charger CSV avec détection encodage
def charger_csv(fichier):
    raw_data = fichier.read()
    encodage = chardet.detect(raw_data)['encoding']
    fichier.seek(0)
    return pd.read_csv(fichier, encoding=encodage, sep=None, engine='python')

with st.sidebar:
    st.header("🔌 Connecteurs de données")

    source_type = st.selectbox("Choisir une source de données", [
        "CSV / Excel",
        "JSON",
        "Parquet",
        "Google Sheets",
        "Base de données SQL",
        "SQLite local",
        "API REST (JSON)",
        "Google BigQuery",
        "MongoDB",
        "FTP / SFTP",
        "Google Drive (CSV / Excel)",
    ])

    df = None

    if source_type == "CSV / Excel":
        fichier = st.file_uploader("📁 Importer un fichier CSV ou Excel", type=["csv", "xlsx"])
        if fichier:
            try:
                if fichier.name.endswith(".csv"):
                    df = charger_csv(fichier)
                else:
                    df = pd.read_excel(fichier)
                st.success("✅ Fichier chargé avec succès")
            except Exception as e:
                st.error(f"Erreur chargement : {e}")

    elif source_type == "JSON":
        fichier = st.file_uploader("📁 Importer un fichier JSON", type=["json"])
        if fichier:
            try:
                df = pd.read_json(fichier)
                st.success("✅ Fichier JSON chargé")
            except Exception as e:
                st.error(f"Erreur chargement JSON : {e}")

    elif source_type == "Parquet":
        fichier = st.file_uploader("📁 Importer un fichier Parquet", type=["parquet"])
        if fichier:
            try:
                df = pd.read_parquet(fichier)
                st.success("✅ Fichier Parquet chargé")
            except Exception as e:
                st.error(f"Erreur chargement Parquet : {e}")

    elif source_type == "Google Sheets":
        gs_url = st.text_input("🔗 Lien public Google Sheets")
        if gs_url and "docs.google.com" in gs_url:
            try:
                sheet_id = gs_url.split("/d/")[1].split("/")[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                df = pd.read_csv(csv_url)
                st.success("✅ Données Google Sheets chargées")
            except Exception as e:
                st.error(f"Erreur Google Sheets : {e}")

    elif source_type == "Base de données SQL":
        st.markdown("Connexion à une base SQL (MySQL, PostgreSQL, SQL Server)")
        db_type = st.selectbox("Type", ["MySQL", "PostgreSQL", "SQL Server"])
        host = st.text_input("Hôte", value="localhost")
        port = st.text_input("Port", value={
            "MySQL": "3306",
            "PostgreSQL": "5432",
            "SQL Server": "1433"
        }[db_type])
        db_name = st.text_input("Nom de la base")
        user = st.text_input("Utilisateur")
        pwd = st.text_input("Mot de passe", type="password")
        requete = st.text_area("Requête SQL", "SELECT * FROM votre_table")

        if st.button("Se connecter et charger"):
            try:
                if db_type == "MySQL":
                    engine = sqlalchemy.create_engine(f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db_name}")
                elif db_type == "PostgreSQL":
                    engine = sqlalchemy.create_engine(f"postgresql://{user}:{pwd}@{host}:{port}/{db_name}")
                else:  # SQL Server
                    engine = sqlalchemy.create_engine(f"mssql+pyodbc://{user}:{pwd}@{host}:{port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server")
                df = pd.read_sql_query(requete, engine)


                # Convertir les colonnes texte vers UTF-8 en ignorant les erreurs
                for col in df.select_dtypes(include=["object"]).columns:
                    df[col] = df[col].apply(lambda x: x.encode("latin1").decode("utf-8", errors="ignore") if isinstance(x, str) else x)

                st.success("✅ Données SQL chargées")
            except Exception as e:
                st.error(f"Erreur de connexion SQL : {e}")

    elif source_type == "SQLite local":
        sqlite_file = st.file_uploader("📁 Importer un fichier SQLite (.db, .sqlite)", type=["db", "sqlite"])
        if sqlite_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(sqlite_file.read())
                    tmp_path = tmp_file.name
                conn = sqlite3.connect(tmp_path)
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                table_name = st.selectbox("Choisir une table", tables['name'].tolist())
                if st.button("Charger la table"):
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    st.success(f"✅ Table '{table_name}' chargée")
                    conn.close()
            except Exception as e:
                st.error(f"Erreur SQLite : {e}")

    elif source_type == "API REST (JSON)":
        api_url = st.text_input("URL de l'API REST retournant du JSON")
        if st.button("Charger les données"):
            try:
                r = requests.get(api_url)
                r.raise_for_status()
                data = r.json()
                # On essaye de convertir en DataFrame (extraction automatique)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.json_normalize(data)
                else:
                    st.error("Format JSON non supporté")
                    df = None
                if df is not None:
                    st.success("✅ Données API chargées")
            except Exception as e:
                st.error(f"Erreur API REST : {e}")

    elif source_type == "Google BigQuery":
        credentials_json = st.text_area("Clé JSON service account Google Cloud (BigQuery)", height=200)
        project_id = st.text_input("ID du projet Google Cloud")
        query = st.text_area("Requête SQL BigQuery")
        if st.button("Exécuter la requête BigQuery"):
            try:
                credentials = service_account.Credentials.from_service_account_info(json.loads(credentials_json))
                client_bq = bigquery.Client(credentials=credentials, project=project_id)
                df = client_bq.query(query).to_dataframe()
                st.success("✅ Requête BigQuery exécutée")
            except Exception as e:
                st.error(f"Erreur BigQuery : {e}")

    elif source_type == "MongoDB":
        mongo_uri = st.text_input("URI MongoDB (ex: mongodb://user:pwd@host:port/db)")
        db_name = st.text_input("Nom base MongoDB")
        collection_name = st.text_input("Nom collection")
        if st.button("Charger MongoDB"):
            try:
                client_mongo = MongoClient(mongo_uri)
                db_mongo = client_mongo[db_name]
                collection = db_mongo[collection_name]
                data = list(collection.find())
                df = pd.DataFrame(data)
                st.success("✅ Données MongoDB chargées")
            except Exception as e:
                st.error(f"Erreur MongoDB : {e}")

    elif source_type == "FTP / SFTP":
        ftp_type = st.selectbox("Protocole", ["FTP", "SFTP"])
        host = st.text_input("Hôte")
        port = st.text_input("Port", value="21" if ftp_type == "FTP" else "22")
        user = st.text_input("Utilisateur")
        pwd = st.text_input("Mot de passe", type="password")

        if ftp_type == "FTP":
            if st.button("Liste fichiers FTP"):
                try:
                    ftp = FTP()
                    ftp.connect(host, int(port))
                    ftp.login(user, pwd)
                    files = ftp.nlst()
                    ftp.quit()
                    file_choice = st.selectbox("Fichiers disponibles", files)
                    if st.button("Charger fichier FTP"):
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            ftp = FTP()
                            ftp.connect(host, int(port))
                            ftp.login(user, pwd)
                            ftp.retrbinary(f"RETR {file_choice}", tmp_file.write)
                            ftp.quit()
                            tmp_file_path = tmp_file.name
                        df = pd.read_csv(tmp_file_path)
                        st.success(f"✅ Fichier '{file_choice}' chargé depuis FTP")
                except Exception as e:
                    st.error(f"Erreur FTP : {e}")

        else:  # SFTP
            sftp_host = host
            sftp_port = int(port)
            sftp_user = user
            sftp_pwd = pwd
            if st.button("Liste fichiers SFTP"):
                try:
                    transport = paramiko.Transport((sftp_host, sftp_port))
                    transport.connect(username=sftp_user, password=sftp_pwd)
                    sftp = paramiko.SFTPClient.from_transport(transport)
                    files = sftp.listdir()
                    transport.close()
                    file_choice = st.selectbox("Fichiers disponibles", files)
                    if st.button("Charger fichier SFTP"):
                        transport = paramiko.Transport((sftp_host, sftp_port))
                        transport.connect(username=sftp_user, password=sftp_pwd)
                        sftp = paramiko.SFTPClient.from_transport(transport)
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            sftp.get(file_choice, tmp_file.name)
                            tmp_path = tmp_file.name
                        sftp.close()
                        transport.close()
                        df = pd.read_csv(tmp_path)
                        st.success(f"✅ Fichier '{file_choice}' chargé depuis SFTP")
                except Exception as e:
                    st.error(f"Erreur SFTP : {e}")

    elif source_type == "Google Drive (CSV / Excel)":
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive

        file_id = st.text_input("ID du fichier Google Drive (CSV ou Excel)")
        if st.button("Charger fichier Google Drive"):
            try:
                # Authentification automatique simplifiée (attention : config OAuth nécessaire)
                gauth = GoogleAuth()
                gauth.LocalWebserverAuth()
                drive = GoogleDrive(gauth)
                downloaded = drive.CreateFile({'id': file_id})
                downloaded.GetContentFile('tempfile')

                if file_id.endswith('.csv'):
                    df = pd.read_csv('tempfile')
                else:
                    df = pd.read_excel('tempfile')
                st.success("✅ Fichier Google Drive chargé")
            except Exception as e:
                st.error(f"Erreur Google Drive : {e}")

# --- TRAITEMENT DES DONNEES ---

if df is not None:
    try:
        df = df.dropna(axis=1, how="all")

        # Conversion automatique des dates
        colonnes_date = [col for col in df.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])]
        for col in colonnes_date:
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                df[f"{col}_année"] = df[col].dt.year
                df[f"{col}_mois"] = df[col].dt.month
            except Exception as e:
                st.warning(f"⚠️ Erreur conversion date sur {col} : {e}")

        st.success("✅ Données prêtes à l'analyse")
        st.dataframe(df.head())

        # Filtres dynamiques + Boutons
        st.subheader("🎛️ Filtres dynamiques (optionnels)")
        colonnes_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
        filtres = {}
        for col in colonnes_cat:
            valeurs = df[col].dropna().unique().tolist()
            selection = st.multiselect(f"Filtrer par {col}", valeurs)
            if selection:
                filtres[col] = selection

        col1, col2 = st.columns(2)
        btn_filtrer = col1.button("✅ Appliquer les filtres et analyser")
        btn_sans_filtre = col2.button("🚀 Analyser sans filtres")

        def appliquer_filtres(df, filtres):
            for col, valeurs in filtres.items():
                df = df[df[col].isin(valeurs)]
            return df

        def detecter_domaine(df):
            colonnes = [col.lower() for col in df.columns]
            if any("prix" in col or "ca" in col or "vente" in col for col in colonnes):
                return "ventes"
            elif any("note" in col or "classe" in col for col in colonnes):
                return "scolaire"
            elif any("salaire" in col or "poste" in col for col in colonnes):
                return "rh"
            elif any("maladie" in col or "patient" in col for col in colonnes):
                return "sante"
            elif any("montant" in col or "finance" in col for col in colonnes):
                return "finance"
            elif any("stock" in col or "logistique" in col for col in colonnes):
                return "logistique"
            elif any("campagne" in col or "clic" in col for col in colonnes):
                return "webmarketing"
            elif any("ticket" in col or "statut" in col for col in colonnes):
                return "it_support"
            elif any("avancement" in col or "projet" in col for col in colonnes):
                return "projets"
            else:
                return "autre"

        if btn_filtrer or btn_sans_filtre:
            if btn_filtrer:
                df_filtré = appliquer_filtres(df, filtres)
                domaine = detecter_domaine(df_filtré)
                st.info(f"🧠 Domaine détecté : **{domaine.upper()}** avec filtres")
                df_to_use = df_filtré
            else:
                domaine = detecter_domaine(df)
                st.info(f"🧠 Domaine détecté : **{domaine.upper()}** sans filtres")
                df_to_use = df

            # Affichage des KPIs
            afficher_kpis(df_to_use, domaine)

            # Suggestions IA (si activées)
            st.subheader("💡 Suggestions de graphiques basées sur l'IA")
            suggestions = generer_suggestions(df_to_use.head(ECHANTILLON_IA))
            if suggestions:
                for i in range(0, len(suggestions), 3):
                    colonnes = st.columns(3)
                    for j, sugg in enumerate(suggestions[i:i+3]):
                        with colonnes[j]:
                            st.markdown(f"**Suggestion {i+j+1} :** {sugg.get('objectif', 'Sans description')}")
                            afficher_graphique(df_to_use, sugg, colonnes[j])
            else:
                st.info("Aucune suggestion de graphique générée.")

            # Résumé automatique
            st.subheader("📝 Résumé automatique des données")
            resume = generer_resume(df_to_use.head(ECHANTILLON_IA))
            st.write(resume)

            # Sauvegarde session (optionnel)
            if "utilisateur" in st.session_state:
                sauvegarder_session(st.session_state.utilisateur, resume, suggestions)

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement des données : {e}")
else:
    st.info("Veuillez choisir une source de données pour commencer.")

st.info("⚠️ L'agent IA peut faire des erreurs. Réimportez les données pour améliorer l'analyse si besoin.")