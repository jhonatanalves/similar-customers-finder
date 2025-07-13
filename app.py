import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

st.set_page_config(layout="wide", page_title="Clientes Similares")

st.title("🔍 Identificador de Clientes Similares ao top K")
st.markdown("Visualize os clientes regulares com maior similaridade com os clientes ativos com base em **faturamento** e **frequência de uso da plataforma**.")

# Upload do CSV
uploaded_file = st.file_uploader("📁 Envie um arquivo CSV com os dados dos clientes", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = {"name", "revenue", "frequency", "type"}
    if not required_cols.issubset(df.columns):
        st.error(f"O arquivo precisa conter as colunas: {', '.join(required_cols)}")
    else:
        # Separar clientes ativos e regulares
        active_df = df[df['type'] == 'active'].copy()
        regular_df = df[df['type'] == 'regular'].copy()

        if active_df.empty or regular_df.empty:
            st.warning("É necessário ter pelo menos um cliente ativo e um regular no arquivo.")
        else:
            st.sidebar.header("🔧 Configurações")

            # Escolha de k
            k = st.sidebar.slider("Número de clientes mais similares (k)", min_value=1, max_value=min(20, len(regular_df)), value=5)

            # Escolha da métrica
            distance_metric = st.sidebar.selectbox("📐 Métrica de distância", ["euclidiana", "manhattan", "minkowski"])

            if distance_metric == "euclidiana":
                metric = "minkowski"
                p = 2
            elif distance_metric == "manhattan":
                metric = "manhattan"
                p = 1
            else:
                metric = "minkowski"
                p = st.sidebar.slider("Valor de p (Minkowski)", min_value=1, max_value=5, value=3)

            # Normalização dos dados
            scaler = StandardScaler()
            all_features = pd.concat([
                active_df[['revenue', 'frequency']],
                regular_df[['revenue', 'frequency']]
            ])
            scaled = scaler.fit_transform(all_features)

            active_scaled = scaled[:len(active_df)]
            regular_scaled = scaled[len(active_df):]

            # Treinar KNN não supervisionado
            knn = NearestNeighbors(n_neighbors=len(active_df), metric=metric, p=p)
            knn.fit(active_scaled)

            # Calcular distâncias para cada cliente regular em relação aos ativos
            distances, _ = knn.kneighbors(regular_scaled)
            mean_distances = distances.mean(axis=1)

            regular_df["mean_distance"] = mean_distances
            top_k_similars = regular_df.nsmallest(k, "mean_distance")

            # Visualização com Plotly
            st.subheader("📈 Visualização 2D (Faturamento x Frequência)")
            fig = px.scatter(
                df,
                x="frequency",
                y="revenue",
                color="type",
                text="name",
                color_discrete_map={"active": "blue", "regular": "gray"},
                labels={"frequency": "Frequência", "revenue": "Faturamento"},
                title="Distribuição dos Clientes"
            )

            # Destacar os clientes similares
            fig.add_scatter(
                x=top_k_similars["frequency"],
                y=top_k_similars["revenue"],
                mode="markers+text",
                name="Similares",
                text=top_k_similars["name"],
                textposition="top center",
                marker=dict(color="green", size=12, line=dict(width=2, color='darkgreen'))
            )

            st.plotly_chart(fig, use_container_width=True)

            # Resultados detalhados
            st.subheader("📋 Top k Clientes Mais Similares")
            st.dataframe(top_k_similars[["name", "revenue", "frequency", "mean_distance"]].reset_index(drop=True))
            
else:
    st.info("Envie um arquivo CSV com colunas: `name`, `revenue`, `frequency`, `type` (`active` ou `regular`).")