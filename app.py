import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

st.set_page_config(layout="wide", page_title="Clientes Similares")

st.title("🔍 Identificador de Clientes Similares")
st.markdown("Visualize os clientes regulares com maior similaridade com clientes-alvo com base em **faturamento** e **frequência de uso da plataforma**.")

# CSV upload
uploaded_file = st.file_uploader("📁 Envie um arquivo CSV com os dados dos clientes", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = {"name", "revenue", "frequency", "type"}
    if not required_cols.issubset(df.columns):
        st.error(f"O arquivo precisa conter as colunas: {', '.join(required_cols)}")
    else:
        # Separate target and regular customers
        target_df = df[df['type'] == 'target'].copy()
        regular_df = df[df['type'] == 'regular'].copy()

        if target_df.empty or regular_df.empty:
            st.warning("É necessário ter pelo menos um cliente-alvo e um regular no arquivo.")
        else:
            st.sidebar.header("🔧 Configurações")

            # Choose k
            k = st.sidebar.slider("Número de clientes mais similares (k)", min_value=1, max_value=min(20, len(regular_df)), value=5)

            # Select distance metric
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

            # Normalize the data
            scaler = StandardScaler()
            all_features = pd.concat([
                target_df[['revenue', 'frequency']],
                regular_df[['revenue', 'frequency']]
            ])
            scaled = scaler.fit_transform(all_features)

            target_scaled = scaled[:len(target_df)]
            regular_scaled = scaled[len(target_df):]

            # Adjust model to target customers
            knn = NearestNeighbors(n_neighbors=len(target_df), metric=metric, p=p)
            knn.fit(target_scaled)

            # Calculate distances from each regular customer to the target customers
            distances, _ = knn.kneighbors(regular_scaled)
            mean_distances = distances.mean(axis=1)

            regular_df["mean_distance"] = mean_distances
            top_k_similars = regular_df.nsmallest(k, "mean_distance")

            # Visualization with Plotly
            st.subheader("📈 Visualização - Faturamento x Frequência")
            non_similar_df = regular_df.drop(top_k_similars.index)
            plot_df = pd.concat([target_df, non_similar_df])

            fig = px.scatter(
                plot_df, # Usa o novo dataframe filtrado
                x="frequency",
                y="revenue",
                color="type",
                text="name",
                color_discrete_map={"target": "blue", "regular": "gray"},
                labels={"frequency": "Frequência", "revenue": "Faturamento"},
                title="Distribuição dos Clientes"
            )

            # Highlight similar customers
            fig.add_scatter(
                x=top_k_similars["frequency"],
                y=top_k_similars["revenue"],
                mode="markers+text",
                name="Similares", # Legenda para os pontos verdes
                text=top_k_similars["name"],
                textposition="top center",
                marker=dict(color="green", size=12, line=dict(width=2, color='darkgreen'))
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detail results
            st.subheader("📋 Top k Clientes Mais Similares")
            st.dataframe(top_k_similars[["name", "revenue", "frequency", "mean_distance"]].reset_index(drop=True))
            
else:
    st.info("Envie um arquivo CSV com colunas: `name`, `revenue`, `frequency`, `type` (`target` ou `regular`).")