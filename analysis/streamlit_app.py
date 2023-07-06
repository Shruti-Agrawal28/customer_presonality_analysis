import streamlit as st
import pandas as pd
from analysis.entity import artifact_entity
import plotly.figure_factory as ff
import plotly.express as px


def custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
            font-family: Arial, sans-serif;
        }
        h1 {
            color: #333333;
        }
        .stApp {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


class CustomerSegmentationApp:
    def __init__(self, data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        self.data_transformation_artifact = data_transformation_artifact
        self.data = None

    def import_data(self):
        self.data = pd.read_csv(self.data_transformation_artifact.train_file_path)

    def display_cluster_distribution(self):
        cluster_counts = self.data['Clusters'].value_counts().sort_index()
        st.subheader("Cluster Distribution")
        grouped_data = self.data.groupby('Clusters').mean()

        # Adding the money spent group column based on the money spent values
        grouped_data['Money Spent Group'] = pd.cut(grouped_data['Money_Spent'], bins=[0, 500, 1000,2000, float('inf')],
                                                   labels=['<500', '500-1000', '1000-2000', '2000+'])

        #grouped_data['Cluster'] = grouped_data.index.map(lambda cluster: f"Cluster {cluster}")

        # Reorder the columns for better readability
        cluster_table = grouped_data[['Money Spent Group']].reset_index()

        # Render the cluster distribution table using st.dataframe
        st.dataframe(cluster_table)

    def visualize_clusters(self, features):
        st.subheader("Customer Segmentation")
        custom_css()
        grouped_data = self.data.groupby(self.data['Clusters']).mean()

        # Create the scatter plot using plotly express
        fig = px.scatter(
            grouped_data,
            x=self.data['Money_Spent'],
            y=self.data['Age'],
            color=self.data['Clusters'],
            size=self.data['Education'],
            labels={'Money_Spent': 'Money Spent', 'Age': 'Age', 'Clusters': 'Cluster', 'Education': 'Education'},
            title='Cluster Distribution'
        )
        fig.update_xaxes(range=[0, 3000])
        fig.update_yaxes(range=[0, 100])

        fig.update_layout(
            xaxis_title="Money_Spent",
            yaxis_title="Age",
            coloraxis_colorbar_title="Cluster",
            title="Cluster Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)



