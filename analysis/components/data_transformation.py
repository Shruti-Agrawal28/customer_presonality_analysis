from analysis.entity import artifact_entity, config_entity
from analysis.exception import SensorException
from analysis.logger import logging
from typing import Optional
import os, sys
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from analysis import utils
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from analysis.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


class DataTransformation:

    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>' * 20} Data Transformation {'<<' * 20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:

            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps=[
                ('RobustScaler', robust_scaler)
            ])
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def preprocess_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info(f"Adding necessary columns")
            df["Age"] = 2023 - df["Year_Birth"]
            df["Money_Spent"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + \
                                df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
            df["children"] = df["Kidhome"] + df["Teenhome"]
            df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
            df["Enrollment_days"] = df["Dt_Customer"] - df["Dt_Customer"].min()
            df['Income'] = df['Income'].fillna(df['Income'].mean())
            delta = pd.Timedelta(days=365)
            df["Enrollment_duration"] = df["Enrollment_days"] / delta
            df["Living_With"] = df["Marital_Status"].replace(
                {"Married": "Partner", "Together": "Partner", "Absurd": "Alone",
                 "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone", })
            df["Family_Size"] = df["Living_With"].replace({"Alone": 1, "Partner": 2}) + df["children"]
            df["Education"] = df["Education"].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate",
                                                       "Graduation": "Graduate", "Master": "Postgraduate",
                                                       "PhD": "Postgraduate"})
            drop_it = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID",
                       "Enrollment_days"]
            df = df.drop(drop_it, axis=1)
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
            for i in df.columns:
                if i in categorical_features:
                    df[i] = le.fit_transform(df[i])

            cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain',
                        'Response']
            df = df.drop(cols_del, axis=1)

            scaler = StandardScaler()
            scaler.fit(df)
            scaled_ds = pd.DataFrame(scaler.transform(df), columns=df.columns)
            logging.info(df)
            pca = PCA(n_components=4)
            pca.fit(scaled_ds)
            PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1", "col2", "col3", 'col4']))
            AC = AgglomerativeClustering(n_clusters=4)
            # fit model and predict clusters
            yhat_AC = AC.fit_predict(PCA_ds)
            PCA_ds["Clusters"] = yhat_AC
            df["Clusters"] = yhat_AC
            return df

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_transformation(self, ) -> artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)
            df = DataTransformation.preprocess_data(df=df)
            print(df)

            logging.info("split dataset into train and test set")
            # split dataset into train and test set
            train_df, test_df = train_test_split(df, test_size=self.data_transformation_config.test_size, random_state=42)

            logging.info("create dataset directory folder if not available")
            # create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_transformation_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info("Save df to feature store folder")
            # Save df to feature store folder
            train_df.to_csv(path_or_buf=self.data_transformation_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_transformation_config.test_file_path, index=False, header=True)

            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]


            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            # transforming input features
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)


            # target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            # save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                              obj=transformation_pipleine)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                train_file_path = self.data_transformation_config.train_file_path,
                test_file_path = self.data_transformation_config.test_file_path

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)
