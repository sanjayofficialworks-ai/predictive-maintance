import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_and_preprocess(csv_path):

    df = pd.read_csv(csv_path)

    df['Temp_diff [K]'] = (
        df['Process temperature [K]']
        - df['Air temperature [K]']
    )

    df['Mechanical Power [W]'] = (
        df['Torque [Nm]']
        * df['Rotational speed [rpm]']
        * 2 * np.pi / 60
    )

    df = df.drop(columns=['UDI', 'Product ID'])

    x = df.drop(columns=[
        'Machine failure',
        'TWF',
        'HDF',
        'PWF',
        'OSF',
        'RNF'
    ])

    y = df['Machine failure']

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    categorical_features = ['Type']

    numerical_features = x_train.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    type_encoder = OrdinalEncoder(
        categories=[['L', 'M', 'H']]
    )

    scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('type', type_encoder, categorical_features),
            ('num', scaler, numerical_features)
        ]
    )

    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    return (
        x_train_processed,
        x_test_processed,
        y_train,
        y_test,
        preprocessor
    )