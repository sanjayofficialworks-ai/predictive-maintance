import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_and_preprocess(csv_path):

    df = pd.read_csv(csv_path)

    # Feature engineering
    df['Temp_diff [K]'] = (
        df['Process temperature [K]']
        - df['Air temperature [K]']
    )

    df['Mechanical Power [W]'] = (
        df['Torque [Nm]']
        * df['Rotational speed [rpm]']
        * 2 * np.pi / 60
    )

    # Remove identifier columns
    df = df.drop(columns=['UDI', 'Product ID'])

    # Binary target
    x = df.drop(columns=[
        'Machine failure',
        'TWF',
        'HDF',
        'PWF',
        'OSF',
        'RNF'
    ])

    y = df['Machine failure']

    # Multi-label targets
    y_multi = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Use the SAME indices to split multi-label targets
    y_train_multi = y_multi.loc[x_train.index]
    y_test_multi = y_multi.loc[x_test.index]

    # Feature types
    categorical_features = ['Type']

    numerical_features = x_train.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    # Encoding
    type_encoder = OrdinalEncoder(
        categories=[['L', 'M', 'H']]
    )

    # Scaling
    scaler = StandardScaler()

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('type', type_encoder, categorical_features),
            ('num', scaler, numerical_features)
        ]
    )

    # Fit only on training data
    x_train_processed = preprocessor.fit_transform(x_train)

    # Transform test data using training preprocessing
    x_test_processed = preprocessor.transform(x_test)

    return (
        x_train_processed,
        x_test_processed,
        y_train,
        y_test,
        preprocessor,
        y_train_multi,
        y_test_multi
    )