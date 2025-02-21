import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


df = pd.read_csv('Australian_Vehicle_Prices.csv')


df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price']).reset_index(drop=True)


train_df, test_df = train_test_split(df, test_size=0.2, random_state=80)


class FilterCarType(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.replacement_dict = {
            'Hatchback': 'Car',
            'Sedan': 'Car',
            'Wagon': 'Car',
            'Coupe': 'Car',
            'Convertible': 'Car',
            'SUV': 'Suv'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Car/Suv'] = X['Car/Suv'].replace(self.replacement_dict)
        X['Car/Suv'] = X['Car/Suv'].apply(self.classify_car_type)
        return X

    def classify_car_type(self, value):
        if isinstance(value, str):
            if 'SUV' in value.upper():
                return 'Suv'
            elif any(keyword.lower() != 'suv' and keyword.lower() in value.lower()
                     for keyword in ['Car'] + list(self.replacement_dict.keys())):
                return 'Car'
        return 'Unknown'


class ExtractEngineInfo(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract_engine_info(engine_str):
            cylinders = None
            liters = None
            if isinstance(engine_str, str):
                cyl_match = re.search(r'(\d+)\s*cyl', engine_str)
                if cyl_match:
                    cylinders = int(cyl_match.group(1))
                liters_match = re.search(r'(\d+\.?\d*)\s*L', engine_str)
                if liters_match:
                    liters = float(liters_match.group(1))
            return pd.Series([cylinders, liters])

        X[['Cylinders', 'Liters']] = X['Engine'].apply(extract_engine_info)
        X = X.drop(columns=['Engine'])
        return X


class ExtractSeats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Seats'] = X['Seats'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else None)
        return X


class FillUnknown(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Car/Suv'] = X.apply(self.fill_unknown, axis=1)
        return X


    def fill_unknown(self, row):
        if row['Car/Suv'] == 'Unknown':
            return 'Suv' if row['Seats'] >= 7 else 'Car'
        return row['Car/Suv']


class ExtractFuelConsumption(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['FuelConsumption'] = X['FuelConsumption'].apply(self.extract_fuel_consumption)
        median_value = X['FuelConsumption'].median()
        X['FuelConsumption'] = X['FuelConsumption'].replace(0.0, median_value)
        X['FuelConsumption'] = pd.to_numeric(X['FuelConsumption'], errors='coerce')
        X['FuelConsumption'] = X['FuelConsumption'].fillna(median_value)
        return X

    def extract_fuel_consumption(self, value):
        if isinstance(value, str):
            match = re.search(r'(\d+\.?\d*)\s*L\s*/\s*100\s*km', value)
            if match:
                return float(match.group(1))

        elif isinstance(value, (int, float)):
            return float(value)

        return None


class GroupFuelTypes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fuel_mapping = {
            'Unleaded': 'Unleaded',
            'Diesel': 'Diesel',
            'Premium': 'Premium',
            'Hybrid': 'Other',
            'Electric': 'Other',
            'Other': 'Other',
            'LPG': 'Other',
            'Leaded': 'Other'
        }
        X['FuelType'] = X['FuelType'].map(fuel_mapping)
        return X


class CategorizeYear(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Year'] = pd.cut(
            X['Year'],
            bins=[-float('inf'), 2000, 2010, 2020, float('inf')],
            labels=['before_2000', '2000_to_2010', '2010_to_2020', 'after_2020']
        )
        return X


numeric_features = ['Price', 'Cylinders', 'Liters', 'Seats']
categorical_features = ['Car/Suv', 'FuelType', 'Year']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=False
)


pipeline = Pipeline(steps=[
    ('filter_car_type', FilterCarType()),
    ('extract_engine_info', ExtractEngineInfo()),
    ('extract_seats', ExtractSeats()),
    ('fill_unknown', FillUnknown()),
    ('extract_fuel_consumption', ExtractFuelConsumption()),
    ('group_fuel_types', GroupFuelTypes()),
    ('categorize_year', CategorizeYear()),
    ('preprocessor', preprocessor)
])


train_df_preprocessed = pipeline.fit_transform(train_df)
test_df_preprocessed = pipeline.transform(test_df)


def generate_columns(names_of_features, numeric, categorical, prefixes):
    new_columns = numeric.copy()
    for col in categorical:
        prefix = prefixes.get(col, '')
        new_columns += [f"{prefix}{name.split('_')[-1] if 'Year' not in col else name}"
                        for name in names_of_features if name.startswith(col)]
    return new_columns


feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

custom_prefixes = {
    'Car/Suv': 'Car_Type_',
    'FuelType': 'Fuel_Type_',
    'Year': ''
}


new_columns = generate_columns(feature_names, numeric_features, categorical_features, custom_prefixes)

train_df_preprocessed = pd.DataFrame(train_df_preprocessed, columns=new_columns)
test_df_preprocessed = pd.DataFrame(test_df_preprocessed, columns=new_columns)


print("Training data after preprocessing")
print(train_df_preprocessed.head())

print("Test data after preprocessing")
print(test_df_preprocessed.head())

