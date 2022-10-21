import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class DataCleaning:
    def __init__(self, path: str) -> None:
        self.path = path
        self.df = None
        self.encoded_columns = []
        self.columns_to_keep = [
            'Host Since',
            'Is Superhost',
            'Latitude',
            'Longitude',
            'Is Exact Location',
            'Property Type',
            'Room Type',
            'Accomodates',
            'Bathrooms',
            'Bedrooms',
            'Beds',
            'Guests Included',
            'Min Nights',
            'Reviews',
            'Overall Rating',
            'Accuracy Rating',
            'Cleanliness Rating',
            'Checkin Rating',
            'Communication Rating',
            'Location Rating',
            'Value Rating',
            'Instant Bookable',
            'Business Travel Ready'
        ]
        self.target_column_name = 'Price'
        self.numerical_columns = [
            "Accomodates",
            "Bathrooms",
            "Bedrooms",
            "Beds",
            "Guests Included",
            "Min Nights"
        ]
        self.columns_to_impute = [
            'Overall Rating',
            'Accuracy Rating',
            'Cleanliness Rating',
            'Checkin Rating',
            'Communication Rating',
            'Location Rating',
            'Value Rating'
        ]
        self.bool_columns = ["Is Superhost", "Instant Bookable", "Business Travel Ready", "Is Exact Location"]
        self.categorical_columns = ["Property Type", "Room Type"]

    def df_creation(self):
        """ Loading the DataFrame.
            Replacing '*' with NaN values.
            Dropping NaN value regarding the target column. """
        self.df = pd.read_csv(self.path)
        # Replacing * with nan and dropping instances with NaN values in the column Price
        self.df = self.df.replace("*", np.nan)
        if self.target_column_name in self.df.columns:
            self.df = self.df[self.columns_to_keep + [self.target_column_name]]
            self.df = self.df.dropna(subset=[self.target_column_name])
        else:
            self.df = self.df[self.columns_to_keep]

    def to_one_hot(self):
        """ Applies OH encoding on the concerned columns. """
        for column in self.categorical_columns:
            one_hot = pd.get_dummies(self.df[column], prefix=column)
            self.encoded_columns += list(one_hot.columns)
            self.df = self.df.drop(column, axis=1)
            self.df = self.df.join(one_hot)

    def to_float(self):
        """ Converts to float the concerned columns. """
        for column in self.numerical_columns:
            self.df[column] = self.df[column].astype(np.float64)

    def bool_to_numerical(self):
        """ Converts to numerical values the ordinal categorical columns. """
        for column in self.bool_columns:
            self.df[column] = self.df[column].replace("t", 1).replace("f", 0)

    def date_to_numerical(self, column:str = "Host Since"):
        """ Dates become the difference in terms of days with the most recent one. """
        try:
            df = pd.to_datetime(self.df[column], format="%Y-%m-%d")
        except:
            df = pd.to_datetime(self.df[column], format="%d-%m-%Y")
        res = min(df)
        df = df - res
        df = df.dt.days.astype(np.float64)
        self.df[column] = df

    def feature_radius(self):
        """ Adding a feature: the radius from the most dense part of Berlin
            in terms of the quantity of airbnb flats in the area. """
        mean_latitude = sum(self.df['Latitude'])/len(self.df['Latitude'])
        mean_longitude = sum(self.df['Longitude'])/len(self.df['Longitude'])
        self.df['dist_to_center'] = np.sqrt(
            (self.df['Longitude']-mean_longitude)**2 +(self.df['Latitude']-mean_latitude)**2
        )
        self.df = self.df.drop(columns=['Latitude', 'Longitude'], axis=1)

    def standard_scaling(self):
        """ Scaling the data. """
        columns = list(self.df.columns)
        columns_to_scale = columns
        for column in self.bool_columns + self.encoded_columns:
            columns_to_scale.remove(column)
        self.df[columns_to_scale] = pd.DataFrame(
            scale(self.df[columns_to_scale]),
            columns=columns_to_scale
        )

    def dropping_nan_values(self):
        """ Dropping NaN values for the concerned columns. """
        columns = [
            'Host Since',
            'Is Superhost',
            'Latitude',
            'Longitude',
            'Is Exact Location',
            'Property Type',
            'Room Type',
            'Accomodates',
            'Bathrooms',
            'Bedrooms',
            'Beds',
            'Guests Included',
            'Min Nights',
            'Instant Bookable',
            'Business Travel Ready'
        ]
        self.df = self.df.dropna(subset=columns)

    def imputation(self, strategy: str):
        """ Imputation for the concerned columns.
            Attribute:
                - strategy : the imputation strategy. Must be in ['stochastic'] """
        if strategy == "stochastic":
            columns = self.df.columns
            it_imp = IterativeImputer(sample_posterior=True)
            imputed_df = it_imp.fit_transform(self.df)
            self.df = pd.DataFrame(imputed_df, columns=columns)
    
    def save_csv(self, csv_name:str):
        """ Saves the cleaned dataframe in a csv file in the same folder as the
            one containing the not-cleaned dataframe.
            Attribute:
                - csv_name : name of the csv file created """
        self.df.to_csv(os.path.join(self.path.rsplit("/", 1)[0], csv_name), index=False)

    def data_cleaning(self, imputation_strategy: str = "stochastic", csv_name: str = "train_airbnb_berlin_cleaned.csv") -> pd.DataFrame:
        """ Main method: applies all the preprocesses. """
        self.df_creation()
        self.dropping_nan_values()
        self.to_one_hot()
        self.bool_to_numerical()
        self.date_to_numerical()
        self.feature_radius()
        self.to_float()
        self.imputation(strategy=imputation_strategy)
        self.standard_scaling()
        self.save_csv(csv_name=csv_name)

data_cleaner = DataCleaning(path="/Users/cha/Desktop/3A/code/ml-project/data/train_airbnb_berlin.csv")
data_cleaner.data_cleaning(csv_name="train_airbnb_berlin_cleaned.csv")
data_cleaner = DataCleaning(path="/Users/cha/Desktop/3A/code/ml-project/data/test_airbnb_berlin.csv")
data_cleaner.data_cleaning(csv_name="test_airbnb_berlin_cleaned.csv")
