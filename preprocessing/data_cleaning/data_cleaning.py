import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


scaler = MinMaxScaler(feature_range=[0, 1])


class DataCleaning:
    columns_to_keep = [
        'Host Since',
        'Host Response Time',
        'Is Superhost',
        'Neighborhood Group',
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
        'Instant Bookable'
    ]
    numerical_columns = [
        "Accomodates",
        "Bathrooms",
        "Bedrooms",
        "Beds",
        "Guests Included",
        "Min Nights"
    ]
    bool_columns = ["Is Superhost", "Instant Bookable", "Is Exact Location"]
    OH_columns = ["Property Type", "Room Type", "Neighborhood Group"]

    def __init__(self, path: str) -> None:
        self.path = path
        self.df = None
        self.is_dataset = True
        self.data_train = None
        self.data_test = None
        self.data_val = None
        self.encoded_columns = []

    def df_creation(self) -> None:
        """ Loading the DataFrame.
            Replacing '*' with NaN values.
            Dropping NaN value regarding the target column. """
        self.df = pd.read_csv(self.path)
        # Replacing * with nan
        self.df = self.df.replace("*", np.nan)
        # Dropping instances with NaN values in the target column
        if 'Price' in self.df.columns:
            self.df = self.df[DataCleaning.columns_to_keep + ['Price']]
            self.df = self.df.dropna(subset=['Price'])
        else:
            self.is_dataset = False
            self.df = self.df[DataCleaning.columns_to_keep]

    def to_one_hot(self) -> None:
        """ Applies OH encoding on the concerned columns. """
        for column in DataCleaning.OH_columns:
            one_hot = pd.get_dummies(self.df[column], prefix=column)
            self.encoded_columns += list(one_hot.columns)
            self.df = self.df.drop(column, axis=1)
            self.df = self.df.join(one_hot)

    def to_float(self) -> None:
        """ Converts to float the concerned columns. """
        for column in DataCleaning.numerical_columns:
            self.df[column] = self.df[column].astype(np.float64)

    def bool_to_numerical(self) -> None:
        """ Converts to numerical values the ordinal categorical columns. """
        for column in self.bool_columns:
            self.df[column] = self.df[column].replace("t", 1).replace("f", 0)

    def date_to_numerical(self, column: str = "Host Since") -> None:
        """ Dates become the difference in terms of days with the most
            recent one. """
        try:
            df = pd.to_datetime(self.df[column], format="%Y-%m-%d")
        except:
            df = pd.to_datetime(self.df[column], format="%d-%m-%Y")
        res = min(df)
        df = df - res
        df = df.dt.days.astype(np.float64)
        self.df[column] = df

    def host_response_time(self) -> None:
        is_nan = self.df['Host Response Time'].isna()
        self.df['Host Response Time'] = is_nan.replace(
            False, 0).replace(True, 1)

    def feature_radius(self) -> None:
        """ Adding a feature: the radius from the most dense part of Berlin
            in terms of the quantity of airbnb flats in the area. """
        mean_latitude = sum(self.df['Latitude'])/len(self.df['Latitude'])
        mean_longitude = sum(self.df['Longitude'])/len(self.df['Longitude'])
        self.df['dist_to_center'] = np.sqrt(
            (self.df['Longitude']-mean_longitude)**2 +
            (self.df['Latitude']-mean_latitude)**2
        )
        # self.df = self.df.drop(columns=['Latitude', 'Longitude'], axis=1)

    def dropping_nan_values(self) -> None:
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
            'Instant Bookable'
        ]
        self.df = self.df.dropna(subset=columns)

    def imputation(self, strategy: str = "stochastic") -> None:
        """ Imputation for the concerned columns.
            The Imputation concerns the following columns:
            ['Overall Rating',
            'Accuracy Rating',
            'Cleanliness Rating',
            'Checkin Rating',
            'Communication Rating',
            'Location Rating',
            'Value Rating'].
            For these columns, the values can be any int number from 0 to 10
            except for Overall Rating (between 0 and 100).
            Attribute:
                - strategy : the imputation strategy. Must be in ['stochastic', None] """

        if strategy == "stochastic":
            columns = self.df.columns

            # Stochastic regression to imput missing values
            it_imp = IterativeImputer(sample_posterior=True)
            imputed_df = it_imp.fit_transform(self.df)
            self.df = pd.DataFrame(imputed_df, columns=columns)

            # Cleaning obtained result to get only int values between 0 and 10
            rating_columns = ['Accuracy Rating', 'Cleanliness Rating', 'Checkin Rating',
                              'Communication Rating', 'Location Rating', 'Value Rating']
            self.df[rating_columns] = self.df[rating_columns].astype(int)
            for column in rating_columns:
                self.df[column] = self.df[column].apply(
                    lambda x: x if x <= 10 else 10
                )

            # Values are between 0 and 100 for this variable
            self.df['Overall Rating'] = self.df['Overall Rating'].astype(int)
            self.df['Overall Rating'] = self.df['Overall Rating'].apply(
                lambda x: x if x <= 100 else 100
            )

    def delete_price_outliers(self, max_price: float) -> None:
        if self.is_dataset:
            self.df = self.df[self.df["Price"] < max_price]

    def train_test_splitting(self) -> None:
        # Only one instance has a value of 16 in Accomodates column:
        # we delete to make the stratification possible.
        whole_data_train, self.data_test = train_test_split(
            self.df,
            test_size=0.2,
            random_state=42, shuffle=True
        )
        self.data_train, self.data_val = train_test_split(
            whole_data_train,
            test_size=0.2,
            random_state=42, shuffle=True
        )

    def scaling(self) -> None:
        """ Scaling the data. Method chosen is Min Max because most of the
        features are not Gaussian and we have a lot of feature OH encoded
        which have as values 0 or 1. """
        if self.is_dataset:
            # Saving column "Price"
            train_price_column = self.data_train["Price"]
            # Reseting index of "Price"
            train_price_column = train_price_column.reset_index(drop=True)
            # Deleting column "Price"
            self.data_train.drop(["Price"], axis=1, inplace=True)
            # Scaling all the feature
            self.data_train = pd.DataFrame(scaler.fit_transform(
                self.data_train), columns=self.data_train.columns)
            # Replacing scaled "Price" with the former values
            self.data_train["Price"] = train_price_column

            val_price_column = self.data_val["Price"]
            val_price_column = val_price_column.reset_index(drop=True)
            self.data_val.drop(["Price"], axis=1, inplace=True)
            self.data_val = pd.DataFrame(scaler.transform(
                self.data_val), columns=self.data_val.columns)
            self.data_val["Price"] = val_price_column
            test_price_column = self.data_test["Price"]
            test_price_column = test_price_column.reset_index(drop=True)
            self.data_test.drop(["Price"], axis=1, inplace=True)
            self.data_test = pd.DataFrame(scaler.transform(
                self.data_test), columns=self.data_test.columns)
            self.data_test["Price"] = test_price_column
        else:
            self.df = pd.DataFrame(scaler.transform(
                self.df), columns=self.df.columns)

    def save_csv(self, df: pd.DataFrame, csv_name: str) -> None:
        """ Saves the cleaned dataframe in a csv file in the same folder as the
            one containing the not-cleaned dataframe.
            Attribute:
                - csv_name : name of the csv file created """
        df.to_csv(os.path.join(self.path.rsplit(
            "/", 1)[0], csv_name), index=False)

    def data_cleaning(self, imputation_strategy: str = "stochastic", max_price: float = 1000.0) -> None:
        """ Main method: applies all the preprocesses. """
        # Loading the dataframe from the given path
        self.df_creation()
        # Dropping NaN values for some columns
        self.dropping_nan_values()
        # OH for some columns
        self.to_one_hot()
        # 0 / 1 encoding for some columns
        self.bool_to_numerical()
        # Dealing with dates
        self.date_to_numerical()
        # Dealing with Host Response Time column
        self.host_response_time()
        # Creating a new feature
        self.feature_radius()
        # Conversion to float
        self.to_float()
        # Imputing missing values for some columns
        self.imputation(strategy=imputation_strategy)
        # Deleting "Price" outliers given a certain max price
        self.delete_price_outliers(max_price=max_price)
        if self.is_dataset:
            self.save_csv(self.df, csv_name="train_airbnb_berlin_cleaned.csv")
            # Splitting into Train - Val - Test
            self.train_test_splitting()
            # Scaling the features
            self.scaling()
            # Saving the dataframes in csv files
            self.save_csv(self.data_train, csv_name="data_train.csv")
            self.save_csv(self.data_val, csv_name="data_val.csv")
            self.save_csv(self.data_test, csv_name="data_test.csv")
        else:
            self.scaling()
            self.save_csv(self.df, csv_name="test_airbnb_berlin_cleaned.csv")


train_path = "/Users/cha/Desktop/3A/code/ml-project/data/train_airbnb_berlin.csv"
test_path = "/Users/cha/Desktop/3A/code/ml-project/data/test_airbnb_berlin.csv"

data_cleaner = DataCleaning(path=train_path)
data_cleaner.data_cleaning()
data_cleaner = DataCleaning(path=test_path)
data_cleaner.data_cleaning()
