import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "NHIJOBIO") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """
    # tidy headers
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace

    df = df[['EDAD', 'EC', 'PAGOVI','ESTUDIOSA','INGRESOS','NHOGAR','TRABAJAACT','INGREHOG_INTER','NHIJOBIO']]

    df = df.rename(columns={
        'EDAD': 'age',
        'EC': 'married',
        'PAGOVI': 'housing_payment',
        'ESTUDIOSA': 'level_studies',
        'INGRESOS': 'income',
        'NHOGAR': 'number_living',
        'TRABAJAACT': 'job',
        'INGREHOG_INTER': 'house_income',
        'NHIJOBIO': 'nr_children'
    })

    # Binary definition for binary cols

    df["married"] = df["married"].isin([2, 3, 5]).astype(int)
    df['job'] = df["job"].isin([1]).astype(int)

    # Transformation to multi values cols - reagrupping information

    mapping = {1: 1, 2: 1, 3: 2,4: 2,5: 2,6: 3, 7: 3, 8: 4, 9: 4}
    df["level_studies"] = df["level_studies"].map(mapping)

    df["number_living"] = (df["number_living"] < 4).astype(int)

    mapping = {1: 1, 2: 1, 3: 1,4: 2,5: 2,6: 3, 7: 3, 8: 3, 9: 3}
    df["income"] = df["income"].map(mapping)

    mapping = {1: 1, 2: 1, 3: 1,4: 2,5: 2,6: 3, 7: 3, 8: 3}
    df["house_income"] = df["house_income"].map(mapping)

    mapping = {0: 1, 1: 1, 2: 2,3: 2,4: 2,5: 3}
    df["housing_payment"] = df["housing_payment"].map(mapping)


    df["age"] = (df["age"] >= 30).astype(int)

    ## We are also transforming 'target' column as we just want to predict childlesness

    df["nr_children"] = (df["nr_children"] > 0).astype(int)


    # simple NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle (get_dummies ignores NaN safely)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df
