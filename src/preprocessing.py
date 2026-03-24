import pandas as pd
import os


def load_data(file_path):
    """
    Load raw dataset
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """
    Basic cleaning:
    - Convert timestamp
    - Handle missing values
    """
    # Rename first column if needed
    if df.columns[0].lower() != "timestamp":
        df = df.rename(columns={df.columns[0]: "timestamp"})

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort
    df = df.sort_values("timestamp")

    # Handle missing values (simple forward fill)
    df = df.fillna(method="ffill")

    return df


def reshape_data(df):
    """
    Convert wide → long format
    """
    df_long = df.melt(
        id_vars=["timestamp"],
        var_name="meter_id",
        value_name="consumption"
    )

    return df_long


def save_data(df, output_path):
    """
    Save processed data
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example execution

    input_path = "data/raw/data.csv"
    output_path = "data/processed/processed_data.csv"

    df = load_data(input_path)
    df = clean_data(df)
    df = reshape_data(df)

    save_data(df, output_path)

    print(" Data preprocessing completed. File saved!")
