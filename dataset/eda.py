import pandas as pd

def load_and_explore_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset and performs Exploratory Data Analysis (EDA).
    """
    print("--- 1. LOADING DATASET ---")
    df = pd.read_csv(filepath)
    print("Dataset successfully loaded!\n")
    
    print("--- 2. DATASET FORMAT & COLUMNS ---")
    print(f"Shape of the dataset: {df.shape[0]} rows (requirements) and {df.shape[1]} columns.\n")
    
    print("Columns Overview:")
    for col in df.columns:
        print(f" - {col}: ", end="")
        if col == "Requirement_ID":
            print("A unique identifier for each software requirement.")
        elif col == "Requirement_Text":
            print("The actual textual software requirement that needs to be analyzed for ambiguity.")
    
    print("\n--- 3. EXPLORATORY DATA ANALYSIS (EDA) ---")
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset Info (Checking for missing values and data types):")
    df.info()
    
    print("\nBasic Statistics:")
    # We will look at length of the requirements for some early numerical insight
    df['Text_Length'] = df['Requirement_Text'].apply(len)
    print(df['Text_Length'].describe())
    
    return df

if __name__ == "__main__":
    dataset_path = "dataset/software_requirements.csv"
    df = load_and_explore_data(dataset_path)
