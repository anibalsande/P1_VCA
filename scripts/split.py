import pandas as pd
from sklearn.model_selection import train_test_split

def get_stratified_indexes(csv_path, test_size=0.2, seed=42):
    """
    Lee el CSV y devuelve índices estratificados para Train y Test.
    """
    # Cargamos el CSV solo para sacar las etiquetas
    df = pd.read_csv(csv_path, sep=";")
    labels = df.iloc[:, 1].values 
    
    all_index = list(range(len(df)))
    train_idx, test_idx = train_test_split(
        all_index, 
        test_size=test_size, 
        stratify=labels, 
        random_state=seed
    )
    
    return train_idx, test_idx