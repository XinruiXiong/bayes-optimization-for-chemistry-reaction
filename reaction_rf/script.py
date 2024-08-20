import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 读取数据集
file_path = '/mnt/petrelfs/xiongxinrui/reaction_rf/data.csv'
df = pd.read_csv(file_path)

# 拆分reaction_smiles列
def split_reaction_smiles(smiles):
    reactants, product = smiles.split('>>')
    reactants_parts = reactants.split('.')
    return reactants_parts[0], reactants_parts[1], reactants_parts[2], product

print(df)

df[['reactant1', 'reactant2', 'reactant3', 'product']] = df['reaction_smiles'].apply(split_reaction_smiles).apply(pd.Series)


# 定义检查SMILES有效性的函数
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

# 检查每一列的SMILES有效性
columns_to_check = ['ligand_smiles', 'reactant1', 'reactant2', 'reactant3', 'product']
invalid_smiles_entries = {}

for column in columns_to_check:
    invalid_smiles_entries[column] = df[~df[column].apply(is_valid_smiles)]

# 输出结果
for column, invalid_entries in invalid_smiles_entries.items():
    print(f"Invalid SMILES in column {column}:")
    if invalid_entries.empty:
        print("None")
    else:
        print(invalid_entries)

