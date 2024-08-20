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

# 定义SMILES转换函数
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"SMILES '{smiles}' could not be parsed")
        fingerprint = GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
        arr = np.zeros((nBits,))
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return arr
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return np.zeros(nBits)

# 过滤无效的SMILES
def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

df = df[df['ligand_smiles'].apply(is_valid_smiles)]

# 特征和目标变量
X = df[['temperature', 'ligand', 'base', 'solvent', 'ligand_smiles']]
y_yield = df['yield']
y_ee = df['ee']

# 数值特征预处理
numerical_features = ['temperature']
numerical_transformer = StandardScaler()

# 文本特征预处理
categorical_features = ['ligand', 'base', 'solvent']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# SMILES特征预处理
def smiles_transformer(smiles_series):
    return np.array([smiles_to_fingerprint(smiles) for smiles in smiles_series])

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('smi', FunctionTransformer(smiles_transformer), 'ligand_smiles')
    ])

# 随机森林回归器的参数网格
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__bootstrap': [True, False]
}

# 构建包含预处理和随机森林回归器的管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 使用网格搜索进行超参数调整
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='r2')

# 分割数据集
X_train, X_test, y_yield_train, y_yield_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
_, _, y_ee_train, y_ee_test = train_test_split(X, y_ee, test_size=0.2, random_state=42)

# 训练模型并进行超参数调整
grid_search.fit(X_train, y_yield_train)

# 输出最佳参数
print("Best parameters for yield model:", grid_search.best_params_)

# 预测
y_yield_pred = grid_search.predict(X_test)

# 评估模型
mse_yield = mean_squared_error(y_yield_test, y_yield_pred)
r2_yield = r2_score(y_yield_test, y_yield_pred)

print(f'Yield - Mean Squared Error: {mse_yield}')
print(f'Yield - R^2 Score: {r2_yield}')

# 对ee模型进行相同的步骤
grid_search.fit(X_train, y_ee_train)
print("Best parameters for ee model:", grid_search.best_params_)
y_ee_pred = grid_search.predict(X_test)
mse_ee = mean_squared_error(y_ee_test, y_ee_pred)
r2_ee = r2_score(y_ee_test, y_ee_pred)

print(f'EE - Mean Squared Error: {mse_ee}')
print(f'EE - R^2 Score: {r2_ee}')
