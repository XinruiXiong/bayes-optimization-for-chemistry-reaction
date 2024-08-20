# last edited: 08/20/2024


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib  # 用于保存模型
from joblib import parallel_backend

# 读取数据集
file_path = 'new_data.csv'
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
X = df[['temperature', 'base', 'solvent', 'ligand_smiles']]
y_yield = df['yield']
y_ee = df['ee']

# 数值特征预处理
numerical_features = ['temperature']
numerical_transformer = StandardScaler()

# 文本特征预处理
categorical_features = ['base', 'solvent']
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

# 随机森林回归器的参数空间
param_space = {
    'regressor__n_estimators': Integer(100, 300),
    'regressor__max_depth': Integer(10, 30),
    'regressor__min_samples_split': Integer(2, 10),
    'regressor__min_samples_leaf': Integer(1, 4),
    'regressor__bootstrap': Categorical([True, False])
}

# 构建包含预处理和随机森林回归器的管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 使用贝叶斯优化进行超参数调整，启用多核处理
kf = KFold(n_splits=5, shuffle=True, random_state=42)
bayes_search = BayesSearchCV(model, param_space, n_iter=32, cv=kf, scoring='r2', random_state=42, n_jobs=-1)

# 使用交叉验证训练和评估 Yield 模型
with parallel_backend('threading'):
    bayes_search.fit(X, y_yield)

# 保存最佳的 Yield 模型
best_yield_model = bayes_search.best_estimator_
joblib.dump(best_yield_model, 'yield_pipeline_model.pkl')

# 输出最佳参数和评估结果
print("Best parameters for Yield model:", bayes_search.best_params_)
y_yield_pred = bayes_search.predict(X)
mse_yield = mean_squared_error(y_yield, y_yield_pred)
r2_yield = r2_score(y_yield, y_yield_pred)
print(f'Yield - Mean Squared Error: {mse_yield}')
print(f'Yield - R^2 Score: {r2_yield}')

# 使用交叉验证训练和评估 EE 模型
with parallel_backend('threading'):
    bayes_search.fit(X, y_ee)

# 保存最佳的 EE 模型
best_ee_model = bayes_search.best_estimator_
joblib.dump(best_ee_model, 'ee_pipeline_model.pkl')

# 输出最佳参数和评估结果
print("Best parameters for EE model:", bayes_search.best_params_)
y_ee_pred = bayes_search.predict(X)
mse_ee = mean_squared_error(y_ee, y_ee_pred)
r2_ee = r2_score(y_ee, y_ee_pred)
print(f'EE - Mean Squared Error: {mse_ee}')
print(f'EE - R^2 Score: {r2_ee}')
