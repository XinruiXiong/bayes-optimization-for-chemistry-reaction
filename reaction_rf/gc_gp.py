import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
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

# 检查初始数据形状
print("Initial data shape:", df.shape)

# 拆分reaction_smiles列
def split_reaction_smiles(smiles):
    try:
        reactants, product = smiles.split('>>')
        reactants_parts = reactants.split('.')
        return reactants_parts[0], reactants_parts[1], reactants_parts[2], product
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None, None, None, None

df[['reactant1', 'reactant2', 'reactant3', 'product']] = df['reaction_smiles'].apply(split_reaction_smiles).apply(pd.Series)

# 检查拆分后的数据形状
print("After splitting reaction_smiles:", df.shape)

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


# 特征和目标变量
X = df[['temperature', 'base', 'solvent', 'ligand_smiles', 'reactant1', 'reactant2', 'reactant3', 'product']]
y = df['yield']
print("Features and target variable:", X.shape, y.shape)

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
        ('smi_ligand', FunctionTransformer(smiles_transformer), 'ligand_smiles'),
        ('smi_reactant1', FunctionTransformer(smiles_transformer), 'reactant1'),
        ('smi_reactant2', FunctionTransformer(smiles_transformer), 'reactant2'),
        ('smi_reactant3', FunctionTransformer(smiles_transformer), 'reactant3'),
        ('smi_product', FunctionTransformer(smiles_transformer), 'product')
    ])

# 高斯过程回归器的参数网格
param_grid = {
    'regressor__alpha': [1e-2, 1e-3],
    'regressor__kernel': [C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))]
}

# 构建包含预处理和高斯过程回归器的管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GaussianProcessRegressor(random_state=42))
])

# 使用10折交叉验证进行超参数调整
kf = KFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='r2')

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training and testing split:", X_train.shape, X_test.shape)

# 训练模型并进行超参数调整
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters for yield model:", grid_search.best_params_)

# 预测
y_pred = grid_search.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
