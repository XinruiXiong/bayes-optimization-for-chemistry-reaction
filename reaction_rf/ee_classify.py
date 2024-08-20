import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
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

df[['reactant1', 'reactant2', 'reactant3', 'product']] = df['reaction_smiles'].apply(split_reaction_smiles).apply(pd.Series)

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


# 修改yield列为二分类标签
df['ee'] = df['ee'].apply(lambda x: 0 if x == 0 else 1)

df.to_csv('/mnt/petrelfs/xiongxinrui/reaction_rf/temp.csv')

# 特征和目标变量
X = df[['temperature', 'base', 'solvent', 'ligand_smiles', 'reactant1', 'reactant2', 'reactant3', 'product']]
y = df['ee']

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

# 随机森林分类器的参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# 构建包含预处理和随机森林分类器的管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 使用3折交叉验证进行超参数调整
kf = KFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy')

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型并进行超参数调整
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters for yield model:", grid_search.best_params_)

# 预测
y_pred = grid_search.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report_str)
