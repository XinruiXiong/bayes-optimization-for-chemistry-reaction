import numpy as np
import pandas as pd
import argparse
import json
from skopt import gp_minimize
from skopt.space import Categorical
import os
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.preprocessing import OneHotEncoder
from rdkit.Chem import Draw

# SMILES特征预处理
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"SMILES '{smiles}' could not be parsed")
        fingerprint = GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
        arr = np.zeros((nBits,))
        ConvertToNumpyArray(fingerprint, arr)
        return arr
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return np.zeros(nBits)

# 独热编码函数
def encode_categorical_features(temperature, base, solvent, encoders):
    # 使用 DataFrame 包装数据并指定列名以避免警告
    temperature_encoded = encoders['temperature'].transform(pd.DataFrame([[temperature]], columns=['temperature'])).flatten()
    base_encoded = encoders['base'].transform(pd.DataFrame([[base]], columns=['base'])).flatten()
    solvent_encoded = encoders['solvent'].transform(pd.DataFrame([[solvent]], columns=['solvent'])).flatten()
    
    return np.concatenate([temperature_encoded, base_encoded, solvent_encoded])

# 初始化并拟合编码器
def fit_encoders(space_df):
    encoders = {}
    encoders['temperature'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['temperature']])
    encoders['base'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['base']])
    encoders['solvent'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['solvent']])
    return encoders


# 修改后的 objective 函数，保存 SMILES 对应的图像文件
def objective(params, encoders):
    temperature, base, solvent, ligand_smiles = params
    
    # 打印当前参数组合
    print(f"当前参数组合: temperature={temperature}, base={base}, solvent={solvent}, ligand_smiles={ligand_smiles}")
    
    # 将 ligand_smiles 转换为分子指纹
    ligand_fingerprint = smiles_to_fingerprint(ligand_smiles)
    
    # 对 categorical 特征进行编码
    categorical_features = encode_categorical_features(temperature, base, solvent, encoders)
    
    # 将所有特征组合起来
    combined_features = np.concatenate([categorical_features, ligand_fingerprint])
    
    # 生成并保存分子图像
    try:
        molecule = Chem.MolFromSmiles(ligand_smiles)
        if molecule:
            img_filename = f"ligand_{ligand_smiles}.png".replace("/", "_")
            img_path = os.path.join("images", img_filename)
            if not os.path.exists("images"):
                os.makedirs("images")
            Draw.MolToFile(molecule, img_path)
        else:
            img_path = None
            print(f"Error: Cannot generate image for SMILES '{ligand_smiles}'")
    except Exception as e:
        img_path = None
        print(f"Error processing SMILES image '{ligand_smiles}': {e}")
    
    value = float(input("请输入该参数组合的目标值: "))
    
    # 每次输入目标值后即时记录到 JSON 文件
    result_entry = {
        'temperature': temperature,
        'base': base,
        'solvent': solvent,
        'ligand_smiles': ligand_smiles,  # 存储原始的 SMILES
        'ligand_image': img_path,  # 存储图像路径
        'value': value
    }
    
    if os.path.exists(args.output_json):
        with open(args.output_json, 'r+') as f:
            data = json.load(f)
            data['results'].append(result_entry)
            data['function_values'].append(-value)
            f.seek(0)
            json.dump(data, f, indent=4)
    else:
        with open(args.output_json, 'w') as f:
            json.dump({'results': [result_entry], 'function_values': [-value]}, f, indent=4)
    
    return -value  # 返回负的目标值用于最小化


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize reaction parameters manually")
    parser.add_argument("--optimize", action='store_true', help="Run Bayesian Optimization")
    parser.add_argument("--output_json", type=str, default="manual_optimization_results.json", help="Output JSON file for optimization results")
    args = parser.parse_args()

    if args.optimize:
        # 读取历史数据和参数空间定义的CSV文件
        history_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/new_data.csv')
        space_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/space.csv')  # 使用最新上传的文件路径

        # 初始化并拟合编码器
        encoders = fit_encoders(space_df)
        
        history_df['temperature'] = history_df['temperature'].astype(float).astype(str)
        
        # 准备初始点和目标值，注意这里使用原始的分类值
        initial_x = history_df[['temperature', 'base', 'solvent', 'ligand_smiles']].values.tolist()
        initial_y = [-float(y) for y in history_df['ee']]  # 假设使用 'ee' 作为目标列

        # 定义搜索空间，使用去除 NaN 后的唯一值
        unique_temperatures = space_df['temperature'].dropna().astype(str).to_list()
        unique_bases = space_df['base'].dropna().astype(str).to_list()
        unique_solvents = space_df['solvent'].dropna().astype(str).to_list()
        unique_ligand_smiles = space_df['ligand_smiles'].dropna().astype(str).to_list()

        search_space = [
            Categorical(unique_temperatures, name='temperature'),
            Categorical(unique_bases, name='base'),
            Categorical(unique_solvents, name='solvent'),
            Categorical(unique_ligand_smiles, name='ligand_smiles')
        ]

        # 执行贝叶斯优化
        res = gp_minimize(lambda params: objective(params, encoders), 
                          search_space, 
                          x0=initial_x, 
                          y0=initial_y, 
                          n_calls=30, 
                          verbose=True)
        print(f"最佳参数组合: {res.x}")
        print(f"最佳目标值: {-res.fun}")

    else:
        raise ValueError("Optimization flag (--optimize) must be set for this script.")


# import numpy as np
# import pandas as pd
# import argparse
# import json
# from skopt import gp_minimize
# from skopt.space import Categorical, Space
# from skopt.utils import cook_estimator
# from rdkit import Chem
# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
# from rdkit.DataStructs import ConvertToNumpyArray
# from sklearn.preprocessing import OneHotEncoder
# import os

# # SMILES特征预处理
# def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
#     try:
#         molecule = Chem.MolFromSmiles(smiles)
#         if molecule is None:
#             raise ValueError(f"SMILES '{smiles}' could not be parsed")
#         fingerprint = GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
#         arr = np.zeros((nBits,))
#         ConvertToNumpyArray(fingerprint, arr)
#         return arr
#     except Exception as e:
#         print(f"Error processing SMILES '{smiles}': {e}")
#         return np.zeros(nBits)

# # 独热编码函数
# def encode_categorical_features(reaction, temperature, base, solvent, encoders):
#     reaction_encoded = encoders['reaction'].transform(pd.DataFrame([[reaction]], columns=['reaction'])).flatten()
#     temperature_encoded = encoders['temperature'].transform(pd.DataFrame([[temperature]], columns=['temperature'])).flatten()
#     base_encoded = encoders['base'].transform(pd.DataFrame([[base]], columns=['base'])).flatten()
#     solvent_encoded = encoders['solvent'].transform(pd.DataFrame([[solvent]], columns=['solvent'])).flatten()
#     return np.concatenate([reaction_encoded, temperature_encoded, base_encoded, solvent_encoded])

# # 初始化并拟合编码器
# def fit_encoders(space_df):
#     encoders = {}
#     encoders['reaction'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['reaction']])
#     encoders['temperature'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['temperature']])
#     encoders['base'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['base']])
#     encoders['solvent'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['solvent']])
#     return encoders

# # 修改的 objective 函数，手动输入目标值并即时更新 JSON 文件
# def objective(params, encoders, gp_model, history_x):
#     reaction, temperature, base, solvent, ligand_smiles = params
    
#     print(f"当前参数组合: reaction={reaction}, temperature={temperature}, base={base}, solvent={solvent}, ligand_smiles={ligand_smiles}")
    
#     ligand_fingerprint = smiles_to_fingerprint(ligand_smiles)
#     categorical_features = encode_categorical_features(reaction, temperature, base, solvent, encoders)
#     combined_features = np.concatenate([categorical_features, ligand_fingerprint])
    
#     value = float(input("请输入该参数组合的目标值: "))
    
#     # 使用 GP 模型预测目标函数值
#     gp_prediction, gp_std = gp_model.predict([combined_features], return_std=True)
#     print(f"GP 模型预测的目标函数值: {gp_prediction[0]}, 标准差: {gp_std[0]}")
    
#     # 每次输入目标值后即时记录到 JSON 文件
#     result_entry = {
#         'reaction': reaction,
#         'temperature': temperature,
#         'base': base,
#         'solvent': solvent,
#         'ligand_smiles': ligand_smiles,
#         'input_value': value,
#         'gp_estimated_value': gp_prediction[0]
#     }
    
#     if os.path.exists(args.output_json):
#         with open(args.output_json, 'r+') as f:
#             data = json.load(f)
#             data['results'].append(result_entry)
#             data['function_values'].append(-value)
#             f.seek(0)
#             json.dump(data, f, indent=4)
#     else:
#         with open(args.output_json, 'w') as f:
#             json.dump({'results': [result_entry], 'function_values': [-value]}, f, indent=4)
    
#     return -value

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Optimize reaction parameters manually")
#     parser.add_argument("--optimize", action='store_true', help="Run Bayesian Optimization")
#     parser.add_argument("--output_json", type=str, default="manual_optimization_results.json", help="Output JSON file for optimization results")
#     args = parser.parse_args()

#     if args.optimize:
#         history_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/reaction_rf/data.csv')
#         space_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/space.csv')

#         encoders = fit_encoders(space_df)
        
#         history_df['reaction'] = history_df['reaction'].astype(float).astype(str)
        
#         initial_x = history_df[['reaction', 'temperature', 'base', 'solvent', 'ligand_smiles']].values.tolist()
#         initial_y = [-float(y) for y in history_df['yield']]

#         unique_reactions = space_df['reaction'].dropna().astype(str).to_list()
#         unique_temperatures = space_df['temperature'].dropna().astype(float).to_list()
#         unique_bases = space_df['base'].dropna().astype(str).to_list()
#         unique_solvents = space_df['solvent'].dropna().astype(str).to_list()
#         unique_ligand_smiles = space_df['ligand_smiles'].dropna().astype(str).to_list()

#         search_space = Space([
#             Categorical(unique_reactions, name='reaction'),
#             Categorical(unique_temperatures, name='temperature'),
#             Categorical(unique_bases, name='base'),
#             Categorical(unique_solvents, name='solvent'),
#             Categorical(unique_ligand_smiles, name='ligand_smiles')
#         ])

#         gp_model = cook_estimator("GP", space=search_space, random_state=42)
        
#         res = gp_minimize(lambda params: objective(params, encoders, gp_model, initial_x), 
#                           search_space, 
#                           x0=initial_x, 
#                           y0=initial_y, 
#                           n_calls=30, 
#                           verbose=True)
#         print(f"最佳参数组合: {res.x}")
#         print(f"最佳目标值: {-res.fun}")

#     else:
#         raise ValueError("Optimization flag (--optimize) must be set for this script.")

# import numpy as np
# import pandas as pd
# import argparse
# import json
# from skopt import gp_minimize
# from skopt.space import Categorical
# from skopt.learning import GaussianProcessRegressor
# from rdkit import Chem
# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
# from rdkit.DataStructs import ConvertToNumpyArray
# from sklearn.preprocessing import OneHotEncoder
# import os

# # SMILES特征预处理
# def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
#     try:
#         molecule = Chem.MolFromSmiles(smiles)
#         if molecule is None:
#             raise ValueError(f"SMILES '{smiles}' could not be parsed")
#         fingerprint = GetMorganFingerprintAsBitVect(molecule, radius, nBits=nBits)
#         arr = np.zeros((nBits,))
#         ConvertToNumpyArray(fingerprint, arr)
#         return arr
#     except Exception as e:
#         print(f"Error processing SMILES '{smiles}': {e}")
#         return np.zeros(nBits)

# # 独热编码函数
# def encode_categorical_features(reaction, temperature, base, solvent, encoders):
#     reaction_encoded = encoders['reaction'].transform(pd.DataFrame([[reaction]], columns=['reaction'])).flatten()
#     temperature_encoded = encoders['temperature'].transform(pd.DataFrame([[temperature]], columns=['temperature'])).flatten()
#     base_encoded = encoders['base'].transform(pd.DataFrame([[base]], columns=['base'])).flatten()
#     solvent_encoded = encoders['solvent'].transform(pd.DataFrame([[solvent]], columns=['solvent'])).flatten()
    
#     return np.concatenate([reaction_encoded, temperature_encoded, base_encoded, solvent_encoded])

# # 初始化并拟合编码器
# def fit_encoders(space_df):
#     encoders = {}
#     encoders['reaction'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['reaction']])
#     encoders['temperature'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['temperature']])
#     encoders['base'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['base']])
#     encoders['solvent'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['solvent']])
#     return encoders

# # 修改的 objective 函数
# def objective(params, encoders, gp_model=None):
#     reaction, temperature, base, solvent, ligand_smiles = params
    
#     print(f"当前参数组合: reaction={reaction}, temperature={temperature}, base={base}, solvent={solvent}, ligand_smiles={ligand_smiles}")
    
#     ligand_fingerprint = smiles_to_fingerprint(ligand_smiles)
#     categorical_features = encode_categorical_features(reaction, temperature, base, solvent, encoders)
#     combined_features = np.concatenate([categorical_features, ligand_fingerprint])
    
#     # 在目标函数中直接获取GP的预测值
#     if gp_model:
#         y_pred, sigma = gp_model.predict([combined_features], return_std=True)
#         print(f"GP预测值: {y_pred[0]}, 预测不确定性: {sigma[0]}")
    
#     # 手动输入目标值
#     value = float(input("请输入该参数组合的目标值: "))
    
#     result_entry = {
#         'reaction': reaction,
#         'temperature': temperature,
#         'base': base,
#         'solvent': solvent,
#         'ligand_smiles': ligand_smiles,
#         'value': value
#     }
    
#     # 更新 JSON 文件
#     if os.path.exists(args.output_json):
#         with open(args.output_json, 'r+') as f:
#             data = json.load(f)
#             data['results'].append(result_entry)
#             data['function_values'].append(-value)
#             f.seek(0)
#             json.dump(data, f, indent=4)
#     else:
#         with open(args.output_json, 'w') as f:
#             json.dump({'results': [result_entry], 'function_values': [-value]}, f, indent=4)
    
#     return -value

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Optimize reaction parameters manually")
#     parser.add_argument("--optimize", action='store_true', help="Run Bayesian Optimization")
#     parser.add_argument("--output_json", type=str, default="manual_optimization_results.json", help="Output JSON file for optimization results")
#     args = parser.parse_args()

#     if args.optimize:
#         history_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/reaction_rf/data.csv')
#         space_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/space.csv')

#         encoders = fit_encoders(space_df)
        
#         history_df['reaction'] = history_df['reaction'].astype(float).astype(str)
        
#         initial_x = history_df[['reaction', 'temperature', 'base', 'solvent', 'ligand_smiles']].values.tolist()
#         initial_y = [-float(y) for y in history_df['yield']]  # 假设使用 'yield' 作为目标列

#         unique_reactions = space_df['reaction'].dropna().astype(str).to_list()
#         unique_temperatures = space_df['temperature'].dropna().astype(float).to_list()
#         unique_bases = space_df['base'].dropna().astype(str).to_list()
#         unique_solvents = space_df['solvent'].dropna().astype(str).to_list()
#         unique_ligand_smiles = space_df['ligand_smiles'].dropna().astype(str).to_list()

#         search_space = [
#             Categorical(unique_reactions, name='reaction'),
#             Categorical(unique_temperatures, name='temperature'),
#             Categorical(unique_bases, name='base'),
#             Categorical(unique_solvents, name='solvent'),
#             Categorical(unique_ligand_smiles, name='ligand_smiles')
#         ]
        
#         # 自定义目标函数用于捕获GP模型
#         def custom_objective(params):
#             return objective(params, encoders, gp_model=res.models[-1] if res.models else None)
        
#         res = gp_minimize(custom_objective, 
#                           search_space, 
#                           x0=initial_x, 
#                           y0=initial_y, 
#                           n_calls=30, 
#                           verbose=True)

#         print(f"最佳参数组合: {res.x}")
#         print(f"最佳目标值: {-res.fun}")

#     else:
#         raise ValueError("Optimization flag (--optimize) must be set for this script.")
