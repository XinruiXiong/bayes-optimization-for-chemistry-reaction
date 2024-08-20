# import numpy as np
# import pandas as pd
# import argparse
# import json
# from skopt.space import Categorical
# from skopt.learning import GaussianProcessRegressor
# from rdkit import Chem
# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
# from rdkit.Chem import rdFingerprintGenerator
# from rdkit.DataStructs import ConvertToNumpyArray
# from sklearn.preprocessing import OneHotEncoder
# import os
# from sklearn.gaussian_process.kernels import Matern

# # 初始化模型和数据
# X_data = []
# y_data = []
# gp_model = None

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
# def encode_categorical_features(temperature, base, solvent, encoders):
#     temperature_encoded = encoders['temperature'].transform(pd.DataFrame([[temperature]], columns=['temperature'])).flatten()
#     base_encoded = encoders['base'].transform(pd.DataFrame([[base]], columns=['base'])).flatten()
#     solvent_encoded = encoders['solvent'].transform(pd.DataFrame([[solvent]], columns=['solvent'])).flatten()
    
#     return np.concatenate([temperature_encoded, base_encoded, solvent_encoded])

# # 初始化并拟合编码器
# def fit_encoders(space_df):
#     encoders = {}
#     encoders['temperature'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['temperature']])
#     encoders['base'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['base']])
#     encoders['solvent'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(space_df[['solvent']])
#     return encoders

# # UCB采集函数
# def ucb(x, gp_model, kappa=100):
#     y_pred, sigma = gp_model.predict([x], return_std=True)
#     return y_pred[0] + kappa * sigma[0]

# # 目标函数
# def objective(params, encoders):
#     global X_data, y_data, gp_model  # 使用全局变量
#     temperature, base, solvent, ligand_smiles = params
    
#     print(f"当前参数组合: temperature={temperature}, base={base}, solvent={solvent}, ligand_smiles={ligand_smiles}")
    
#     ligand_fingerprint = smiles_to_fingerprint(ligand_smiles)
#     categorical_features = encode_categorical_features(temperature, base, solvent, encoders)
#     combined_features = np.concatenate([categorical_features, ligand_fingerprint])
    
#     y_pred = None
#     sigma = None

#     if gp_model is not None and len(X_data) > 1:
#         y_pred, sigma = gp_model.predict([combined_features], return_std=True)
#         print(f"GP 预测值: {y_pred[0]}, 预测不确定性: {sigma[0]}")
#     else:
#         print("GP 模型未初始化或不可用。")
    
#     # 手动输入目标值
#     value = float(input("请输入该参数组合的目标值: "))
    
#     # 保存数据
#     X_data.append(combined_features)
#     y_data.append(-value)

#     # 训练 GP 模型
#     if len(X_data) > 1:
#         gp_model = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=5)
#         gp_model.fit(X_data, y_data)

#     result_entry = {
#         'temperature': temperature,
#         'base': base,
#         'solvent': solvent,
#         'ligand_smiles': ligand_smiles,
#         'value': value,
#         'y_pred': y_pred[0] if y_pred is not None else None,
#         'sigma': sigma[0] if sigma is not None else None
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

# # 从历史数据中初始化模型
# def initialize_model_from_history(history_df, encoders):
#     global X_data, y_data, gp_model

#     for _, row in history_df.iterrows():
#         temperature = row['temperature']
#         base = row['base']
#         solvent = row['solvent']
#         ligand_smiles = row['ligand_smiles']
#         value = row['ee']  

#         ligand_fingerprint = smiles_to_fingerprint(ligand_smiles)
#         categorical_features = encode_categorical_features(temperature, base, solvent, encoders)
#         combined_features = np.concatenate([categorical_features, ligand_fingerprint])

#         X_data.append(combined_features)
#         y_data.append(-value)

#     if len(X_data) > 1:
#         gp_model = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=5)
#         gp_model.fit(X_data, y_data)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Optimize reaction parameters manually")
#     parser.add_argument("--optimize", action='store_true', help="Run Bayesian Optimization")
#     parser.add_argument("--output_json", type=str, default="manual_optimization_results.json", help="Output JSON file for optimization results")
#     args = parser.parse_args()

#     if args.optimize:
#         history_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/new_data.csv')
#         space_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/space.csv')

#         encoders = fit_encoders(space_df)
        
#         # 使用历史数据初始化模型
#         initialize_model_from_history(history_df, encoders)
        
#         unique_temperatures = space_df['temperature'].dropna().astype(float).to_list()
#         unique_bases = space_df['base'].dropna().astype(str).to_list()
#         unique_solvents = space_df['solvent'].dropna().astype(str).to_list()
#         unique_ligand_smiles = space_df['ligand_smiles'].dropna().astype(str).to_list()

#         search_space = [
#             Categorical(unique_temperatures, name='temperature'),
#             Categorical(unique_bases, name='base'),
#             Categorical(unique_solvents, name='solvent'),
#             Categorical(unique_ligand_smiles, name='ligand_smiles')
#         ]
        
#         while True:
#             if gp_model is None or len(X_data) <= 1:
#                 print("初始数据不足，随机选择参数组合进行探索。")
#                 params = [
#                     np.random.choice(unique_temperatures),
#                     np.random.choice(unique_bases),
#                     np.random.choice(unique_solvents),
#                     np.random.choice(unique_ligand_smiles)
#                 ]
#             else:
#                 # 遍历搜索空间中的所有可能点
#                 candidate_points = [
#                     np.concatenate([encode_categorical_features(t, b, s, encoders), smiles_to_fingerprint(l)])
#                     for t in unique_temperatures
#                     for b in unique_bases
#                     for s in unique_solvents
#                     for l in unique_ligand_smiles
#                 ]

#                 # 计算 UCB 值并选择最优点
#                 ucb_values = [ucb(point, gp_model) for point in candidate_points]
#                 best_point = candidate_points[np.argmax(ucb_values)]

#                 # 获取参数组合
#                 temperature, base, solvent, ligand_smiles = [
#                     unique_temperatures[(np.argmax(best_point) // (len(unique_bases) * len(unique_solvents) * len(unique_ligand_smiles))) % len(unique_temperatures)],
#                     unique_bases[(np.argmax(best_point) // (len(unique_solvents) * len(unique_ligand_smiles))) % len(unique_bases)],
#                     unique_solvents[(np.argmax(best_point) // len(unique_ligand_smiles)) % len(unique_solvents)],
#                     unique_ligand_smiles[np.argmax(best_point) % len(unique_ligand_smiles)]
#                 ]

#                 params = [temperature, base, solvent, ligand_smiles]

#             # 调用 objective 函数，并使用返回值更新模型
#             result = objective(params, encoders)

#     else:
#         raise ValueError("Optimization flag (--optimize) must be set for this script.")



import numpy as np
import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from joblib import Parallel, delayed
import os

# 初始化数据
X_data = []
y_data = []
gp_model = None

# SMILES特征预处理
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"SMILES '{smiles}' could not be parsed")
        
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
        fingerprint = generator.GetFingerprint(molecule)
        
        arr = np.zeros((nBits,))
        ConvertToNumpyArray(fingerprint, arr)
        return arr
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return np.zeros(nBits)

# 独热编码函数
def encode_categorical_features(temperature, base, solvent, encoders):
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

# 目标函数：使用手动输入的值进行计算
def objective(params, iteration, encoders):
    temperature, base, solvent, ligand_smiles = params
    
    # 打印当前参数组合并提示输入
    print(f"Iteration {iteration + 1}: 当前参数组合: Temperature={temperature}, Base={base}, Solvent={solvent}, Ligand SMILES={ligand_smiles}", flush=True)
    value = float(input("请输入该参数组合的目标值: "))

    # 返回负值用于最小化
    return -value, value

# 初始化GP模型
def initialize_gp_model_from_history(history_df, encoders):
    global X_data, y_data, gp_model

    # 清空之前的数据
    X_data = []
    y_data = []

    for _, row in history_df.iterrows():
        # 从历史数据中获取目标值
        y_pred = row['ee']  # 假设历史数据中有一个列名为 'ee' 的列表示目标值

        # 对输入数据进行编码
        ligand_fingerprint = smiles_to_fingerprint(row['ligand_smiles'])
        categorical_features = encode_categorical_features(row['temperature'], row['base'], row['solvent'], encoders)
        combined_features = np.concatenate([categorical_features, ligand_fingerprint])

        # 添加到训练数据集中
        X_data.append(combined_features)
        y_data.append(y_pred)

    # 如果有足够的数据点，拟合高斯过程模型
    if len(X_data) > 1:
        kernel = Matern(nu=2.5)
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp_model.fit(X_data, y_data)

# UCB采集函数
def ucb(x, gp_model, kappa=1000):
    y_pred, sigma = gp_model.predict([x], return_std=True)
    return y_pred[0] + kappa * sigma[0]

# 将结果写入 JSON 文件
def write_results_to_json(output_file, params, result):
    data = {
        'temperature': params[0],
        'base': params[1],
        'solvent': params[2],
        'ligand_smiles': params[3],
        'result': result
    }
    
    # 如果文件不存在，创建新文件
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([data], f, indent=4)
    else:
        # 如果文件存在，追加内容
        with open(output_file, 'r+') as f:
            file_data = json.load(f)
            file_data.append(data)
            f.seek(0)
            json.dump(file_data, f, indent=4)

if __name__ == "__main__":
    history_df = pd.read_csv('new_data.csv')
    space_df = pd.read_csv('space.csv')

    encoders = fit_encoders(space_df)
    
    # 使用历史数据初始化高斯过程模型
    initialize_gp_model_from_history(history_df, encoders)

    unique_temperatures = space_df['temperature'].dropna().astype(float).to_list()
    unique_bases = space_df['base'].dropna().astype(str).to_list()
    unique_solvents = space_df['solvent'].dropna().astype(str).to_list()
    unique_ligand_smiles = space_df['ligand_smiles'].dropna().astype(str).to_list()

    # 构建所有可能的参数组合
    params_combinations = [
        (t, b, s, l) for t in unique_temperatures
                     for b in unique_bases
                     for s in unique_solvents
                     for l in unique_ligand_smiles
    ]

    # 生成候选点
    candidate_points = [
        np.concatenate([encode_categorical_features(t, b, s, encoders), smiles_to_fingerprint(l)])
        for (t, b, s, l) in params_combinations
    ]

    for iteration in range(10):
        if gp_model is None or len(X_data) <= 1:
            print("初始数据不足，随机选择参数组合进行探索。")
            params = [
                np.random.choice(unique_temperatures),
                np.random.choice(unique_bases),
                np.random.choice(unique_solvents),
                np.random.choice(unique_ligand_smiles)
            ]
        else:
            # 并行计算 UCB 值
            ucb_values = Parallel(n_jobs=-1)(delayed(ucb)(point, gp_model) for point in candidate_points)

            # 找到最佳点
            best_index = np.argmax(ucb_values)
            best_params = params_combinations[best_index]  # 对应的最佳参数组合

            params = list(best_params)

        # 运行目标函数并记录结果
        result, actual_value = objective(params, iteration, encoders)
        print(f"Iteration {iteration + 1}/10: Params: {params}, Result: {actual_value}")

        # 将新数据点添加到训练集中
        new_fingerprint = smiles_to_fingerprint(params[3])  # ligand_smiles
        new_categorical_features = encode_categorical_features(params[0], params[1], params[2], encoders)
        new_combined_features = np.concatenate([new_categorical_features, new_fingerprint])
        
        X_data.append(new_combined_features)
        y_data.append(result)

        # 更新高斯过程模型
        if len(X_data) > 1:
            kernel = Matern(nu=2.5)
            gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
            gp_model.fit(X_data, y_data)

        # 将结果写入 JSON 文件
        write_results_to_json("manual_optimization_results.json", params, actual_value)
