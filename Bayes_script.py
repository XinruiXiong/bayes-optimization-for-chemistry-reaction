import numpy as np
import pandas as pd
import argparse
import json
from skopt.space import Categorical
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from joblib import Parallel, delayed
import os
import joblib

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

# SMILES特征预处理
def smiles_transformer(smiles_series):
    return np.array([smiles_to_fingerprint(smiles) for smiles in smiles_series])

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

# 自定义的 predict 函数，用于封装 rf_model 的预测逻辑
def predict(rf_model, temperature, base, solvent, ligand_smiles, encoders):
    # 构建输入数据
    infer_data = pd.DataFrame({
        'temperature': [temperature],
        'base': [base],
        'solvent': [solvent],
        'ligand_smiles': [ligand_smiles]
    })
    
    # 使用模型（Pipeline）进行预测
    prediction = rf_model.predict(infer_data)[0]
    return prediction

# 目标函数：使用自定义的 predict 函数进行预测
def objective(params, rf_model, encoders):
    temperature, base, solvent, ligand_smiles = params
    
    # 使用自定义的 predict 函数进行预测
    y_pred = predict(rf_model, temperature, base, solvent, ligand_smiles, encoders)
    return -y_pred

# 从历史数据中初始化模型
def initialize_gp_model_from_history(history_df, rf_model, encoders):
    global X_data, y_data, gp_model

    for _, row in history_df.iterrows():
        # 使用自定义的 predict 函数进行预测
        y_pred = predict(rf_model, row['temperature'], row['base'], row['solvent'], row['ligand_smiles'], encoders)

        # 对输入数据进行编码
        ligand_fingerprint = smiles_to_fingerprint(row['ligand_smiles'])
        categorical_features = encode_categorical_features(row['temperature'], row['base'], row['solvent'], encoders)
        combined_features = np.concatenate([categorical_features, ligand_fingerprint])

        X_data.append(combined_features)
        y_data.append(-y_pred)

    if len(X_data) > 1:
        kernel = Matern(nu=2.5)
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp_model.fit(X_data, y_data)

# 加载随机森林模型
def load_rf_model(model_type):
    model_path = f'reaction_rf/{model_type}_pipeline_model.pkl'
    try:
        loaded_object = joblib.load(model_path)
        print("Random Forest model loaded successfully.")
        
        # 如果加载的是字典，提取模型
        if isinstance(loaded_object, dict):
            rf_model = loaded_object.get('model', None)
            if rf_model is None:
                raise ValueError("The loaded dictionary does not contain a model under the key 'model'.")
        else:
            rf_model = loaded_object
        
        return rf_model
    except FileNotFoundError:
        print("Random Forest model file not found.")
        raise

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
    parser = argparse.ArgumentParser(description="Optimize reaction parameters manually or predict using Random Forest")
    parser.add_argument("--optimize", action='store_true', help="Run Bayesian Optimization")
    parser.add_argument("--output_json", type=str, default="manual_optimization_results.json", help="Output JSON file for optimization results")
    parser.add_argument("--model_type", type=str, required=True, choices=['ee', 'yield'], help="Model type to use for prediction")
    parser.add_argument("--temperature", type=float, help="Temperature for prediction")
    parser.add_argument("--base", type=str, help="Base for prediction")
    parser.add_argument("--solvent", type=str, help="Solvent for prediction")
    parser.add_argument("--ligand_smiles", type=str, help="Ligand SMILES for prediction")
    args = parser.parse_args()

    # 加载随机森林模型（只加载一次）
    rf_model = load_rf_model(args.model_type)

    if args.optimize:
        history_df = pd.read_csv('new_data.csv')
        space_df = pd.read_csv('space.csv')

        encoders = fit_encoders(space_df)
        
        # 使用历史数据初始化高斯过程模型
        initialize_gp_model_from_history(history_df, rf_model, encoders)

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
            result = objective(params, rf_model, encoders)
            print(f"Iteration {iteration + 1}/10: Params: {params}, Result: {result}")

            # 将新数据点添加到训练集中
            new_fingerprint = smiles_to_fingerprint(params[3])  # ligand_smiles
            new_categorical_features = encode_categorical_features(params[0], params[1], params[2], encoders)
            new_combined_features = np.concatenate([new_categorical_features, new_fingerprint])
            
            X_data.append(new_combined_features)
            y_data.append(result)

            # 更新高斯过程模型
            kernel = Matern(nu=2.5)
            gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
            gp_model.fit(X_data, y_data)

            # 将结果写入 JSON 文件
            write_results_to_json(args.output_json, params, result)
    
    elif args.temperature is not None and args.base is not None and args.solvent is not None and args.ligand_smiles is not None:
        # 如果提供了具体的参数，使用 RandomForest 进行一次预测
        space_df = pd.read_csv('space.csv')
        encoders = fit_encoders(space_df)
        
        # 预测结果
        result = predict(rf_model, args.temperature, args.base, args.solvent, args.ligand_smiles, encoders)
        print(f"Prediction for parameters: Temperature={args.temperature}, Base={args.base}, Solvent={args.solvent}, Ligand SMILES={args.ligand_smiles} -> Result: {result}")
    
    else:
        raise ValueError("Either provide parameters for prediction or use --optimize flag for Bayesian optimization.")
