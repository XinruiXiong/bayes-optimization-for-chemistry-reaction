import numpy as np
import pandas as pd
import argparse
import json
from skopt import gp_minimize
from skopt.space import Categorical
import os
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# SMILES特征预处理
def smiles_transformer(smiles_series):
    return np.array([smiles_to_fingerprint(smiles) for smiles in smiles_series])

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

# # 修改的 objective 函数，手动输入目标值并即时更新 JSON 文件
# def objective(params):
#     reaction, temperature, base, solvent, ligand_smiles = params
#     print(f"当前参数组合: reaction={reaction}, temperature={temperature}, base={base}, solvent={solvent}, ligand_smiles={ligand_smiles}")
#     value = float(input("请输入该参数组合的目标值: "))
    
#     # 每次输入目标值后即时记录到 JSON 文件
#     result_entry = {
#         'reaction': reaction,
#         'temperature': temperature,
#         'base': base,
#         'solvent': solvent,
#         'ligand_smiles': ligand_smiles,
#         'value': value
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
    
#     return -value  # 目标值传入为负，因为贝叶斯优化最小化目标函数

# 修改的 objective 函数，手动输入目标值并即时更新 JSON 文件
def objective(params):
    reaction, temperature, base, solvent, ligand_smiles = params
    
    # 打印原始的 ligand_smiles
    print(f"当前参数组合: reaction={reaction}, temperature={temperature}, base={base}, solvent={solvent}, ligand_smiles={ligand_smiles}")
    
    # 将 ligand_smiles 转换为分子指纹，用于优化
    ligand_fingerprint = smiles_to_fingerprint(ligand_smiles)
    
    value = float(input("请输入该参数组合的目标值: "))
    
    # 每次输入目标值后即时记录到 JSON 文件
    result_entry = {
        'reaction': reaction,
        'temperature': temperature,
        'base': base,
        'solvent': solvent,
        'ligand_smiles': ligand_smiles,  # 存储原始的 SMILES
        'ligand_fingerprint': ligand_fingerprint.tolist(),  # 也存储转换后的分子指纹
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
    
    return -value  # 使用分子指纹的优化目标值传入为负




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize reaction parameters manually")
    parser.add_argument("--optimize", action='store_true', help="Run Bayesian Optimization")
    parser.add_argument("--output_json", type=str, default="manual_optimization_results.json", help="Output JSON file for optimization results")
    args = parser.parse_args()

    if args.optimize:
        history_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/reaction_rf/data.csv')

        history_df['base'] = history_df['base'].astype(str)
        history_df['solvent'] = history_df['solvent'].astype(str)
        history_df['ligand_smiles'] = history_df['ligand_smiles'].astype(str)
        history_df['temperature'] = history_df['temperature'].astype(float)
        history_df['reaction'] = history_df['reaction'].astype(float).astype(str)  # 将 reaction 列转换为字符串类型

        initial_x = history_df[['reaction', 'temperature', 'base', 'solvent', 'ligand_smiles']].values.tolist()
        initial_y = [-float(y) for y in history_df['yield']]  # 假设使用 'yield' 作为目标列
        
        # 读取参数空间CSV文件
        space_df = pd.read_csv('/mnt/petrelfs/xiongxinrui/ParaOptim/space.csv')  # 使用最新上传的文件路径

        unique_reactions = space_df['reaction'].dropna().astype(str).to_list()
        unique_temperatures = space_df['temperature'].dropna().astype(float).to_list()
        unique_bases = space_df['base'].dropna().astype(str).to_list()
        unique_solvents = space_df['solvent'].dropna().astype(str).to_list()
        unique_ligand_smiles = space_df['ligand_smiles'].dropna().astype(str).to_list()

        # 定义搜索空间，使用去除 NaN 后的唯一值
        search_space = [
            Categorical(unique_reactions, name='reaction'),
            Categorical(unique_temperatures, name='temperature'),
            Categorical(unique_bases, name='base'),
            Categorical(unique_solvents, name='solvent'),
            Categorical(unique_ligand_smiles, name='ligand_smiles')
        ]

        res = gp_minimize(objective, search_space, x0=initial_x, y0=initial_y, n_calls=30, verbose=True)
        print(f"最佳参数组合: {res.x}")
        print(f"最佳目标值: {-res.fun}")

    else:
        raise ValueError("Optimization flag (--optimize) must be set for this script.")
