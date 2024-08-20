# python predict.py --model_type ee --temperature 90 --ligand D-phenylalanine --base K2CO3 --solvent t-BuOH --ligand_smiles "N[C@H](CC1=CC=CC=C1)C(O)=O"

# python predict.py --model_type yield --temperature 90 --ligand D-phenylalanine --base K2CO3 --solvent t-BuOH --ligand_smiles "N[C@H](CC1=CC=CC=C1)C(O)=O"

import numpy as np
import pandas as pd
import joblib
import argparse
from sklearn.preprocessing import FunctionTransformer

# 确保在加载模型之前定义 smiles_transformer
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

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

def smiles_transformer(smiles_series):
    return np.array([smiles_to_fingerprint(smiles) for smiles in smiles_series])

# 加载模型
def load_model(model_type):
    if model_type == 'ee':
        return joblib.load('ee_pipeline_model.pkl')
    elif model_type == 'yield':
        return joblib.load('yield_pipeline_model.pkl')
    else:
        raise ValueError("Invalid model type. Choose 'ee' or 'yield'.")

# 进行预测
def predict(model_type, temperature, ligand, base, solvent, ligand_smiles):
    model = load_model(model_type)
    infer_data = pd.DataFrame({
        'temperature': [temperature],
        'ligand': [ligand],
        'base': [base],
        'solvent': [solvent],
        'ligand_smiles': [ligand_smiles]
    })
    
    prediction = model.predict(infer_data)
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict EE or Yield using pre-trained model")
    parser.add_argument("--model_type", type=str, required=True, choices=['ee', 'yield'], help="Model type to use for prediction (ee or yield)")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature for the reaction")
    parser.add_argument("--ligand", type=str, required=True, help="Ligand for the reaction")
    parser.add_argument("--base", type=str, required=True, help="Base for the reaction")
    parser.add_argument("--solvent", type=str, required=True, help="Solvent for the reaction")
    parser.add_argument("--ligand_smiles", type=str, required=True, help="SMILES string for the ligand")

    args = parser.parse_args()

    # 获取参数
    model_type = args.model_type
    temperature = args.temperature
    ligand = args.ligand
    base = args.base
    solvent = args.solvent
    ligand_smiles = args.ligand_smiles

    # 进行预测
    prediction = predict(model_type, temperature, ligand, base, solvent, ligand_smiles)
    print(f'Predicted {model_type.capitalize()}: {prediction}')

