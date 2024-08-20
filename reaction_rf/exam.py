# 读取数据集并检查'reaction_smiles'列的格式
file_path = '/mnt/data/data.tsv'
df = pd.read_csv(file_path, sep='\t')

# 检查'reaction_smiles'列是否都是'A.B.C>>D'这种形式
def check_format(smiles):
    if '>>' not in smiles:
        return False
    reactants, products = smiles.split('>>')
    reactants_parts = reactants.split('.')
    return len(reactants_parts) > 0 and len(products) > 0

reaction_smiles_format = df['reaction_smiles'].apply(check_format)
valid_format = reaction_smiles_format.all()

# 找出不符合'A.B.C>>D'这种形式的条目
invalid_reactions = df[~reaction_smiles_format]


