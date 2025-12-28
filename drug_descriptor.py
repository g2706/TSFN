import ssl

import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
import os


# 获取药物信息
def get_drug_info(drug_name):
    ssl._create_default_https_context = ssl._create_unverified_context
    compounds = pcp.get_compounds(drug_name, 'name')

    # 如果没有找到药物，给出提示并返回
    if not compounds:
        print(f"Error: Could not find information for '{drug_name}' in PubChem.")
        return None, None

    compound = compounds[0]

    # 获取药物的基本信息
    smiles = compound.canonical_smiles
    iupac_name = compound.iupac_name
    molecular_formula = compound.molecular_formula
    cid = compound.cid

    # 获取 description 属性，如果不存在则给它一个默认值
    description = getattr(compound, 'description', 'No description available')

    # 输出药物信息
    print(f"Drug Name: {drug_name}")
    print(f"IUPAC Name: {iupac_name}")
    print(f"Molecular Formula: {molecular_formula}")
    print(f"Canonical SMILES: {smiles}")
    print(f"PubChem CID: {cid}")
    print(f"Description: {description}")

    return smiles, compound


# 计算并返回分子描述符作为向量
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 计算常见的分子描述符
        mol_weight = Descriptors.MolWt(mol)  # 分子量
        logP = Crippen.MolLogP(mol)  # LogP
        num_h_donors = Descriptors.NumHDonors(mol)  # 氢键供体数量
        num_h_acceptors = Descriptors.NumHAcceptors(mol)  # 氢键受体数量
        heavy_atoms = Descriptors.HeavyAtomCount(mol)  # 重原子数
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)  # 可旋转键数
        tpsa = Descriptors.TPSA(mol)  # 极性表面积
        lipinski_ro5 = (
            Descriptors.MolWt(mol) <= 500,  # MolWt <= 500
            Crippen.MolLogP(mol) <= 5,  # LogP <= 5
            Descriptors.NumHDonors(mol) <= 5,  # Hydrogen Bond Donors <= 5
            Descriptors.NumHAcceptors(mol) <= 10  # Hydrogen Bond Acceptors <= 10
        )

        # 额外的一些描述符
        exact_mol_weight = Descriptors.ExactMolWt(mol)  # 精确分子量
        n_aromatic_rings = Descriptors.NumAromaticRings(mol)  # 芳香环数
        fractional_csp3 = Descriptors.FractionCSP3(mol)  # sp3碳的占比

        # 将所有描述符放入一个向量
        descriptor_vector = [
            mol_weight,  # 分子量
            logP,  # LogP
            num_h_donors,  # 氢键供体数
            num_h_acceptors,  # 氢键受体数
            heavy_atoms,  # 重原子数
            rotatable_bonds,  # 可旋转键数
            tpsa,  # 极性表面积
            lipinski_ro5[0],  # Lipinski's Rule of Five: MolWt <= 500
            lipinski_ro5[1],  # Lipinski's Rule of Five: LogP <= 5
            lipinski_ro5[2],  # Lipinski's Rule of Five: HBD <= 5
            lipinski_ro5[3],  # Lipinski's Rule of Five: HBA <= 10
            exact_mol_weight,  # 精确分子量
            n_aromatic_rings,  # 芳香环数
            fractional_csp3  # sp3碳的占比
        ]

        # 返回描述符向量
        return descriptor_vector
    else:
        print("Invalid SMILES string.")
        return None


# 2D结构图绘制
def draw_2d_structure(smiles, output_filename):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 确保保存路径存在，如果不存在则创建
        directory = os.path.dirname(output_filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        from rdkit.Chem import Draw
        img = Draw.MolToImage(mol)
        img.save(output_filename)
        print(f"2D structure image saved as '{output_filename}'")
    else:
        print("Invalid SMILES string.")


# 主程序
def main():
    # 获取用户输入的药物名称
    drug_name = input("Enter the drug name (e.g., insulin): ").strip()

    if not drug_name:
        print("Drug name cannot be empty. Please try again.")
        return

    # 获取药物信息
    smiles, compound = get_drug_info(drug_name)

    # 如果没有获取到药物信息，终止后续操作
    if compound is None:
        return

    # 计算并输出分子描述符
    descriptor_vector = calculate_descriptors(smiles)

    if descriptor_vector:
        # 打印描述符向量及其含义
        print("\nDescriptor Vector:")
        descriptors = [
            ("Molecular Weight", descriptor_vector[0]),
            ("LogP", descriptor_vector[1]),
            ("Hydrogen Bond Donors", descriptor_vector[2]),
            ("Hydrogen Bond Acceptors", descriptor_vector[3]),
            ("Heavy Atom Count", descriptor_vector[4]),
            ("Rotatable Bonds", descriptor_vector[5]),
            ("Polar Surface Area (TPSA)", descriptor_vector[6]),
            ("Lipinski's Rule of Five (MolWt <= 500)", descriptor_vector[7]),
            ("Lipinski's Rule of Five (LogP <= 5)", descriptor_vector[8]),
            ("Lipinski's Rule of Five (HBD <= 5)", descriptor_vector[9]),
            ("Lipinski's Rule of Five (HBA <= 10)", descriptor_vector[10]),
            ("Exact Molecular Weight", descriptor_vector[11]),
            ("Number of Aromatic Rings", descriptor_vector[12]),
            ("Fraction of sp3 Carbons", descriptor_vector[13])
        ]

        for desc, value in descriptors:
            print(f"{desc}: {value}")

        # 打印描述符向量
        print(descriptor_vector)  # 这里添加打印描述符向量的代码

    # 2D结构图的输出文件名
    output_file_2d = f"images/{drug_name}_2D_structure.png"
    draw_2d_structure(smiles, output_file_2d)


if __name__ == "__main__":
    main()