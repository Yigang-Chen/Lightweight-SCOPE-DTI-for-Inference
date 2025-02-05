import torch
import numpy as np
import pandas as pd
import torch_geometric
import torch_cluster
from rdkit import Chem

ATOM_VOCAB = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'unk']

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def onehot_encoder(a=None, alphabet=None, default=None, drop_first=False):
    '''
    Parameters
    ----------
    a: array of numerical value of categorical feature classes.
    alphabet: valid values of feature classes.
    default: default class if out of alphabet.
    Returns
    -------
    A 2-D one-hot array with size |x| * |alphabet|
    '''
    # replace out-of-vocabulary classes

    alphabet_set = set(alphabet)
    a = [x if x in alphabet_set else default for x in a]

    # cast to category to force class not present
    a = pd.Categorical(a, categories=alphabet)

    onehot = pd.get_dummies(pd.Series(a), columns=alphabet, drop_first=drop_first)
    return onehot.values


def _build_atom_feature(mol):
    # dim: 44 + 7 + 7 + 7 + 1
    feature_alphabet = {
        #(alphabet, default value)
        'GetSymbol': (ATOM_VOCAB, 'unk'),
        'GetDegree': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetTotalNumHs': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetImplicitValence': ([0, 1, 2, 3, 4, 5, 6], 6),
        #原'GetIsAromatic': ([0, 1], 1) #似乎有错误
        'GetIsAromatic': ([False, True], True)  # 调整为布尔类型

    }

    atom_feature = None
    for attr in ['GetSymbol', 'GetDegree', 'GetTotalNumHs',
                'GetImplicitValence', 'GetIsAromatic']:
        feature = [getattr(atom, attr)() for atom in mol.GetAtoms()]
        feature = onehot_encoder(feature,
                    alphabet=feature_alphabet[attr][0],
                    default=feature_alphabet[attr][1],
                    drop_first=(attr in ['GetIsAromatic']) # binary-class feature
                )
        atom_feature = feature if atom_feature is None else np.concatenate((atom_feature, feature), axis=1)
    atom_feature = atom_feature.astype(np.float32)
    return atom_feature

def _build_edge_feature(coords, edge_index, D_max=4.5, num_rbf=16):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


def sdf_to_graphs(sdf_content, edge_cutoff=4.5, num_rbf=16):
    """
    Parameters
    ----------
    sdf_content: str
        Content of sdf file
    Returns
    -------
    graph: torch_geometric.data.Data
        A torch_geometric graph
    """

    mol = Chem.MolFromMolBlock(sdf_content)
    conf = mol.GetConformer()
    with torch.no_grad():
        coords = conf.GetPositions()
        coords = torch.as_tensor(coords, dtype=torch.float32)
        atom_feature = _build_atom_feature(mol)
        atom_feature = torch.as_tensor(atom_feature, dtype=torch.float32)
        edge_index = torch_cluster.radius_graph(coords, r=edge_cutoff)

    node_s = atom_feature
    node_v = coords.unsqueeze(1)
    edge_s, edge_v = _build_edge_feature(coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)

    data = torch_geometric.data.Data(
        x=coords, edge_index=edge_index,
        node_v=node_v, node_s=node_s, edge_v=edge_v, edge_s=edge_s)

    return data

