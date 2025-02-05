from yacs.config import CfgNode as CN

_C = CN()  # create a node

# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_DIM = [66, 1]
_C.DRUG.NODE_H_DIM = [128, 64]
_C.DRUG.EDGE_IN_DIM = [16, 1]
_C.DRUG.EDGE_H_DIM = [32, 1]
_C.DRUG.NUM_LAYERS = 3
_C.DRUG.DROP_RATE = 0.1

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 1

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 200
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048


def get_cfg_defaults():
    return _C.clone()
