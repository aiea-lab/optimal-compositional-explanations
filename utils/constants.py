"""Constants used in the repository."""

NETDISSECT_QUANTILE = 0.005
# BATCH_SIZE = 32
# EPSILON = 1e-06


# INDICES FOR QUANTITIES
TOP_INDEX_SAMPLE = 0
BOTTOM_INDEX_SAMPLE = 1
TOP_INDEX_SUM = 2
BOTTOM_INDEX_SUM = 3

INDEX_INDIVIDUAL = 0
INDEX_OR = 1
INDEX_AND = 2
INDEX_NOT = 3

INDEX_NODE_IOU_ESTI = 0
INDEX_NODE_NEXT_OP = 1
INDEX_NODE_LABEL = 2
INDEX_NODE_OPS = 3

INDEX_TUPLE_MAX = 0
INDEX_TUPLE_MIN = 1
INDEX_TUPLE_SAMPLE = 0
INDEX_TUPLE_SUM = 1

QUANTITIES = [
    "common_intersection",
    "unique_intersection",
    "common_extras",
    "unique_extras",
    "common_uncovered",
    "unique_uncovered",
]
