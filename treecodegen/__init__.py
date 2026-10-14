"""Prompt -> line-level syntax tree, with order-free generation and random node IDs."""
from .tree import Node, canonical, code_to_tree, nodes_to_tree, tree_to_code, tree_to_nodes
from .vocab import Vocab
