from typing import Any, Dict, Set, Tuple, Union

# Custom types for type hints
NodeInfoDict = Dict[str, Dict[str, Any]]
AdjacencyDict = Dict[str, Union[Set[str], Set[Tuple[str, Tuple[str]]]]]