from enum import Enum


class FilterType(Enum):
    """Used for filtering d3plot arrays

    Attributes
    ----------
    BEAM: str
        Filters for beam elements
    SHELL: str
        Filters for shell elements
    SOLID: str
        Filters for solid elements
    TSHELL: str
        Filters for thick shells elements
    PART: str
        Filters for parts
    NODE: str
        Filters for nodes

    Examples
    --------
        >>> part_ids = [13, 14]
        >>> d3plot.get_part_filter(FilterType.SHELL, part_ids)
    """

    BEAM = "beam"
    SHELL = "shell"
    SOLID = "solid"
    TSHELL = "tshell"
    PART = "part"
    NODE = "node"
