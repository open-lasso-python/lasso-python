
from enum import Enum


class FilterType(Enum):
    """ Used for filtering d3plot arrays

    Use PART, BEAM, SHELL, SOLID or TSHELL

    Examples
    --------
        >>> part_ids = [13, 14]
        >>> d3plot.get_part_filter(FilterType.SHELL, part_ids)
    """

    BEAM = "beam"  #:
    SHELL = "shell"  #:
    SOLID = "solid"  #:
    TSHELL = "tshell"  #:
    PART = "part"  #:
    NODE = "node"  # :
