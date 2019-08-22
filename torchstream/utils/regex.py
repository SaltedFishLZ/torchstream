"""
"""
import re

def match_first(regex, string):
    """
    - regex
    - string
    """
    return re.findall(regex + "|$", string)[0]