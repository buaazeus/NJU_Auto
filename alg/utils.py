# -*- coding: utf-8 -*-


def cmp_strnumber(s):
    """Sort the list s by string, which only contains digit.

    Args:
        s: an unsorted list

    Returns:
        v: the numerical value of string.

    """
    s = s[6:]  # remove prefix "model-"
    v = 0
    c = 1
    for i in range(len(s))[::-1]:
        v += (ord(s[i]) - 48) * c
        c *= 10
    return v


def get_model_name(l):
    """Get model name from list and ignore the name which contains non-numerals.

    Args:
        l: a list

    Returns:
        ans: a list

    """
    ans = []
    for i in range(len(l)):
        if l[i].startswith("model-"):
            ans.append(l[i])
    return ans
