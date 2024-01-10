import numpy as np
from typing import List

def factorize(n: int) -> List[int]:
    """Return the most average two factorization of n."""
    for i in range(int(np.sqrt(n)) + 1, 1, -1):
        if n % i == 0:
            return [i, n // i]
    return [n, 1]

def group_factorize(n: List[int]) -> List[List[int]]:
    decompose_list1 = []
    decompose_list2 = []
    for i in n:
        decompose_list1.append(factorize(i)[0])
        decompose_list2.append(factorize(i)[1])
    return decompose_list1, decompose_list2

if __name__ == '__main__':
    print(factorize(100))
    print(group_factorize([100, 200, 300]))