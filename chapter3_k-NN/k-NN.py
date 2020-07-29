import numpy as np


class TreeNode:
    """
    二叉树结构
    val: 节点的值
    d: 节点所在层数对应的切分维度; d = j%k,j为层深度,k为特征总维度
    """

    def __init__(self, val, d):
        self.val = val
        self.d = d
        self.left = None
        self.right = None


def quicksort(arr, l, r, d):
    """
    针对某一维度的快排

    Args:
        arr: 待排序数组, 数组内元素是维度相同的多维数组
        l: 待排序的左边界
        r: 待排序的右边界
        d: 排序的维度，即多维数组按照第d维进行排序

    return:
        arr: 按照第d维排序后的数组
    """
    if not (0 <= l < len(arr) and 0 <= r < len(arr) and l < r):
        return arr
    i, j = l, r
    temp = arr[i]
    while i < j:
        while i < j and arr[j][d] > temp[d]:
            j -= 1
        if i < j:
            arr[i] = arr[j]
            i += 1
        while i < j and arr[i][d] < temp[d]:
            i += 1
        if i < j:
            arr[j] = arr[i]
            j -= 1
    arr[i] = temp
    quicksort(arr, l, i - 1, d)
    quicksort(arr, i + 1, r, d)
    return arr


def kdTree(arr, d, dim):
    """
    构建kd树

    Args:
        arr: 训练特征数组, 数组内元素是维度相同的多维数组
        d: 节点所在层数对应的切分维度
        dim: 特征总维度

    return:
        root: kd树结构, root为树的根节点
    """
    if not arr:
        return None
    mid = len(arr) // 2
    arr = quicksort(arr, 0, len(arr) - 1, d)
    root = TreeNode(arr[mid], d)
    root.left = kdTree(arr[:mid], (d + 1) % dim, dim)
    root.right = kdTree(arr[mid + 1:], (d + 1) % dim, dim)
    return root


def distance(val1, val2):
    """
    欧式距离计算
    """
    val1 = np.array(val1)
    val2 = np.array(val2)
    dis = np.sqrt(np.sum((val1 - val2) ** 2))
    return dis


def NN(node, target, res):
    """
    Nearest Neighbor(最近邻)

    Args:
        node: 当前结点
        target: 待判断的结点
        res: 返回数组，res = [最近的结点值，最近的距离]

    return:
        res


    """
    if not node:
        return res
    d = node.d

    # 递归找到target所在区域对应的叶子结点
    if target[d] < node.val[d]:
        res = NN(node.left, target, res)
    else:
        res = NN(node.right, target, res)

    # 计算当前结点到target的距离
    dis = distance(node.val, target)
    if dis < res[1]:
        res = [node.val, dis]

    # 判断最近点是否可能存在于另一子树，即当前结点的最近邻范围内是否包含另一区域
    if node.left and target[d] > node.val[d] and target[d] - res[1] < node.val[d]:
        res = NN(node.left, target, res)
    if node.right and target[d] < node.val[d] and target[d] + res[1] > node.val[d]:
        res = NN(node.right, target, res)
    return res

# k-Nearest Neighbor(K近邻)


def kNN(k, node, val, res):
    """
        Nearest Neighbor(最近邻)

        Args:
                k: 近邻的个数
            node: 当前结点
            target: 待判断的结点
            res: 返回数组，res = [k近邻的结点值数组，对应的结点距离的数组]

        return:
            res
    """
    if not node:
        return res
    d = node.d
    if val[d] < node.val[d]:
        res = kNN(k, node.left, val, res)
    else:
        res = kNN(k, node.right, val, res)

    # 判断当前距离是否比k近邻表中的最大值小，小则更新k近邻表
    dis = distance(node.val, val)
    if len(res[0]) < k:
        res[0].append(node.val)
        res[1].append(dis)
    elif dis < max(res[1]):
        ind = res[1].index(max(res[1]))
        res[0][ind] = node.val
        res[1][ind] = dis

    if node.left and val[d] > node.val[d] and val[d] - max(res[1]) < node.val[d]:
        res = kNN(k, node.left, val, res)
    if node.right and val[d] < node.val[d] and val[d] + max(res[1]) > node.val[d]:
        res = kNN(k, node.right, val, res)
    return res


if __name__ == '__main__':
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    dim = len(data[0])
    root = kdTree(data, 0, dim)

    res = [[], np.Inf]
    res = NN(root, [6, 3], res)
    print(res[0], res[1])

    k_res = [[], []]
    k_res = kNN(2, root, [6, 3], k_res)
    print(k_res[0], k_res[1])
