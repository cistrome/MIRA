
import numpy as np
from scipy.cluster.hierarchy import to_tree

def centered_boxcox_transform(x, a = 'log'):

    if a == 'log' or a == 0:
        return np.log(x) - np.log(x).mean(-1, keepdims = True)

    else:

        assert(isinstance(a, float) and a > 0 and a < 1)
        x = (x**a)/(x**a).mean(-1, keepdims = True)
        return ( x - 1 )/a


def gram_schmidt_basis(n):
    basis = np.zeros((n, n-1))
    for j in range(n-1):
        i = j + 1
        e = np.array([(1/i)]*i + [-1] +
                    [0]*(n-i-1))*np.sqrt(i/(i+1))
        basis[:, j] = e
    return basis


def get_hierarchical_gram_schmidt_basis(num_topics, linkage_matrix):
        
    root, nodes = to_tree(linkage_matrix, rd = True)

    def children(x):

        if x is None:
            return []

        if x.left is None and x.right is None:
            return [x.id]

        return children(x.left) + children(x.right)
    
    balance_matrix = np.zeros((num_topics-1, num_topics))
    
    n_plus, n_minus = [],[]
    for i, node in enumerate(nodes[num_topics:]):
        balance_matrix[i, children(node.left)] = 1
        balance_matrix[i, children(node.right)] = -1
        n_plus.append(len(children(node.left)))
        n_minus.append(len(children(node.right)))

    n_plus, n_minus = np.array(n_plus), np.array(n_minus)

    plus_sites = 1/n_plus * np.sqrt((n_plus*n_minus)/(n_plus + n_minus))
    minus_sites = -1/n_minus * np.sqrt((n_plus*n_minus)/(n_plus + n_minus))

    g_matrix = np.zeros_like(balance_matrix) + (balance_matrix == 1).astype(int) * plus_sites[:,np.newaxis] +\
        (balance_matrix == -1).astype(int) * minus_sites[:,np.newaxis]

    return g_matrix.T


'''def create_joint_space(
    topics1, topics2,
    dendogram1 = None, dendogram2 = None,
    style = 'arbitrary', box_cox = 0.5,
):

    assert(style in ['arbitrary','hierarchical'])

    if style == 'hierarchical':

        assert(not dendogram1 is None and not dendogram2 is None)
        basis = np.vstack([
            get_hierarchical_gram_schmidt_basis(topics1.shape[-1], dendogram1),
            get_hierarchical_gram_schmidt_basis(topics2.shape[-2], dendogram2),
        ])

    else:
        basis = np.vstack([
            gram_schmidt_basis(topics1.shape[-1]),
            gram_schmidt_basis(topics2.shape[-1])
        ])

    joint_topics = np.hstack([topics1, topics2])

    print(basis.shape, joint_topics.shape)

    return centered_boxcox_transform(joint_topics, a = box_cox).dot(basis)'''