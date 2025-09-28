import torch


def distance(embedding, index):
    dis = torch.norm(embedding[index] - embedding[0])
    for i in range(1,index):
        cur_dis = torch.norm(embedding[index] - embedding[i])
        if(cur_dis<dis):
            dis=cur_dis
    return dis


def top_x_indices(vec, x, largest):
    """
    Returns the indices of the x largest/smallest entries in vec.

    Args:
        vec: tensor, number of samples to be selected
        x: int, number of indices to be returned
        smallest: bool, if true, the x largest entries are selected; if false,
        the x smallest entries are selected
    Returns:
        top_x_indices: tensor, top x indices, sorted
        other_indices: tensor, the indices that were not selected
    """

    sorted_idx = torch.argsort(vec, descending=largest)

    top_x_indices = sorted_idx[:x]
    other_indices = sorted_idx[x:]

    return top_x_indices, other_indices


def top_x_distant_indices(vec, x, largest, embedding, dis):
    sorted_idx = torch.argsort(vec, descending=largest)
    L=len(sorted_idx)

    embedding = embedding[sorted_idx]
    i=1
    top_x_indices = sorted_idx[:i]
    other_indices = sorted_idx[i:i]

    j=0
    K=L
    while(i<x) and (j<L) and (i<K):
        j+=1
        if(distance(embedding, i)>dis):
            top_x_indices = torch.cat([top_x_indices, sorted_idx[j:j+1]])
            i+=1
        else:
            other_indices = torch.cat([other_indices, sorted_idx[j:j+1]])
            embedding = torch.cat([embedding[:i], embedding[i+1:]])
            K-=1
    other_indices = torch.cat([other_indices, sorted_idx[j+1:]])
    return top_x_indices, other_indices


def create_logging_dict(variables_to_log, selected_minibatch, not_selected_minibatch):
    metrics_to_log = {}
    for name, metric in variables_to_log.items():
        metrics_to_log["selected_" + name] = metric[selected_minibatch].cpu().numpy()
        if not_selected_minibatch is None or len(not_selected_minibatch) == 0:
            metrics_to_log["not_selected_" + name] = []
        else:
            metrics_to_log["not_selected_" + name] = metric[not_selected_minibatch].cpu().numpy()
    return metrics_to_log
