import torch
from collections import namedtuple
from .. import DATA_TYPE_REGISTRY
from ..dataloader import Dataset

def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)
    deg_t = adj_t.sum(dim=0)
    deg_t_inv_sqrt = deg_t.pow(-0.5)
    deg_t_inv_sqrt.masked_fill_(torch.isinf(deg_t_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_t_inv_sqrt.view(1, -1))

    edge_index = adj_t.to_sparse()
    return edge_index

FullGraphData = namedtuple("FullGraphData", ["u_edge", "v_edge",
                                             "interact_mat","interact_mat_t", "label", "interaction_pair", "valid_mask"])

@DATA_TYPE_REGISTRY.register()
class FullGraphDataset(Dataset):
    def __init__(self, dataset, mask, fill_unkown=True, **kwargs):
        super(FullGraphDataset, self).__init__(dataset, mask, fill_unkown=True, **kwargs)
        assert fill_unkown, "fill_unkown need True!"
        self.data = self.build_data()

    def build_data(self):
        # v is disease, u is drug
        u_edge = self.get_u_edge(union_graph=False)
        v_edge = self.get_v_edge(union_graph=False)
        uv_edge = self.get_uv_edge(union_graph=False)
        vu_edge = self.get_vu_edge(union_graph=False)

        interact_mat = torch.sparse_coo_tensor(indices=vu_edge[0], values=vu_edge[1], size=(self.size_v, self.size_u))
        interact_mat_t = torch.sparse_coo_tensor(indices=uv_edge[0], values=uv_edge[1], size=(self.size_u, self.size_v))
        # x = x.to_dense()
        u_mat = torch.sparse_coo_tensor(indices=u_edge[0], values=u_edge[1], size=(self.size_u, self.size_u))
        v_mat = torch.sparse_coo_tensor(indices=v_edge[0], values=v_edge[1], size=(self.size_v, self.size_v))

        interact_mat = gcn_norm(edge_index=interact_mat, add_self_loops=False)
        interact_mat_t = gcn_norm(edge_index=interact_mat_t, add_self_loops=False)
        u_mat = gcn_norm(edge_index=u_mat, add_self_loops=False)
        v_mat = gcn_norm(edge_index=v_mat, add_self_loops=False)
        # uv_edge = norm_uv_edge * torch.norm(uv_edge) / torch.norm(norm_uv_edge)
        data = FullGraphData(u_edge=u_mat,
                             v_edge=v_mat,
                             interact_mat=interact_mat,
                             interact_mat_t=interact_mat_t,
                             label=self.label,
                             valid_mask=self.valid_mask,
                             interaction_pair=self.interaction_edge
                             )
        return data

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.data