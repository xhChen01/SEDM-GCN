import torch
from torch import nn, optim
import torch.nn.functional as F
from functools import partial
from .. import MODEL_REGISTRY
from ..model_help import BaseModel
from .dataset import FullGraphData



class Aggregator(nn.Module):

    def __init__(self, n_drugs, n_diseases, n_factors):
        super(Aggregator, self).__init__()
        self.n_drugs = n_drugs
        self.n_diseases = n_diseases
        self.n_factors = n_factors

    def forward(self, dis_emb, dr_emb, latent_emb, di_lantent_weight, dr_lantent_weight, interact_mat, interact_mat_t,
                u_edge, v_edge):

        dim = dis_emb.shape[1]
        n_factors = self.n_factors

        """dis aggregate  interact_mat is disease-drug """
        dis_agg = torch.mm(interact_mat, dr_emb)  # [n_diseases, dim]
        disen_weight = latent_emb.expand(self.n_diseases, n_factors, dim)
        dis_agg = dis_agg * ((disen_weight )*di_lantent_weight.unsqueeze(-1)).sum(dim=1) + dis_agg # [n_diseases, dim]

        """drug aggregate"""
        drug_agg = torch.mm(interact_mat_t, dis_emb)  # [n_diseases, dim]
        # 直接使用带系数的权重值，而不使用注意力机制
        drug_weight = latent_emb.expand(self.n_drugs, n_factors, dim)
        drug_agg = drug_agg * ((drug_weight )*dr_lantent_weight.unsqueeze(-1)).sum(dim=1) + drug_agg  # [n_diseases, dim]

        return dis_agg, drug_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, latent_emb, n_hops, n_drugs, n_diseases,
                 n_factors,ind):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.latent_emb = latent_emb
        self.n_diseases = n_diseases
        self.n_drugs = n_drugs
        self.n_factors = n_factors
        self.ind = ind
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        self.temperature = 0.2

        for i in range(n_hops):
            self.convs.append(Aggregator(n_drugs = n_drugs, n_diseases=n_diseases, n_factors=n_factors))

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [dim]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [dim]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            dim = tensor_1.shape[0]
            zeros = torch.zeros(dim, dim).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [dim, dim]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [dim, dim]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / dim ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / dim ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / dim ** 2, zero) + 1e-8)
            score = dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
            return score[0]
        def MutualInformation():
            # disen_T: [num_factor, dimension]
            latent_emb_T = self.latent_emb.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_latent_emb_T = latent_emb_T / latent_emb_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_latent_emb_T * normalized_latent_emb_T, dim=1)
            ttl_scores = torch.sum(torch.mm(latent_emb_T, self.latent_emb), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        elif self.ind == "None":
            cor = 0
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.latent_emb[i], self.latent_emb[j])
                    else:
                        cor += CosineSimilarity(self.latent_emb[i], self.latent_emb[j])
        return cor

    def forward(self, dis_emb, dr_emb, latent_emb, di_lantent_weight, dr_lantent_weight, interact_mat, interact_mat_t,
                u_edge, v_edge, di_emb_sim, dr_emb_sim):

        dis_res_emb = torch.cat((F.normalize(dis_emb), F.normalize(di_emb_sim)), -1)
        drug_res_emb = torch.cat((F.normalize(dr_emb), F.normalize(dr_emb_sim)), -1)
        # 用于计算加权的损失函数
        cor = self._cul_cor()
        for i in range(len(self.convs)):
            dis_emb, dr_emb = self.convs[i](dis_emb, dr_emb, latent_emb, di_lantent_weight, dr_lantent_weight,
                                            interact_mat, interact_mat_t, u_edge, v_edge)
            # v_edge的值为SMILES计算出的相似度，通过sim_threhold保留高于阈值的邻居。
            di_emb_sim = torch.mm(v_edge, di_emb_sim)
            dr_emb_sim = torch.mm(u_edge, dr_emb_sim)
            """result emb"""
            dis_res_emb = torch.cat((dis_res_emb, F.normalize(dis_emb), F.normalize(di_emb_sim)), -1)
            drug_res_emb = torch.cat((drug_res_emb, F.normalize(dr_emb), F.normalize(dr_emb_sim)), -1)

        return dis_res_emb, drug_res_emb, cor

@MODEL_REGISTRY.register()
class Recommender(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SEPN model config")
        parser.add_argument('--dim', type=int, default=256, help='embedding size')
        parser.add_argument('--sim_dim', type=int, default=256, help='sim embedding size')
        parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--factor_regularity', type=float, default=1e-6,
                            help='regularization weight for latent factor')
        parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
        parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
        parser.add_argument("--n_factors", type=int, default=2, help="number of pathogenic factors")
        parser.add_argument("--ind", type=str, default='consine', help="Independence modeling: mi, distance, cosine")
        parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
        parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
        parser.add_argument('--context_hops', type=int, default=7 , help='number of context hops')

        return parent_parser

    def __init__(self, n_diseases, n_drugs, lr, l2, factor_regularity, dim, sim_dim, context_hops, n_factors,
                 ind, pos_weight, **kwargs):
        super(Recommender, self).__init__()

        self.n_diseases = n_diseases
        self.n_drugs = n_drugs
        self.lr = lr
        self.decay = l2
        self.fact_decay = factor_regularity
        self.emb_size = dim
        self.sim_dim = sim_dim
        self.context_hops = context_hops
        self.n_factors = n_factors
        self.ind = ind
        # pos_weight = neg_num / pos_num
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.loss_fn = partial(self.bce_loss_fn, pos_weight=self.pos_weight)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.all_embed_sim = nn.Parameter(self.all_embed_sim)
        self.latent_emb = nn.Parameter(self.latent_emb)
        self.dr_lantent_weight = nn.Parameter(self.dr_lantent_weight)
        self.di_lantent_weight = nn.Parameter(self.di_lantent_weight)

        self.gcn = self._init_model()

        self.save_hyperparameters()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_diseases+self.n_drugs, self.emb_size))
        self.all_embed_sim = initializer(torch.empty(self.n_diseases+self.n_drugs, self.sim_dim))
        # n_factors为病因因子数
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))
        # dr_lantent_weight 为药物的病因因子系数
        self.dr_lantent_weight = initializer(torch.empty(self.n_drugs, self.n_factors))
        # di_lantent_weight 为药物的病因因子系数
        self.di_lantent_weight = initializer(torch.empty(self.n_diseases, self.n_factors))

        # [n_diseases, n_entities]
        # self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        # ind : Independence modeling: mi, distance, cosine
        return GraphConv(latent_emb=self.latent_emb,
                         n_hops=self.context_hops,
                         n_drugs=self.n_drugs,
                         n_diseases=self.n_diseases,
                         n_factors=self.n_factors,
                         ind=self.ind)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().cuda(), type.long().cuda()

    def forward(self, interact_mat, interact_mat_t, u_edge, v_edge):

        di_emb = self.all_embed[:self.n_diseases, :]
        dr_emb = self.all_embed[self.n_diseases:, :]
        di_emb_sim = self.all_embed_sim[:self.n_diseases, :]
        dr_emb_sim = self.all_embed_sim[self.n_diseases:, :]

        # entity_gcn_emb: [n_entity, dim]
        # disease_gcn_emb: [n_diseases, dim]
        dis_gcn_emb, drug_gcn_emb, cor = self.gcn(di_emb,
                                                     dr_emb,
                                                     self.latent_emb,
                                                     self.di_lantent_weight,
                                                     self.dr_lantent_weight,
                                                     interact_mat,
                                                     interact_mat_t,
                                                     u_edge,
                                                     v_edge,di_emb_sim,dr_emb_sim)
        predict = torch.sigmoid(torch.matmul(drug_gcn_emb, dis_gcn_emb.t()))

        return predict, dis_gcn_emb, drug_gcn_emb, cor

    def step(self, batch:FullGraphData):
        label = batch.label
        predict, u, v, cor = self.forward(batch.interact_mat, batch.interact_mat_t, batch.u_edge, batch.v_edge)

        # 将下面的if条件屏蔽掉，就可以读取最后一次的测试结果
        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)]
            label = label[batch.valid_mask]
        ans = self.loss_fn(predict=predict, label=label)
        if self.training:
            ans['loss'] += self.fact_decay * cor
        ans["predict"] = predict.reshape(-1)
        ans["label"] = label.reshape(-1)
        return ans

    def training_step(self, batch, batch_idx=None):

        return self.step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr= self.lr, weight_decay=self.decay)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr, max_lr=10*self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]