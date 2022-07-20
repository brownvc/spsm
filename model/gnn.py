import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class Part():
    def __init__(self):
        pass

class Linear3(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None):
        super(Linear3, self).__init__()
        if hidden_size is None:
            hidden_size = max(in_size, out_size)

        self.model = nn.Sequential(
                        nn.Linear(in_size, hidden_size),
                        nn.LeakyReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.LeakyReLU(),
                        nn.Linear(hidden_size, out_size),
                        )

    def forward(self, x):
        return self.model(x)

class Propagator(nn.Module):
    """
    Propagate information across nodes
    """
    def __init__(self, gn, rounds_of_propagation=10, dual_passing=True):
        super(Propagator, self).__init__()
        self.rounds_of_propagation = rounds_of_propagation

        node_size = gn.node_size
        self.node_size = node_size
        edge_size = gn.edge_size

        message_size = node_size * 4 #+ edge_size

        self.dual_passing = dual_passing

        # Models for propagating the state vector
        #Gathering message from an adjacent node
        #Separate weights for rounds_of_propagation rounds

        #Forward direction
        self.f_ef = nn.ModuleList([
            Linear3(message_size, node_size)
        for i in range(rounds_of_propagation)])

        if self.dual_passing:
            self.f_ef2 = nn.ModuleList([
                Linear3(message_size, node_size)
            for i in range(rounds_of_propagation)])

        self.f_n = nn.ModuleList([
            Linear3(node_size * 2, node_size)
        for i in range(rounds_of_propagation)])

        if self.dual_passing:
            self.f_n2 = nn.ModuleList([
                Linear3(node_size * 2, node_size)
            for i in range(rounds_of_propagation)])

    def forward(self, gn):
        for i in range(self.rounds_of_propagation):
            aggregated = torch.zeros(gn.node_vectors.size()[0], self.node_size).to(gn.node_vectors.device)
            if gn.u_indices is not None:
                intra_idxs = torch.where(gn.edge_vectors.squeeze(1) == 0)
                
                u1 = gn.u_indices[intra_idxs]
                v1 = gn.v_indices[intra_idxs]

                messages_raw = torch.cat((gn.node_vectors[u1],
                                gn.node_vectors[v1],
                                gn.node_vectors_initial[u1],
                                gn.node_vectors_initial[v1],
                                ), 1)

                messages_forward = self.f_ef[i](messages_raw)

                index = torch.LongTensor(u1).to(messages_raw.device)
                aggregated = scatter_add(messages_forward, index, out=aggregated, dim=0)

                messages_raw = torch.cat((gn.node_vectors[v1],
                                gn.node_vectors[u1],
                                gn.node_vectors_initial[v1],
                                gn.node_vectors_initial[u1],
                                ), 1)

                messages_forward = self.f_ef[i](messages_raw)

                index2 = torch.LongTensor(v1).to(messages_raw.device)
                aggregated = scatter_add(messages_forward, index2, out=aggregated, dim=0)

            gn.node_vectors = self.f_n[i](torch.cat([aggregated, gn.node_vectors], dim=1))

            if self.dual_passing:
                aggregated = torch.zeros(gn.node_vectors.size()[0], self.node_size).to(gn.node_vectors.device)
                if gn.u_indices is not None:
                    inter_idxs = torch.where(gn.edge_vectors.squeeze(1) == 1)

                    if inter_idxs[0].shape[0] > 0:
                        u1 = gn.u_indices[inter_idxs]
                        v1 = gn.v_indices[inter_idxs]

                        messages_raw = torch.cat((gn.node_vectors[u1],
                                        gn.node_vectors[v1],
                                        gn.node_vectors_initial[u1],
                                        gn.node_vectors_initial[v1],
                                        ), 1)

                        messages_forward = self.f_ef2[i](messages_raw)

                        index = torch.LongTensor(u1).to(messages_raw.device)
                        aggregated = scatter_add(messages_forward, index, out=aggregated, dim=0)

                        messages_raw = torch.cat((gn.node_vectors[v1],
                                        gn.node_vectors[u1],
                                        gn.node_vectors_initial[v1],
                                        gn.node_vectors_initial[u1],
                                        ), 1)

                        messages_forward = self.f_ef2[i](messages_raw)

                        index2 = torch.LongTensor(v1).to(messages_raw.device)
                        aggregated = scatter_add(messages_forward, index2, out=aggregated, dim=0)

                gn.node_vectors = self.f_n2[i](torch.cat([aggregated, gn.node_vectors], dim=1))

            gn.all_node_vectors.append(gn.node_vectors)

class Aggregator(nn.Module):
    """
    Aggregates information across nodes to create a graph vector
    """
    def __init__(self, gn, iters):
        super(Aggregator, self).__init__()
        self.iters = iters
        node_size = gn.node_size
        graph_emb_size = gn.graph_emb_size

        # Model for computing graph representation
        self.f_m = Linear3(node_size, graph_emb_size)
        #Gated parameter when aggregating for graph representation
        self.g_m = nn.Sequential(
            Linear3(node_size, 1),
            nn.Sigmoid()
        )

    def forward(self, gn, mask):
        h_G = (self.f_m(gn.all_node_vectors[self.iters]) * self.g_m(gn.all_node_vectors[self.iters]))
        if mask is not None:
            h_G = h_G * mask

        h_Gs = torch.split(h_G, gn.graph_sizes)

        h_G = torch.stack(
            [h_G.sum(dim=0) for h_G in h_Gs]
        )
        return h_G

class Initializer(nn.Module):
    def __init__(self, gn, feature_size=64):
        super(Initializer, self).__init__()
        node_size = gn.node_size

        self.f_init = Linear3(feature_size, node_size)
        self.tanh = nn.Tanh()

    def forward(self, gn, e):
        #One hot for now
        h_v = self.f_init(e)
        return self.tanh(h_v)


class GraphNet(nn.Module):

    def __init__(self, node_size=64, edge_size=1, hidden_size=128, graph_emb_size=128, feature_size=64, T=10, masked_aggre=False, dual_passing=True):
        super(GraphNet, self).__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.hidden_size = hidden_size
        self.graph_emb_size = graph_emb_size
        self.feature_size = feature_size
        self.T = T
        self.prop = Propagator(self,T,dual_passing)

        self.aggre = nn.ModuleList([
            Aggregator(self, i)
        for i in range(T+1)])

        if masked_aggre:
            self.aggre2 = nn.ModuleList([
                Aggregator(self, i)
            for i in range(T+1)])

        self.initializer = Initializer(self,feature_size)
        self.clear()
        self.node_emb_mlp = Linear3(node_size*(T+1), node_size)
        self.graph_emb_mlp = Linear3(graph_emb_size * (T+1), graph_emb_size)

    def clear(self):
        self.nodes = []
        self.node_vectors = None
        self.all_node_vectors = []
        self.u_indices = []
        self.v_indices = []
        self.edge_vectors = None
        self.graph_sizes = []

    def init(self, features, edges, edge_label, graph_sizes):
        self.init_nodes(features)
        self.init_edges(edges, edge_label)
        self.graph_sizes = list(graph_sizes)

    def init_edges(self, edges, edge_label):
        if len(edges.shape) < 2:
            self.u_indices = None
            self.v_indices = None
        else:
            self.u_indices = edges[:,0]
            self.v_indices = edges[:,1]

        self.edge_vectors = edge_label.unsqueeze(1)

    def init_nodes(self, features):
        self.node_vectors = self.initializer(self, features)
        self.node_vectors_initial = self.node_vectors.clone()
        self.all_node_vectors.append(self.node_vectors)

    def propagate(self):
        self.prop(self)

    def get_embeddings(self):
        all_vectors = torch.cat(self.all_node_vectors, dim=1)
        return self.node_emb_mlp(all_vectors)

    def get_graph_embeddings(self,mask=None):
        if mask is not None:
            aggre_func = self.aggre2
        else:
            aggre_func = self.aggre
        graph_embs = [aggre_func[i](self, mask) for i in range(self.T+1)]
        all_vectors = torch.cat(graph_embs, dim=1)
        return self.graph_emb_mlp(all_vectors)