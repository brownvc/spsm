
from model.gnn import *

class Model(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, graph_emb_size, feature_size, T, embedding_dim):
        super(Model, self).__init__()

        self.node_size = node_size
        self.edge_size = edge_size
        self.hidden_size = hidden_size
        self.graph_emb_size = graph_emb_size
        self.feature_size = feature_size
        self.T = T
        self.embedding_dim = embedding_dim

        self.gnn_anchor = GraphNet(
            node_size=node_size,
            edge_size=edge_size,
            hidden_size=hidden_size,
            graph_emb_size=graph_emb_size,
            feature_size=feature_size,
            T=T,
            masked_aggre=True,
        )

        self.gnn_target = GraphNet(
            node_size=node_size,
            edge_size=edge_size,
            hidden_size=hidden_size,
            graph_emb_size=graph_emb_size,
            feature_size=feature_size-1,
            T=T//2,
            masked_aggre=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def predict_likelihood(self, features, edges, edge_label, graph_sizes, u_indices, v_indices, features_p, edges_p, edge_label_p, graph_sizes_p):
        self.gnn_anchor.clear()
        self.gnn_anchor.init(features, edges, edge_label, graph_sizes)
        self.gnn_anchor.propagate()
        n2 = self.gnn_anchor.get_embeddings()

        self.gnn_target.clear()
        self.gnn_target.init(features_p, edges_p, edge_label_p, graph_sizes_p)
        self.gnn_target.propagate()
        m2 = self.gnn_target.get_embeddings()

        u2 = n2[u_indices]
        v2 = m2[v_indices]

        uvs = torch.cat([u2, v2], dim=1)
        return self.fc(uvs)

    def get_stats_dict(self):
        return {'loss': 0}

    def get_model_selection_crition(self):
        return 'loss'

    def forward(self, data, device):
        boxes, edges, target_label, edge_label, N, u_indices, v_indices, boxes_p, edges_p, N_p, input_label = data

        boxes, target_label, edge_label, u_indices, v_indices = boxes.to(device), target_label.to(device), edge_label.to(device), u_indices.to(device), v_indices.to(device)
        input_label = input_label.to(device)
        boxes_p = boxes_p.to(device)

        edge_label_p = torch.zeros(len(edges_p)).to(device)

        input_label = input_label.unsqueeze(1)
        input_features = torch.cat([input_label, boxes], dim=1)

        output = self.predict_likelihood(input_features, edges, edge_label, N, u_indices, v_indices, boxes_p, edges_p, edge_label_p, N_p)

        loss = F.cross_entropy(output, target_label)

        stats = self.get_stats_dict()
        stats['loss'] = loss.detach().item() * N.shape[0]

        return loss, stats
