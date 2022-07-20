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

        self.gnn = GraphNet(
            node_size=node_size,
            edge_size=edge_size,
            hidden_size=hidden_size,
            graph_emb_size=graph_emb_size,
            feature_size=feature_size,
            T=T,
            masked_aggre=True,
        )

        self.fc_continue = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

        self.fc_logit = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )
    
    def get_latent(self, features, edges, edge_label, graph_sizes, input_label, predict_idxs):
        self.gnn.clear()
        self.gnn.init(features, edges, edge_label, graph_sizes)
        self.gnn.propagate()
        g1 = self.gnn.get_graph_embeddings()
        return g1

    def predict_likelihood(self, features, edges, edge_label, graph_sizes, input_label, predict_idxs):
        self.gnn.clear()
        self.gnn.init(features, edges, edge_label, graph_sizes)
        self.gnn.propagate()
        g1 = self.gnn.get_graph_embeddings()
        g11 = self.gnn.get_graph_embeddings(input_label)
        p = self.fc_continue(torch.cat([g1, g11], dim=1))

        n2 = self.gnn.get_embeddings()
        logits = self.fc_logit(n2)[predict_idxs]
        return p, logits

    def get_stats_dict(self):
        return {'loss': 0, 'loss_continue': 0, 'loss_first': 0, 'loss_next': 0}

    def get_model_selection_crition(self):
        return 'loss'

    def forward(self, data, device):
        loss_fn = F.cross_entropy

        boxes, edges, input_label, should_continue, next_label, edge_label, N, predict_idxs, N2 = data

        boxes, input_label, should_continue, next_label, edge_label = boxes.to(device), input_label.to(device), should_continue.to(device), next_label.to(device), edge_label.to(device)

        input_label = input_label.unsqueeze(1)
        input_features = torch.cat([input_label, boxes], dim=1)

        p, logits = self.predict_likelihood(input_features, edges, edge_label, N, input_label.detach(), predict_idxs)

        loss_continue = loss_fn(p, should_continue)
        loss_first = torch.zeros(1).to(device)
        loss_next = torch.zeros(1).to(device)

        n_first = 0
        n_next = 0
        logits = torch.split(logits, list(N2))
        input_labels = torch.split(input_label, list(N))
        for i, logit in enumerate(logits):
            if next_label[i] >=0:
                if input_labels[i].sum() == 0:
                    loss_first += loss_fn(logit.view(1,-1), next_label[i:i+1])
                    n_first += 1
                else:
                    loss_next += loss_fn(logit.view(1,-1), next_label[i:i+1])
                    n_next += 1
        if n_first > 0:
            loss_first /= n_first
        if n_next > 0:
            loss_next /= n_next

        loss = loss_continue + loss_next + loss_first * 0.1

        stats = self.get_stats_dict()
        stats['loss'] = loss.detach().item() * N.shape[0]
        stats['loss_continue'] = loss_continue.detach().item() * N.shape[0]
        stats['loss_first'] = loss_first.detach().item() * N.shape[0]
        stats['loss_next'] = loss_next.detach().item() * N.shape[0]

        return loss, stats