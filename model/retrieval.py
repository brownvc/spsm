from model.gnn import *
import math
import random

def gaussian_mixture_loss(Y, log_logits, mus, sigmas):
    Y_normalized = (Y.unsqueeze(1) - mus) / sigmas

    log_probs = -0.5 * (Y_normalized ** 2) - (0.5 * math.log(2*math.pi)) - sigmas.log()

    log_probs = log_probs.sum(dim=2)
    loss = -torch.logsumexp((log_probs + log_logits), dim=1)
    return loss

def embedding_loss(pos, neg, log_logits, mus, sigmas, margin=20):
    loss_pos = gaussian_mixture_loss(pos, log_logits, mus, sigmas)
    loss_neg = gaussian_mixture_loss(neg, log_logits, mus, sigmas)
    return F.relu(loss_pos - loss_neg + margin).mean()

class Model(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, graph_emb_size, feature_size, T, embedding_dim, num_mixtures, margin):
        super(Model, self).__init__()

        self.node_size = node_size
        self.edge_size = edge_size
        self.hidden_size = hidden_size
        self.graph_emb_size = graph_emb_size
        self.feature_size = feature_size
        self.T = T
        self.embedding_dim = embedding_dim
        self.num_mixtures = num_mixtures
        self.margin = margin

        self.gnn_retrieval = GraphNet(
            node_size=node_size,
            edge_size=edge_size,
            hidden_size=hidden_size,
            graph_emb_size=graph_emb_size,
            feature_size=feature_size,
            T=T,
            masked_aggre=True,
        )

        self.gnn_embedding = GraphNet(
            node_size=node_size,
            edge_size=edge_size,
            hidden_size=hidden_size,
            graph_emb_size=graph_emb_size,
            feature_size=feature_size - 1,
            T=T // 2,
            masked_aggre=True,
        )

        self.mdn_fc = nn.Sequential(
            nn.Linear(graph_emb_size * 2, 256), 
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, (embedding_dim * 2 + 1) * num_mixtures)
        )

        self.embedding_fc = nn.Sequential(
            nn.Linear(graph_emb_size, 256), 
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, embedding_dim)
        )

    def get_latent(self, features, edges, edge_label, graph_sizes, input_label):
        self.gnn_retrieval.clear()
        self.gnn_retrieval.init(features, edges, edge_label, graph_sizes)
        self.gnn_retrieval.propagate()
        g1 = self.gnn_retrieval.get_graph_embeddings()
        g11 = self.gnn_retrieval.get_graph_embeddings(input_label)

        h = torch.cat([g1, g11], dim=1)
        return h

    def predict_retrieval(self, features, edges, edge_label, graph_sizes, input_label):
        self.gnn_retrieval.clear()
        self.gnn_retrieval.init(features, edges, edge_label, graph_sizes)
        self.gnn_retrieval.propagate()
        g1 = self.gnn_retrieval.get_graph_embeddings()
        g11 = self.gnn_retrieval.get_graph_embeddings(input_label)

        h = self.mdn_fc(torch.cat([g1, g11], dim=1))
        
        embedding_dim = self.embedding_dim
        num_mixtures = self.num_mixtures
        pred_size = embedding_dim * num_mixtures

        log_logits = h[:, -num_mixtures:]
        log_logits = F.log_softmax(log_logits)

        mus = h[:, 0:pred_size]
        mus = mus.view(-1, num_mixtures, embedding_dim)
        sigmas = h[:, pred_size:pred_size*2]
        sigmas = torch.exp(sigmas)
        sigmas = torch.clamp(sigmas, 1e-6, 50)
        sigmas = sigmas.view(-1, num_mixtures, embedding_dim)

        return log_logits, mus, sigmas

    def predict_embedding(self, features, edges, edge_label, graph_sizes):
        self.gnn_embedding.clear()
        self.gnn_embedding.init(features, edges, edge_label, graph_sizes)
        self.gnn_embedding.propagate()
        g1 = self.gnn_embedding.get_graph_embeddings()
        emb = self.embedding_fc(g1)
        return emb

    def get_stats_dict(self):
        return {'loss': 0, 'hard_percentage': 0, 'semi_hard_percentage': 0}
    
    def get_model_selection_crition(self):
        return 'semi_hard_percentage'

    def forward(self, data, device, neg_iterator):
        boxes, edges, input_label, edge_label, N, boxes_p, edges_p, N_p = data

        boxes, input_label, edge_label = boxes.to(device), input_label.to(device), edge_label.to(device)
        boxes_p = boxes_p.to(device)

        input_label = input_label.unsqueeze(1)
        input_features = torch.cat([input_label, boxes], dim=1)
        log_logits, mus, sigmas = self.predict_retrieval(input_features, edges, edge_label, N, input_label.detach())
        
        edge_label_p = torch.zeros(len(edges_p)).to(device)
        positive_embeddings = self.predict_embedding(boxes_p, edges_p, edge_label_p, N_p)

        boxes_n, edges_n, _, splits = next(neg_iterator)
        boxes_n = boxes_n.to(device)

        edge_label_n = torch.zeros(len(edges_n)).to(device)
        negative_embeddings = self.predict_embedding(boxes_n, edges_n, edge_label_n, splits)

        negative_indices = []

        stats = self.get_stats_dict()
        for i in range(N.shape[0]):
            with torch.no_grad():
                n_neg = negative_embeddings.shape[0]

                pos_likelihoods = gaussian_mixture_loss(positive_embeddings[i:i+1], log_logits[i], mus[i], sigmas[i])
                likelihoods = gaussian_mixture_loss(negative_embeddings, log_logits[i], mus[i], sigmas[i])
                indices_hard = torch.where((pos_likelihoods - likelihoods > 0))[0]
                stats['hard_percentage'] += indices_hard.shape[0] / n_neg
                indices = torch.where((pos_likelihoods - likelihoods > -self.margin))[0]
                stats['semi_hard_percentage'] += indices.shape[0] / n_neg
                indices = list(indices.cpu().numpy())
                if len(indices) == 0:
                    min_idx = random.randint(0, negative_embeddings.size()[0]-1)
                else: 
                    min_idx = random.choice(indices)
            negative_indices.append(int(min_idx))

        negative_embeddings = negative_embeddings[negative_indices]

        loss = embedding_loss(positive_embeddings, negative_embeddings, log_logits, mus, sigmas, margin=self.margin)

        stats['loss'] = loss.detach().item() * N.shape[0]

        return loss, stats