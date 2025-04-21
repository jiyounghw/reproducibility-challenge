import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, activation, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.activation = activation

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, graph):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        src, dst = graph.edges()
        adj = torch.zeros((graph.number_of_nodes(), graph.number_of_nodes())).to(h.device)
        adj[src, dst] = 1

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return self.activation(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.activation(e) # activation function varient

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, activation):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.activation = activation

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=self.dropout, alpha=alpha, concat=True, activation=activation) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, self.dropout, alpha, concat=False, activation=activation)

    def forward(self, x, graph):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([att(x, graph) for att in self.attentions], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.out_att(x, graph))
        return F.log_softmax(x, dim=1)