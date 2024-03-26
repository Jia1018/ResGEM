import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_undirected
from torch_scatter import scatter
from torch_geometric.nn import graclus
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos
from torch_sparse import coalesce


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")  # "Max" aggregation.
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        return self.propagate(edge_index, x=x)  # shape [num_nodes, out_channels]

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_j: Source node features of shape [num_edges, in_channels]
        # x_i: Target node features of shape [num_edges, in_channels]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)  # shape [num_edges, out_channels]
    

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def softmax(src, index, num_nodes):
    """
    Given a value tensor: `src`, this function first groups the values along the first dimension
    based on the indices specified in: `index`, and then proceeds to compute the softmax individually for each group.
    """
    N = int(index.max()) + 1 if num_nodes is None else num_nodes
    out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    out = out.exp()
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    return out / (out_sum + 1e-16)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k[0], dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k=k)   # (batch_size, num_points, k)
    #device = torch.device('cuda')
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.long()

    idx = idx + idx_base

    idx = idx.contiguous().view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k[0], num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k[0], 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2) 
  
    return feature

def get_knn_feature(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k=k)   # (batch_size, num_points, k)
    #device = torch.device('cuda')
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.long()

    idx = idx + idx_base

    idx = idx.contiguous().view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k[0], num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims)
    
    feature = torch.cat((x, feature), dim=2).permute(0, 3, 1, 2) 
  
    return feature

def get_graph_feature_db(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    positions = x[:, 3:6, :]
    idx = knn(positions, k=k)   # (batch_size, num_points, k)
    #device = torch.device('cuda')
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.long()

    idx = idx + idx_base

    idx = idx.contiguous().view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k[0], num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k[0], 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


class DGCNN(nn.Module):
    def __init__(self, k, init_dims, emb_dims, dropout, output_channels=3):
        super(DGCNN, self).__init__()
        self.k = torch.IntTensor([k])
        self.k_g = torch.IntTensor([3])
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(init_dims*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1024, emb_dims, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn8 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn9 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, 64)
        self.bn10 = nn.BatchNorm1d(64)
        self.linear4 = nn.Linear(64, output_channels)

    def forward(self, inputs):
        x = inputs[:, 0:17, :]
        idx = inputs[:, 17:20, :]
        idx = idx.long()
        idx = idx.permute(0, 2, 1)

        batch_size = x.size(0)
        x = get_graph_feature_idx(x, self.k_g, idx)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature_idx(x1, self.k_g, idx)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature_idx(x2, self.k_g, idx)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x4, k=self.k)
        x = self.conv5(x)
        x5 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x5, k=self.k)
        x = self.conv6(x)
        x6 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)

        x = self.conv7(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn8(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn9(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = F.leaky_relu(self.bn10(self.linear3(x)), negative_slope=0.2)
        x = self.linear4(x)
        return x

class FeaStConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=8,
                 bias=True,
                 t_inv=True,
                 **kwargs):
        super(FeaStConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.t_inv = t_inv

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.u = Parameter(torch.Tensor(in_channels, heads))
        self.c = Parameter(torch.Tensor(heads))
        if not self.t_inv:
            self.v = Parameter(torch.Tensor(in_channels, heads))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        normal(self.weight, mean=0, std=0.1)
        normal(self.u, mean=0, std=0.1)
        normal(self.c, mean=0, std=0.1)
        normal(self.bias, mean=0, std=0.1)
        if not self.t_inv:
            normal(self.v, mean=0, std=0.1)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j):
        # dim: x_i, [E, F_in];
        if self.t_inv:
            # with translation invariance
            q = torch.mm((x_i - x_j), self.u) + self.c  #[E, heads]
        else:
            q = torch.mm(x_i, self.u) + torch.mm(x_j, self.v) + self.c
        q = F.softmax(q, dim=1)  #[E, heads]

        x_j = torch.mm(x_j, self.weight).view(-1, self.heads,
                                              self.out_channels)
        return (x_j * q.view(-1, self.heads, 1)).sum(dim=1)

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



class FeatureEmbeddingNet(torch.nn.Module):
    def __init__(self, in_channels, k):
        super(FeatureEmbeddingNet, self).__init__()
        self.k = torch.IntTensor([k])
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ELU())
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ELU())
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.ELU())
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.ELU())
        self.fc1 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.unsqueeze(0).permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k)
        #x = get_graph_feature_db(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1).permute(0, 2, 1)

        x = self.fc1(x).squeeze(0)
        x = F.elu(self.bn5(x))
        
        return x

    
class ResBlock(torch.nn.Module):
    def __init__(self, channels, in_channels, heads, t_inv=True):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.fc1 = nn.Linear(channels, in_channels)
        self.fc2 = nn.Linear(in_channels, channels)
        self.conv = FeaStConv(in_channels, in_channels, heads, t_inv)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        
    def forward(self, x, edge_index):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.conv(x, edge_index)))
        x = self.fc2(x)
        return x


class ResGEM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, neighbor_sizes, heads, p=0.5, emb_dims=1024, t_inv=True):
        super(ResGEM, self).__init__()
        self.in_channels = in_channels
        self.nsizes = len(neighbor_sizes)
        self.FeatureEmbModules = nn.ModuleList([FeatureEmbeddingNet(in_channels, k) for k in neighbor_sizes])
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        #self.bn0 = nn.BatchNorm1d(emb_dims)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(emb_dims)
        self.bn8 = nn.BatchNorm1d(emb_dims)
        #self.bn10 = nn.BatchNorm1d(emb_dims)
        #self.bn11 = nn.BatchNorm1d(emb_dims)
        self.bn9 = nn.BatchNorm1d(64)
        self.dp1 = nn.Dropout(p)

        self.conv1 = EdgeConv(in_channels, 64)
        #self.conv2 = EdgeConv(64, 64)
        self.conv3 = EdgeConv(64, 128)
        self.conv4 = EdgeConv(128, 256)
        #self.fc0 = nn.Linear(emb_dims + in_channels, emb_dims)
        #self.fc0 = nn.Linear(512, 256)
        
        self.resblock1 = ResBlock(emb_dims, 64, heads)
        self.resblock2 = ResBlock(emb_dims, 64, heads)
        self.resblock3 = ResBlock(emb_dims, 64, heads)
        self.resblock4 = ResBlock(emb_dims, 64, heads)

        self.fc1 = nn.Linear(emb_dims, 64)
        self.fc2 = nn.Linear(64, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.resblock1.reset_parameters()
        self.resblock2.reset_parameters()
        self.resblock3.reset_parameters()
        self.resblock4.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x[:, :self.in_channels]
        neigh_x = torch.cat([FEM(x) for i, FEM in enumerate(self.FeatureEmbModules)], dim=1)

        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        x = F.elu(self.bn4(self.conv4(x, edge_index)))

        x = torch.cat((x, neigh_x), dim=-1)

        x = F.elu(self.bn5(self.resblock1(x, edge_index) + x))
        x = F.elu(self.bn6(self.resblock2(x, edge_index) + x)) 
        x = F.elu(self.bn7(self.resblock3(x, edge_index) + x))
        x = F.elu(self.bn8(self.resblock4(x, edge_index) + x))
        
        x = F.elu(self.bn9(self.fc1(x)))
        x = self.dp1(x)
        x = self.fc2(x)
        return x
    