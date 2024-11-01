from __future__ import annotations
from matgl.layers import MLP, ActivationFunction, EmbeddingBlock
from torch import nn
from matgl.models._megnet import MEGNet
from matgl.utils.io import IOMixIn
import dgl
import torch
from torch import nn
from matgl.layers import EmbeddingBlock, MLP

NUM_ACTIONS = 88

class EmbeddingBlockDev(EmbeddingBlock):
    """Embedding block for generating node, bond and state features."""

    def __init__(
        self,
        degree_rbf: int,
        activation: nn.Module,
        dim_node_embedding: int,
        dim_edge_embedding: int | None = None,
        dim_state_feats: int | None = None,
        ntypes_node: int | None = None,
        include_state: bool = False,
        ntypes_state: int | None = None,
        dim_state_embedding: int | None = None,
        device: str = 'cuda',
    ):
        """
        Args:
            degree_rbf (int): number of rbf
            activation (nn.Module): activation type
            dim_node_embedding (int): dimensionality of node features
            dim_edge_embedding (int): dimensionality of edge features
            dim_state_feats: dimensionality of state features
            ntypes_node: number of node labels
            include_state: Whether to include state embedding
            ntypes_state: number of state labels
            dim_state_embedding: dimensionality of state embedding.
        """
        super().__init__(degree_rbf, activation, dim_node_embedding)
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.dim_edge_embedding = dim_edge_embedding
        self.dim_state_feats = dim_state_feats
        self.ntypes_node = ntypes_node
        self.dim_state_embedding = dim_state_embedding
        self.activation = activation
        if ntypes_state is not None and dim_state_embedding is not None:
            self.layer_state_embedding = nn.Embedding(ntypes_state, dim_state_embedding, device = device)  # type: ignore
        if ntypes_node is not None:
            self.layer_node_embedding = nn.Embedding(ntypes_node, dim_node_embedding, device = device)
        if dim_edge_embedding is not None:
            dim_edges = [degree_rbf, dim_edge_embedding]
            self.layer_edge_embedding = MLP(dim_edges, activation=activation, activate_last=True, device = device)


class MEGNetRL(MEGNet, nn.Module, IOMixIn):
    def __init__(self, 
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 1,
        dim_state_embedding: int = 8,
        ntypes_state: int = 21,
        hidden_layer_sizes_input: tuple = (64, 32),
        hidden_layer_sizes_conv: tuple = (64, 64, 32),
        hidden_layer_sizes_output: tuple = (32, 16),
        activation_type: str = "softplus2",
        include_state: bool = True,
        no_condition = False,
        device: str = 'cuda',
        num_actions = NUM_ACTIONS, 
        critic = False,
        **kwargs,
    
    ):
        super(MEGNetRL, self).__init__(dim_edge_embedding = dim_edge_embedding)
        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None
        self.no_condition = no_condition
        if self.no_condition:
            print('Warning, no condition')
            dim_state_embedding -= 1
        self.embedding = EmbeddingBlockDev(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=num_actions + 1,
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )
        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding * 2 - 1, *hidden_layer_sizes_input]
        self.edge_encoder = MLP(edge_dims, activation, activate_last=True).to(device = device)
        self.node_encoder = MLP(node_dims, activation, activate_last=True).to(device = device)
        self.state_encoder = MLP(state_dims, activation, activate_last=True).to(device = device)

        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]

        if critic:
            out_dim = 1
        else:
            out_dim = num_actions

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, out_dim],
            activation=activation,
            activate_last=False,
        )   

        self.blocks = self.blocks.to(device = device)
        self.output_proj = self.output_proj.to(device = device)
        self.device = device

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ):
        """Forward pass of MEGnet. Executes all blocks.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: State features.

        Returns:
            Prediction
        """
        if self.no_condition:
            state_feat = state_feat[:,:-1]
        try:
            edge_feat = self.bond_expansion(edge_feat).to(device = self.device)
        except:
            edge_feat = self.bond_expansion(edge_feat.cpu()).to(device = self.device)
        node_feat = node_feat.to(dtype = torch.int64)
        focus_feat = graph.focus 
        node_feat, edge_feat, focus_feat = self.embedding(node_feat, edge_feat, focus_feat)
        edge_feat = self.edge_encoder(edge_feat.to(dtype = torch.float32))
        node_feat = self.node_encoder(node_feat)
        state_feat = torch.cat((state_feat, focus_feat), dim = 1)
        state_feat = self.state_encoder(state_feat.to(dtype = torch.float32))

        for block in self.blocks:
            output = block(graph, edge_feat, node_feat, state_feat)
            edge_feat, node_feat, state_feat = output

        node_vec = self.node_s2s.to(device = self.device)(graph, node_feat)
        edge_vec = self.edge_s2s.to(device = self.device)(graph, edge_feat)

        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        state_feat = torch.squeeze(state_feat)

        vec = torch.hstack([node_vec, edge_vec, state_feat])

        if self.dropout:
            vec = self.dropout(vec)  

        output = self.output_proj(vec)
        return output