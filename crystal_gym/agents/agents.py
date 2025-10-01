"""
MEGNet and CHGNet agents
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional, Dict, Any
from collections.abc import Sequence

import dgl
import torch
from torch import nn, Tensor

from matgl.layers import MLP, ActivationFunction, EmbeddingBlock
from matgl.models._megnet import MEGNet
from matgl.utils.io import IOMixIn

from chgnet.model.functions import MLP as MLP_CH
from chgnet.model.model import CHGNet, BatchedGraph
from chgnet.graph import CrystalGraph

if TYPE_CHECKING:
    from chgnet import PredTask

# Define the number of actions as a constant
NUM_ACTIONS: int = 88


class EmbeddingBlockDev(EmbeddingBlock):
    """Embedding block for generating node, bond, and state features."""

    def __init__(
        self,
        degree_rbf: int,
        activation: nn.Module,
        dim_node_embedding: int,
        dim_edge_embedding: Optional[int] = None,
        dim_state_feats: Optional[int] = None,
        ntypes_node: Optional[int] = None,
        include_state: bool = False,
        ntypes_state: Optional[int] = None,
        dim_state_embedding: Optional[int] = None,
        device: str = 'cuda',
    ) -> None:
        """
        Initialize the EmbeddingBlockDev.

        Args:
            degree_rbf (int): Degree of the radial basis function.
            activation (nn.Module): Activation function to use.
            dim_node_embedding (int): Dimension of the node embedding.
            dim_edge_embedding (Optional[int]): Dimension of the edge embedding.
            dim_state_feats (Optional[int]): Dimension of the state features.
            ntypes_node (Optional[int]): Number of node types.
            include_state (bool): Whether to include state features.
            ntypes_state (Optional[int]): Number of state types.
            dim_state_embedding (Optional[int]): Dimension of the state embedding.
            device (str): Device to use ('cuda' or 'cpu').
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
            self.layer_state_embedding = nn.Embedding(ntypes_state, dim_state_embedding, device=device)
        if ntypes_node is not None:
            self.layer_node_embedding = nn.Embedding(ntypes_node, dim_node_embedding, device=device)
        if dim_edge_embedding is not None:
            dim_edges = [degree_rbf, dim_edge_embedding]
            self.layer_edge_embedding = MLP(dim_edges, activation=activation, activate_last=True, device=device)


class MEGNetRL(MEGNet, nn.Module, IOMixIn):
    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 1,
        dim_state_embedding: int = 8,
        ntypes_state: int = 21,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        activation_type: str = "softplus2",
        include_state: bool = True,
        no_condition: bool = False,
        device: str = 'cuda',
        num_actions: int = NUM_ACTIONS,
        critic: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(dim_edge_embedding=dim_edge_embedding)
        
        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None
            
        self.no_condition = no_condition
        if self.no_condition:
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
        
        self.edge_encoder = MLP(edge_dims, activation, activate_last=True).to(device=device)
        self.node_encoder = MLP(node_dims, activation, activate_last=True).to(device=device)
        self.state_encoder = MLP(state_dims, activation, activate_last=True).to(device=device)

        dim_blocks_out = hidden_layer_sizes_conv[-1]
        out_dim = 1 if critic else num_actions

        self.output_proj = MLP(
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, out_dim],
            activation=activation,
            activate_last=False,
        )   

        self.blocks = self.blocks.to(device=device)
        self.output_proj = self.output_proj.to(device=device)
        self.device = device

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ) -> Tensor:
        """Forward pass of MEGnet.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: State features

        Returns:
            Model prediction
        """
        if self.no_condition:
            state_feat = state_feat[:, :-1]
            
        try:
            edge_feat = self.bond_expansion(edge_feat).to(device=self.device)
        except Exception:
            edge_feat = self.bond_expansion(edge_feat.cpu()).to(device=self.device)
        node_feat = node_feat.to(dtype=torch.int64)
        focus_feat = graph.focus
        node_feat, edge_feat, focus_feat = self.embedding(node_feat, edge_feat, focus_feat)
        edge_feat = self.edge_encoder(edge_feat.to(dtype=torch.float32))
        node_feat = self.node_encoder(node_feat)
        state_feat = torch.cat((state_feat, focus_feat), dim=1)
        state_feat = self.state_encoder(state_feat.to(dtype=torch.float32))

        for block in self.blocks:
            output = block(graph, edge_feat, node_feat, state_feat)
            edge_feat, node_feat, state_feat = output

        node_vec = self.node_s2s.to(device=self.device)(graph, node_feat)
        edge_vec = self.edge_s2s.to(device=self.device)(graph, edge_feat)

        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        state_feat = torch.squeeze(state_feat)

        vec = torch.hstack([node_vec, edge_vec, state_feat])

        if self.dropout:
            vec = self.dropout(vec)  

        output = self.output_proj(vec)
        return output
    

class CHGNetRL(nn.Module):
    def __init__(
        self,
        atom_fea_dim: int = 64,
        mlp_hidden_dims: Sequence[int] | int = (64, 64, 64),
        mlp_dropout: float = 0,
        non_linearity: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        critic: bool = False,
        num_actions: int = NUM_ACTIONS,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        out_dim = 1 if critic else num_actions

        self.chgnet = CHGNet().load()
        self.mlp_bg = MLP_CH(
            input_dim=atom_fea_dim,
            hidden_dim=mlp_hidden_dims,
            output_dim=out_dim,
            dropout=mlp_dropout,
            activation=non_linearity,
        ).cuda()

    def forward(
        self,
        graphs: Sequence[CrystalGraph],
        task: PredTask = "bg",
    ) -> Dict[str, Tensor]:
        """Forward pass through CHGNet.

        Args:
            graphs: Sequence of crystal graphs
            task: Prediction task type

        Returns:
            Dictionary containing predictions
        """
        batched_graph = BatchedGraph.from_graphs(
            graphs,
            bond_basis_expansion=self.chgnet.bond_basis_expansion,
            angle_basis_expansion=self.chgnet.angle_basis_expansion,
            compute_stress="s" in task,
        )
        return self._compute(batched_graph)

    def _compute(self, g: BatchedGraph) -> Dict[str, Tensor]:
        """Compute predictions from batched graph.

        Args:
            g: Batched graph

        Returns:
            Dictionary containing predictions
        """
        prediction: Dict[str, Tensor] = {}
        atoms_per_graph = torch.bincount(g.atom_owners)
        prediction["atoms_per_graph"] = atoms_per_graph
        
        atom_feas = self.chgnet.atom_embedding(g.atomic_numbers - 1)
        bond_feas = self.chgnet.bond_embedding(g.bond_bases_ag)
        bond_weights_ag = self.chgnet.bond_weights_ag(g.bond_bases_ag)
        bond_weights_bg = self.chgnet.bond_weights_bg(g.bond_bases_bg)
        
        if len(g.angle_bases) != 0:
            angle_feas = self.chgnet.angle_embedding(g.angle_bases)

        # Message Passing
        for idx, (atom_layer, bond_layer, angle_layer) in enumerate(
            zip(self.chgnet.atom_conv_layers[:-1], self.chgnet.bond_conv_layers, self.chgnet.angle_layers)
        ):
            # Atom Conv
            atom_feas = atom_layer(
                atom_feas=atom_feas,
                bond_feas=bond_feas,
                bond_weights=bond_weights_ag,
                atom_graph=g.batched_atom_graph,
                directed2undirected=g.directed2undirected,
            )

            # Bond Conv
            if len(g.angle_bases) != 0 and bond_layer is not None:
                bond_feas = bond_layer(
                    atom_feas=atom_feas,
                    bond_feas=bond_feas,
                    bond_weights=bond_weights_bg,
                    angle_feas=angle_feas,
                    bond_graph=g.batched_bond_graph,
                )

                # Angle Update
                if angle_layer is not None:
                    angle_feas = angle_layer(
                        atom_feas=atom_feas,
                        bond_feas=bond_feas,
                        angle_feas=angle_feas,
                        bond_graph=g.batched_bond_graph,
                    )
            
            # Last conv layer
            atom_feas = self.chgnet.atom_conv_layers[-1](
                atom_feas=atom_feas,
                bond_feas=bond_feas,
                bond_weights=bond_weights_ag,
                atom_graph=g.batched_atom_graph,
                directed2undirected=g.directed2undirected,
            )
            if self.chgnet.readout_norm is not None:
                atom_feas = self.chgnet.readout_norm(atom_feas)

        # Aggregate nodes and ReadOut
        if self.chgnet.mlp_first:
            crystal_feas = self.chgnet.pooling(atom_feas, g.atom_owners)
            output = self.mlp_bg(crystal_feas)
            return output
        
        return prediction
