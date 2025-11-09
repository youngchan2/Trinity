#!/usr/bin/env python3
"""
Gradient Rule Table
Provides rule-based mappings from forward IR operations to backward IR fragments.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Set
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AstNode import ASTNode
from NodeType import NodeType


def clone_ast(node: Optional[ASTNode]) -> Optional[ASTNode]:
    """Deep clone an AST node."""
    if node is None:
        return None
    return ASTNode(node.node_type,
                   [clone_ast(child) for child in node.children],
                   node.value)


def _split_names(names: str) -> List[str]:
    return [name.strip() for name in names.split(',') if name.strip()]


def var(name: str) -> ASTNode:
    return ASTNode(NodeType.VAR, [], name)


def num(value: int) -> ASTNode:
    return ASTNode(NodeType.NUM, [], value)


def tensor_node(names: str, kind: NodeType = NodeType.TENSOR) -> ASTNode:
    return ASTNode(kind, [var(name) for name in _split_names(names)])


def load_node(kind: NodeType, names: str, index: Optional[ASTNode]) -> ASTNode:
    children = [tensor_node(names, kind)]
    if index is not None:
        children.append(clone_ast(index))
    return ASTNode(NodeType.LOAD, children)


def load_tensor(names: str, index: Optional[ASTNode]) -> ASTNode:
    return load_node(NodeType.TENSOR, names, index)


def load_input(names: str, index: Optional[ASTNode]) -> ASTNode:
    return load_node(NodeType.INPUT, names, index)


def tile(name: str) -> ASTNode:
    return ASTNode(NodeType.TILE, [var(name)])


def elem(name: str) -> ASTNode:
    return ASTNode(NodeType.ELEM, [var(name)])


def fulltile() -> ASTNode:
    return ASTNode(NodeType.FULLTILE, [])


def index_node(*components: ASTNode) -> ASTNode:
    return ASTNode(NodeType.INDEX, [clone_ast(comp) for comp in components])


def index_elem_full_full() -> ASTNode:
    return index_node(elem('n'), fulltile(), fulltile())


def index_elem_full() -> ASTNode:
    return index_node(elem('n'), fulltile())


def index_fulltile_tilen() -> ASTNode:
    return index_node(fulltile(), tile('n'))


def index_fulltile_tilek() -> ASTNode:
    return index_node(fulltile(), tile('k'))


def index_tilek_tilen() -> ASTNode:
    return index_node(tile('k'), tile('n'))


def add(lhs: ASTNode, rhs: ASTNode) -> ASTNode:
    return ASTNode(NodeType.ADD, [lhs, rhs])


def sub(lhs: ASTNode, rhs: ASTNode) -> ASTNode:
    return ASTNode(NodeType.SUB, [lhs, rhs])


def mul(lhs: ASTNode, rhs: ASTNode) -> ASTNode:
    return ASTNode(NodeType.MUL, [lhs, rhs])


def div(lhs: ASTNode, rhs: ASTNode) -> ASTNode:
    return ASTNode(NodeType.DIV, [lhs, rhs])


def matmul(lhs: ASTNode, rhs: ASTNode) -> ASTNode:
    return ASTNode(NodeType.MATMUL, [lhs, rhs])


def transpose(node: ASTNode) -> ASTNode:
    return ASTNode(NodeType.TRANSPOSE, [node])

def permute3(node: ASTNode, *dims: int) -> ASTNode:
    return ASTNode(NodeType.PERMUTE3, [node] + [num(dim) for dim in dims])


def squeeze(node: ASTNode, dim: int) -> ASTNode:
    return ASTNode(NodeType.SQUEEZE, [node, num(dim)])


def unsqueeze(node: ASTNode, dim: int) -> ASTNode:
    return ASTNode(NodeType.UNSQUEEZE, [node, num(dim)])


def bcast(node: ASTNode, dim: int) -> ASTNode:
    return ASTNode(NodeType.BCAST, [node, num(dim)])


def rsum(node: ASTNode, dim: int) -> ASTNode:
    return ASTNode(NodeType.RSUM, [node, num(dim)])


def exp_node(node: ASTNode) -> ASTNode:
    return ASTNode(NodeType.EXP, [node])


def store_tensor(names: str, value: ASTNode, index: Optional[ASTNode]) -> ASTNode:
    children = [tensor_node(names), value]
    if index is not None:
        children.append(clone_ast(index))
    return ASTNode(NodeType.STORE, children)


def _extract_tensor_name(tensor_node: ASTNode) -> str:
    """Extract tensor name from tensor/input/output node"""
    names = []
    for child in tensor_node.children:
        if child.node_type == NodeType.VAR:
            names.append(child.value)
    return ','.join(names) if names else ''


def compute_gradient_recursive(output_grad: ASTNode, expr: ASTNode, wrt_tensors: Set[str]) -> Dict[str, List[ASTNode]]:
    """
    Recursively compute gradients using chain rule.

    Args:
        output_grad: Gradient of the output (e.g., dO2)
        expr: Forward expression AST
        wrt_tensors: Set of tensor names to compute gradients for

    Returns:
        Dictionary mapping tensor names to list of gradient expressions
    """
    grad_dict = defaultdict(list)

    if expr.node_type == NodeType.SQUEEZE:
        # Y = squeeze(X, dim) → dX = unsqueeze(dY, dim)
        if len(expr.children) < 2:
            return grad_dict
        inner_expr = expr.children[0]
        dim = expr.children[1].value if expr.children[1].node_type == NodeType.NUM else 1
        dX = unsqueeze(output_grad, dim)
        inner_grads = compute_gradient_recursive(dX, inner_expr, wrt_tensors)
        for k, v in inner_grads.items():
            grad_dict[k].extend(v)

    elif expr.node_type == NodeType.PERMUTE3:
        # Y = permute3(X, d0, d1, d2) → dX = permute3(dY, inverse_perm)
        if len(expr.children) < 4:
            return grad_dict
        inner_expr = expr.children[0]
        dims = [expr.children[i].value for i in range(1, 4)]
        # Compute inverse permutation
        inv_dims = [dims.index(i) for i in range(3)]
        dX = permute3(output_grad, *inv_dims)
        inner_grads = compute_gradient_recursive(dX, inner_expr, wrt_tensors)
        for k, v in inner_grads.items():
            grad_dict[k].extend(v)

    elif expr.node_type == NodeType.MATMUL:
        # Y = A * B (matmul) → dA = dY * B^T, dB = A^T * dY
        if len(expr.children) < 2:
            return grad_dict
        A = expr.children[0]
        B = expr.children[1]

        # Check if B has permute3 (e.g., K^T)
        if B.node_type == NodeType.PERMUTE3:
            # Y = A * permute3(B_inner, ...)
            # dA = dY * B, dB_inner = permute3(A^T * dY, inverse)
            dA = matmul(output_grad, clone_ast(B))
            inner_grads_A = compute_gradient_recursive(dA, A, wrt_tensors)
            for k, v in inner_grads_A.items():
                grad_dict[k].extend(v)

            # For B: need to apply transpose gradient
            B_inner = B.children[0] if B.children else B
            dims = [B.children[i].value for i in range(1, 4)] if len(B.children) >= 4 else [0, 2, 1]
            inv_dims = [dims.index(i) for i in range(3)]
            dB_transposed = matmul(permute3(clone_ast(A), 0, 2, 1), output_grad)
            dB_inner = permute3(dB_transposed, *inv_dims)
            inner_grads_B = compute_gradient_recursive(dB_inner, B_inner, wrt_tensors)
            for k, v in inner_grads_B.items():
                grad_dict[k].extend(v)
        else:
            # Standard matmul: Y = A * B
            # dA = dY * B^T
            if B.node_type == NodeType.LOAD:
                dA = matmul(output_grad, permute3(clone_ast(B), 0, 2, 1))
            else:
                dA = matmul(output_grad, clone_ast(B))
            inner_grads_A = compute_gradient_recursive(dA, A, wrt_tensors)
            for k, v in inner_grads_A.items():
                grad_dict[k].extend(v)

            # dB = A^T * dY
            if A.node_type == NodeType.LOAD:
                dB = matmul(permute3(clone_ast(A), 0, 2, 1), output_grad)
            else:
                dB = matmul(clone_ast(A), output_grad)
            inner_grads_B = compute_gradient_recursive(dB, B, wrt_tensors)
            for k, v in inner_grads_B.items():
                grad_dict[k].extend(v)

    elif expr.node_type == NodeType.MUL:
        # Y = A * B (element-wise) → dA = dY * B, dB = dY * A
        if len(expr.children) < 2:
            return grad_dict
        A = expr.children[0]
        B = expr.children[1]

        # dA = dY * B
        dA = mul(output_grad, clone_ast(B))
        inner_grads_A = compute_gradient_recursive(dA, A, wrt_tensors)
        for k, v in inner_grads_A.items():
            grad_dict[k].extend(v)

        # dB = dY * A
        dB = mul(output_grad, clone_ast(A))
        inner_grads_B = compute_gradient_recursive(dB, B, wrt_tensors)
        for k, v in inner_grads_B.items():
            grad_dict[k].extend(v)

    elif expr.node_type == NodeType.DIV:
        # Y = A / B → dA = dY / B, dB = -dY * A / B^2
        if len(expr.children) < 2:
            return grad_dict
        A = expr.children[0]
        B = expr.children[1]

        # dA = dY / B
        dA = div(output_grad, clone_ast(B))
        inner_grads_A = compute_gradient_recursive(dA, A, wrt_tensors)
        for k, v in inner_grads_A.items():
            grad_dict[k].extend(v)

        # dB = -dY * A / B^2
        # For softmax, we often don't need dB for the denominator
        # So we'll skip it for now

    elif expr.node_type == NodeType.BCAST:
        # Y = bcast(X, dim) → dX = rsum(dY, dim)
        if len(expr.children) < 2:
            return grad_dict
        inner_expr = expr.children[0]
        dim = expr.children[1].value if expr.children[1].node_type == NodeType.NUM else 2
        dX = rsum(output_grad, dim)
        inner_grads = compute_gradient_recursive(dX, inner_expr, wrt_tensors)
        for k, v in inner_grads.items():
            grad_dict[k].extend(v)

    elif expr.node_type == NodeType.RSUM:
        # Y = rsum(X, dim) → dX = bcast(dY, dim)
        if len(expr.children) < 2:
            return grad_dict
        inner_expr = expr.children[0]
        dim = expr.children[1].value if expr.children[1].node_type == NodeType.NUM else 2
        dX = bcast(output_grad, dim)
        inner_grads = compute_gradient_recursive(dX, inner_expr, wrt_tensors)
        for k, v in inner_grads.items():
            grad_dict[k].extend(v)

    elif expr.node_type == NodeType.EXP:
        # Y = exp(X) → dX = dY * Y
        if not expr.children:
            return grad_dict
        inner_expr = expr.children[0]
        dX = mul(output_grad, clone_ast(expr))  # dY * exp(X)
        inner_grads = compute_gradient_recursive(dX, inner_expr, wrt_tensors)
        for k, v in inner_grads.items():
            grad_dict[k].extend(v)

    elif expr.node_type == NodeType.UNSQUEEZE:
        # Y = unsqueeze(X, dim) → dX = squeeze(dY, dim)
        if len(expr.children) < 2:
            return grad_dict
        inner_expr = expr.children[0]
        dim = expr.children[1].value if expr.children[1].node_type == NodeType.NUM else 1
        dX = squeeze(output_grad, dim)
        inner_grads = compute_gradient_recursive(dX, inner_expr, wrt_tensors)
        for k, v in inner_grads.items():
            grad_dict[k].extend(v)

    elif expr.node_type == NodeType.LOAD:
        # Base case: load(tensor X)
        if not expr.children:
            return grad_dict
        tensor_node_ast = expr.children[0]
        if tensor_node_ast.node_type in [NodeType.TENSOR, NodeType.INPUT]:
            tensor_name = _extract_tensor_name(tensor_node_ast)
            if tensor_name in wrt_tensors:
                grad_dict[tensor_name].append(output_grad)

    elif expr.node_type == NodeType.ADD:
        # Y = A + B → dA = dY, dB = dY
        for child in expr.children:
            if isinstance(child, ASTNode):
                inner_grads = compute_gradient_recursive(clone_ast(output_grad), child, wrt_tensors)
                for k, v in inner_grads.items():
                    grad_dict[k].extend(v)

    return dict(grad_dict)


def accumulate_gradients(grad_dict: Dict[str, List[ASTNode]]) -> Dict[str, ASTNode]:
    """Accumulate multiple gradients for the same tensor by adding them."""
    result = {}
    for name, grad_list in grad_dict.items():
        if len(grad_list) == 1:
            result[name] = grad_list[0]
        else:
            # Sum all gradients
            acc = grad_list[0]
            for g in grad_list[1:]:
                acc = add(acc, g)
            result[name] = acc
    return result


@dataclass
class GradientContext:
    """Context information for gradient rule generation."""
    store_op: ASTNode
    tensor_name: str
    computation: Optional[ASTNode]
    index: Optional[ASTNode]
    loop_context: List[ASTNode]


@dataclass
class GradientRule:
    """Gradient rule entry."""
    name: str
    stage: str  # 'main' for ploop1, 'weight' for ploop2
    match: Callable[[GradientContext], bool]
    generate: Callable[['GradientRuleTable', GradientContext], List[ASTNode]]


class GradientRuleTable:
    """Table containing gradient generation rules."""

    def __init__(self):
        self.rules: List[GradientRule] = []
        self._initialize_rules()

    def _initialize_rules(self):
        """Register gradient rules."""
        self.rules.extend([
            GradientRule(
                name="O2_squeeze_permute",
                stage="main",
                match=lambda ctx: ctx.tensor_name == "O2",
                generate=self._grad_o2
            ),
            GradientRule(
                name="O_div",
                stage="main",
                match=lambda ctx: ctx.tensor_name == "O" and ctx.computation and ctx.computation.node_type == NodeType.DIV,
                generate=self._grad_o_div
            ),
            GradientRule(
                name="O_elementwise_mul",
                stage="main",
                match=lambda ctx: ctx.tensor_name == "O" and ctx.computation and ctx.computation.node_type == NodeType.MATMUL,
                generate=self._grad_o_matmul
            ),
            GradientRule(
                name="C_sum_rsum",
                stage="main",
                match=lambda ctx: ctx.tensor_name == "C_sum" and ctx.computation and ctx.computation.node_type == NodeType.RSUM,
                generate=self._grad_c_sum
            ),
            GradientRule(
                name="C_exp_exp",
                stage="main",
                match=lambda ctx: ctx.tensor_name == "C_exp" and ctx.computation and ctx.computation.node_type == NodeType.EXP,
                generate=self._grad_c_exp
            ),
            GradientRule(
                name="QKV_permute_unsqueeze",
                stage="main",
                match=lambda ctx: "Q,K,V" in ctx.tensor_name,
                generate=self._grad_qkv
            ),
            GradientRule(
                name="Q1K1V1_weight_grad",
                stage="main",
                match=lambda ctx: ctx.tensor_name == "Q1,K1,V1",
                generate=self._grad_weight_q1k1v1
            ),
        ])

    def get_gradients(self, stage: str, context: GradientContext) -> List[ASTNode]:
        """Generate gradient operations for the given stage."""
        for rule in self.rules:
            if rule.stage == stage and rule.match(context):
                return rule.generate(self, context)
        return []

    # === Rule generators ===

    def _grad_o2(self, _ctx_table: 'GradientRuleTable', ctx: GradientContext) -> List[ASTNode]:
        """
        Gradient for O2.
        Handles both fused and unfused versions using recursive gradient computation.
        """
        if ctx.computation is None:
            return []

        # Initial gradient: dO2
        initial_grad = load_input("dO2", index_fulltile_tilen())

        # Recursively compute gradients
        # We want gradients for all tensors that might be used
        wrt_tensors = {"O", "C_exp", "V", "Q", "K", "C_sum"}

        grad_dict = compute_gradient_recursive(
            initial_grad,
            ctx.computation,
            wrt_tensors
        )

        # Accumulate gradients
        final_grads = accumulate_gradients(grad_dict)

        # Create store operations
        ops = []

        for tensor_name, grad_expr in final_grads.items():
            # Determine appropriate index based on tensor dimensionality
            if tensor_name in ["C_sum"]:
                # 2D tensor: (H, M)
                idx = index_elem_full()
            elif tensor_name in ["O", "C_exp", "V", "Q", "K"]:
                # 3D tensors: (H, M, D)
                idx = index_elem_full_full()
            else:
                # Default: use 3D index
                idx = index_elem_full_full()

            # Create gradient tensor name
            grad_name = f"d{tensor_name}"

            # Check if we need to accumulate (for tensors used multiple times)
            if tensor_name in ["C_exp"]:
                # Accumulate gradient
                ops.append(store_tensor(
                    grad_name,
                    add(load_tensor(grad_name, idx), grad_expr),
                    idx
                ))
            else:
                # Direct assignment
                ops.append(store_tensor(grad_name, grad_expr, idx))

        return ops

    def _grad_o_div(self, _ctx_table: 'GradientRuleTable', ctx: GradientContext) -> List[ASTNode]:
        """Gradient for O = O / bcast(C_sum, 2)."""
        if ctx.computation is None or len(ctx.computation.children) < 2:
            return []

        denominator = ctx.computation.children[1]
        index_main = ctx.index or index_elem_full_full()

        grad_ops = []

        # dO_tmp accumulation
        grad_ops.append(
            store_tensor(
                "dO_tmp",
                add(
                    load_tensor("dO_tmp", index_main),
                    mul(
                        load_tensor("dO", index_main),
                        div(num(1), clone_ast(denominator))
                    )
                ),
                index_main
            )
        )

        # dC_sum accumulation
        grad_ops.append(
            store_tensor(
                "dC_sum",
                add(
                    load_tensor("dC_sum", index_elem_full()),
                    sub(
                        num(0),
                        rsum(
                            mul(
                                load_tensor("dO", index_main),
                                div(
                                    load_tensor("O", index_main),
                                    clone_ast(denominator)
                                )
                            ),
                            2
                        )
                    )
                ),
                index_elem_full()
            )
        )

        return grad_ops

    def _grad_o_matmul(self, _ctx_table: 'GradientRuleTable', ctx: GradientContext) -> List[ASTNode]:
        """Gradient for O = C_exp @ V (matrix multiplication)."""
        if ctx.computation is None or len(ctx.computation.children) < 2:
            return []

        index_main = ctx.index or index_elem_full_full()
        # Forward: O = C_exp @ V
        # Backward: dC_exp = dO @ V^T, dV = C_exp^T @ dO

        grad_ops = [
            # dC_exp += dO_tmp @ V^T
            store_tensor(
                "dC_exp",
                add(
                    load_tensor("dC_exp", index_main),
                    matmul(
                        load_tensor("dO_tmp", index_main),
                        permute3(load_tensor("V", index_main), 0, 2, 1)
                    )
                ),
                index_main
            ),
            # dV = C_exp^T @ dO_tmp
            store_tensor(
                "dV",
                matmul(
                    permute3(load_tensor("C_exp", index_main), 0, 2, 1),
                    load_tensor("dO_tmp", index_main)
                ),
                index_main
            )
        ]

        return grad_ops

    def _grad_c_sum(self, _ctx_table: 'GradientRuleTable', _ctx: GradientContext) -> List[ASTNode]:
        """Gradient for C_sum = rsum(C_exp, axis=2)."""
        index_main = index_elem_full_full()
        grad_op = store_tensor(
            "dC_exp",
            add(
                load_tensor("dC_exp", index_main),
                bcast(
                    load_tensor("dC_sum", index_elem_full()),
                    2
                )
            ),
            index_main
        )
        return [grad_op]

    def _grad_c_exp(self, _ctx_table: 'GradientRuleTable', ctx: GradientContext) -> List[ASTNode]:
        """Gradient for C_exp = exp(Q @ K^T)."""
        if ctx.computation is None or not ctx.computation.children:
            return []

        index_main = ctx.index or index_elem_full_full()

        grad_ops: List[ASTNode] = []

        # dC accumulation (gradient of exp)
        # C_exp = exp(C) → dC = dC_exp * C_exp
        grad_ops.append(
            store_tensor(
                "dC",
                add(
                    load_tensor("dC", index_main),
                    mul(
                        load_tensor("dC_exp", index_main),
                        load_tensor("C_exp", index_main)
                    )
                ),
                index_main
            )
        )

        inner = ctx.computation.children[0]
        if inner.node_type == NodeType.MATMUL and len(inner.children) >= 2:
            # C = Q @ K^T → dQ = dC @ K, dK = dC^T @ Q
            # dQ accumulation
            grad_ops.append(
                store_tensor(
                    "dQ",
                    add(
                        load_tensor("dQ", index_main),
                        matmul(
                            load_tensor("dC", index_main),
                            load_tensor("K", index_main)
                        )
                    ),
                    index_main
                )
            )

            # dK accumulation
            grad_ops.append(
                store_tensor(
                    "dK",
                    add(
                        load_tensor("dK", index_main),
                        matmul(
                            permute3(load_tensor("dC", index_main), 0, 2, 1),
                            load_tensor("Q", index_main)
                        )
                    ),
                    index_main
                )
            )

        return grad_ops

    def _grad_qkv(self, _ctx_table: 'GradientRuleTable', ctx: GradientContext) -> List[ASTNode]:
        """Gradient for Q,K,V = permute3(unsqueeze(Q1,K1,V1))."""
        forward_index = ctx.index or index_elem_full_full()

        grad_op = store_tensor(
            "dQ1,dK1,dV1",
            squeeze(
                permute3(
                    load_tensor("dQ,dK,dV", forward_index),
                    1, 0, 2
                ),
                1
            ),
            index_fulltile_tilen()
        )
        return [grad_op]

    def _grad_weight_q1k1v1(self, _ctx_table: 'GradientRuleTable', ctx: GradientContext) -> List[ASTNode]:
        """Gradient for Q1,K1,V1 projection weights."""
        grad_stores = [
            store_tensor(
                "dWQ",
                matmul(
                    transpose(load_input("X", index_fulltile_tilek())),
                    load_tensor("dQ1", index_fulltile_tilen())
                ),
                index_tilek_tilen()
            ),
            store_tensor(
                "dWK",
                matmul(
                    transpose(load_input("X", index_fulltile_tilek())),
                    load_tensor("dK1", index_fulltile_tilen())
                ),
                index_tilek_tilen()
            ),
            store_tensor(
                "dWV",
                matmul(
                    transpose(load_input("X", index_fulltile_tilek())),
                    load_tensor("dV1", index_fulltile_tilen())
                ),
                index_tilek_tilen()
            ),
        ]

        return grad_stores


__all__ = [
    "GradientRuleTable",
    "GradientContext",
    "clone_ast",
]
