#!/usr/bin/env python3
"""
Backward IR Generator using IRParser and Gradient Rule Table
Generates backward IR with automatic recomputation from forward IR
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from IrParser import IRParser
from AstNode import ASTNode
from NodeType import NodeType
from gradient_rules import GradientRuleTable, GradientContext, clone_ast
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
from format import format_lisp_with_rules

@dataclass
class TensorInfo:
    """Information about a tensor"""
    name: str
    tensor_type: NodeType  # TENSOR, INPUT, OUTPUT
    is_offchip: bool
    computation_node: Optional[ASTNode]  # Store node that creates this tensor
    kernel_id: int  # Which kernel (ploop) it belongs to
    dependencies: Set[str]  # Other tensors this depends on

@dataclass
class KernelInfo:
    """Information about a kernel (ploop)"""
    kernel_id: int
    node: ASTNode
    tensors_created: Set[str]
    tensors_used: Set[str]

class BackwardIRGenerator:
    """Generate backward IR from forward IR using AST and gradient rules"""

    def __init__(self):
        self.parser = IRParser()
        self.gradient_rules = GradientRuleTable()
        self.tensor_info = {}  # tensor_name -> TensorInfo
        self.kernel_info = {}  # kernel_id -> KernelInfo
        self.tape = []  # List of operations in order
        self.current_kernel_id = 0
        self.gradient_map = {}  # tensor_name -> gradient_tensor_name
        self.needs_recompute = False
        self.loop_context = {}  # id(operation) -> loop node containing it
        self.current_loop_stack = []  # Stack of current loop nodes
        self.forward_loop_header = None  # Cached forward ploop header metadata
        self.forward_input_names: Set[str] = set()
        self.forward_output_names: Set[str] = set()
        self.backward_input_names: Set[str] = set()
        self.backward_output_names: Set[str] = set()

    def analyze_forward_ir(self, forward_ir: str):
        """Parse and analyze forward IR"""
        # Parse the IR
        self.forward_ast = self.parser.parse(forward_ir)
        self.forward_input_names.clear()
        self.forward_output_names.clear()

        # Cache outer ploop header for reuse in backward kernels
        if self.forward_ast.node_type == NodeType.PLOOP and len(self.forward_ast.children) >= 4:
            self.forward_loop_header = [clone_ast(child) for child in self.forward_ast.children[:4]]

        # Analyze the AST
        self._analyze_node(self.forward_ast, None)

        # Determine which tensors are offchip
        self._identify_offchip_tensors()

    def _analyze_node(self, node: ASTNode, current_kernel: Optional[int]):
        """Recursively analyze AST node"""

        if node.node_type == NodeType.PLOOP:
            # New kernel boundary
            self.current_kernel_id += 1
            kernel = KernelInfo(
                kernel_id=self.current_kernel_id,
                node=node,
                tensors_created=set(),
                tensors_used=set()
            )
            self.kernel_info[self.current_kernel_id] = kernel

            # Process children within this kernel
            for child in node.children[4:]:  # Skip loop bounds and iterator
                self._analyze_node(child, self.current_kernel_id)

        elif node.node_type == NodeType.SLOOP:
            # Track loop context
            self.current_loop_stack.append(node)

            # Process children within this loop
            for child in node.children[4:]:  # Skip loop bounds and iterator
                self._analyze_node(child, current_kernel)

            # Pop loop context
            self.current_loop_stack.pop()

        elif node.node_type == NodeType.STORE:
            # Store operation - record it
            if len(node.children) >= 3:
                target = node.children[0]
                computation = node.children[1]
                if target.node_type == NodeType.OUTPUT:
                    self.forward_output_names.update(self._extract_name_parts(target))

                tensor_name = self._extract_tensor_name(target)
                tensor_type = target.node_type

                if tensor_name:
                    # Record tensor info
                    deps = self._extract_dependencies(computation)
                    self.tensor_info[tensor_name] = TensorInfo(
                        name=tensor_name,
                        tensor_type=tensor_type,
                        is_offchip=False,  # Will be determined later
                        computation_node=node,
                        kernel_id=current_kernel or 0,
                        dependencies=deps
                    )

                    # Record in tape with loop context
                    self.tape.append(node)

                    # Store loop context for this operation
                    if self.current_loop_stack:
                        self.loop_context[id(node)] = list(self.current_loop_stack)

                    # Track in kernel info
                    if current_kernel and current_kernel in self.kernel_info:
                        self.kernel_info[current_kernel].tensors_created.add(tensor_name)
                        self.kernel_info[current_kernel].tensors_used.update(deps)

        elif node.node_type == NodeType.LOAD:
            if node.children:
                tensor = node.children[0]
                if tensor.node_type == NodeType.INPUT:
                    self.forward_input_names.update(self._extract_name_parts(tensor))
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._analyze_node(child, current_kernel)

        elif node.node_type == NodeType.SEQ:
            # Continue analyzing children
            for child in node.children:
                if child.node_type != NodeType.DUMMY:
                    self._analyze_node(child, current_kernel)
        else:
            # Process children
            for child in node.children:
                if isinstance(child, ASTNode):
                    self._analyze_node(child, current_kernel)

    def _extract_tensor_name(self, node: ASTNode) -> str:
        """Extract tensor name from tensor/input/output node"""
        if node.node_type in [NodeType.TENSOR, NodeType.INPUT, NodeType.OUTPUT]:
            if node.children:
                # Handle multiple names (Q,K,V)
                names = []
                for child in node.children:
                    if child.node_type == NodeType.VAR:
                        names.append(child.value)
                if node.node_type == NodeType.INPUT:
                    self.forward_input_names.update(names)
                elif node.node_type == NodeType.OUTPUT:
                    self.forward_output_names.update(names)
                return ','.join(names) if names else ''
        return ''

    def _extract_name_parts(self, node: ASTNode) -> List[str]:
        parts = []
        for child in node.children:
            if child.node_type == NodeType.VAR:
                parts.append(child.value)
        return parts

    def _extract_dependencies(self, node: ASTNode) -> Set[str]:
        """Extract all tensor dependencies from a computation node"""
        deps = set()

        def traverse(n):
            if n.node_type == NodeType.LOAD:
                if n.children and len(n.children) > 0:
                    tensor_name = self._extract_tensor_name(n.children[0])
                    if tensor_name:
                        deps.add(tensor_name)
            for child in n.children:
                if isinstance(child, ASTNode):
                    traverse(child)

        traverse(node)
        return deps

    def _identify_offchip_tensors(self):
        """Identify which tensors are stored offchip"""
        # INPUT and OUTPUT are always offchip
        for name, info in self.tensor_info.items():
            if info.tensor_type in [NodeType.INPUT, NodeType.OUTPUT]:
                info.is_offchip = True

        # Cross-kernel tensors are offchip
        for name, info in self.tensor_info.items():
            if info.tensor_type == NodeType.TENSOR:
                # Check if used in different kernel
                creating_kernel = info.kernel_id
                for _, other_info in self.tensor_info.items():
                    if name in other_info.dependencies:
                        if other_info.kernel_id != creating_kernel:
                            info.is_offchip = True
                            break

    def _filter_offchip_stores(self, node: ASTNode) -> ASTNode:
        """Remove store operations for off-chip tensors from recompute section"""
        if node.node_type == NodeType.STORE:
            # Check if this store creates an off-chip tensor
            if len(node.children) >= 1:
                target = node.children[0]
                tensor_name = self._extract_tensor_name(target)
                if tensor_name:
                    info = self.tensor_info.get(tensor_name)
                    if info and info.is_offchip:
                        # Replace with dummy to maintain structure
                        return ASTNode(NodeType.DUMMY, [])

        # Recursively process children
        new_children = []
        for child in node.children:
            if isinstance(child, ASTNode):
                filtered_child = self._filter_offchip_stores(child)
                new_children.append(filtered_child)
            else:
                new_children.append(child)

        return ASTNode(node.node_type, new_children, node.value)

    def generate_backward(self) -> str:
        """Generate backward IR"""
        # Reset recompute tracking per generation
        self.needs_recompute = False
        self.backward_input_names = {self._gradient_name(name) for name in self.forward_output_names}
        self.backward_output_names = {self._gradient_name(name) for name in self.forward_input_names}

        sections = self._generate_gradient_sections()
        if self.needs_recompute and hasattr(self, "forward_ast"):
            # Filter out off-chip tensor stores from recompute
            recompute_ast = self._filter_offchip_stores(clone_ast(self.forward_ast))
            sections.insert(0, recompute_ast)

        backward_ir = self._chain_seq(sections)
        backward_ir = self._adjust_gradient_io(backward_ir)

        # Convert AST back to string
        return self._ast_to_ir_string(backward_ir)

    def _generate_gradient_sections(self) -> List[ASTNode]:
        """Generate gradient sections (main + weight)"""
        gradient_ops: List[ASTNode] = []

        for op in reversed(self.tape):
            if op.node_type != NodeType.STORE or not op.children:
                continue

            tensor_name = self._extract_tensor_name(op.children[0])
            if not tensor_name:
                continue

            grad_ops = self._generate_gradient_for_operation(op, tensor_name)
            if grad_ops:
                gradient_ops.extend(grad_ops)

        sections: List[ASTNode] = []

        if gradient_ops:
            wrapped = self._wrap_gradients_naive(gradient_ops)
            if wrapped:
                sections.append(wrapped)

        return sections

    def _generate_gradient_for_operation(self, store_op: ASTNode, tensor_name: str) -> List[ASTNode]:
        """Generate gradient computation for a specific operation using gradient rules"""
        if len(store_op.children) < 2:
            return []

        computation = store_op.children[1]
        index = store_op.children[2] if len(store_op.children) > 2 else None
        loops = self.loop_context.get(id(store_op), [])

        # Ensure dependencies are available (recompute on-chip tensors)
        self._add_recompute_if_needed(computation)

        # Ensure the tensor itself is available if it was on-chip
        tensor_info = self.tensor_info.get(tensor_name)
        if tensor_info and not tensor_info.is_offchip:
            self._ensure_tensor_recomputed(tensor_name)

        context = GradientContext(
            store_op=store_op,
            tensor_name=tensor_name,
            computation=computation,
            index=index,
            loop_context=loops,
        )

        return self.gradient_rules.get_gradients("main", context)

    def _resolve_tensor_key(self, name: str) -> Optional[str]:
        """Resolve a tensor alias (like 'Q') to the stored key (e.g., 'Q,K,V')"""
        if name in self.tensor_info:
            return name

        for key in self.tensor_info.keys():
            parts = [part.strip() for part in key.split(',')]
            if name in parts:
                return key
        return None

    def _ensure_tensor_recomputed(self, tensor_name: str):
        """Mark that recomputation is needed for a tensor stored on-chip"""
        resolved_name = self._resolve_tensor_key(tensor_name)
        if not resolved_name:
            return

        info = self.tensor_info.get(resolved_name)
        if info and not info.is_offchip:
            self.needs_recompute = True

    def _extract_tensor_from_load(self, node: ASTNode) -> str:
        """Extract tensor name from a load node"""
        if node.node_type == NodeType.LOAD and node.children:
            tensor_node = node.children[0]
            if tensor_node.node_type == NodeType.TENSOR and tensor_node.children:
                if tensor_node.children[0].node_type == NodeType.VAR:
                    return tensor_node.children[0].value
        return ""

    def _tensor_has_gradient(self, tensor_name: str) -> bool:
        """Check if tensor needs gradient computation"""
        if not tensor_name:
            return False

        # Output always has gradient
        info = self.tensor_info.get(tensor_name)
        if info and info.tensor_type == NodeType.OUTPUT:
            if tensor_name not in self.gradient_map:
                grad_name = f"grad_{tensor_name.replace(',', '_')}"
                self.gradient_map[tensor_name] = grad_name
            return True

        # Check if already has gradient
        if tensor_name in self.gradient_map:
            return True

        # Check if contributes to loss
        if self._contributes_to_loss(tensor_name):
            grad_name = f"grad_{tensor_name.replace(',', '_')}"
            self.gradient_map[tensor_name] = grad_name
            return True

        return False

    def _create_gradient_init(self) -> ASTNode:
        """Create gradient initialization from output"""
        # Find output tensor
        output_name = None
        for name, info in self.tensor_info.items():
            if info.tensor_type == NodeType.OUTPUT:
                output_name = name
                break

        if output_name:
            # For attention, we typically start with dO2
            grad_name = f"d{output_name}"
            self.gradient_map[output_name] = grad_name

            # No initialization needed - dO2 will come from upstream gradient
            # Return empty node
            return ASTNode(NodeType.SEQ, [])

        return ASTNode(NodeType.SEQ, [])  # Empty if no output


    def _add_recompute_if_needed(self, node: ASTNode):
        """Check dependencies and flag recomputation when needed"""
        if node is None:
            return

        deps = self._extract_dependencies(node)
        for dep in deps:
            resolved_name = self._resolve_tensor_key(dep)
            if not resolved_name:
                continue
            info = self.tensor_info.get(resolved_name)
            if info and not info.is_offchip:
                self.needs_recompute = True

    def _extract_operands(self, computation: ASTNode) -> List[ASTNode]:
        """Extract operand nodes from computation - extract all LOAD nodes"""
        operands = []

        def find_load_nodes(node: ASTNode, depth: int = 0) -> List[ASTNode]:
            """Recursively find LOAD nodes representing operands"""
            loads = []

            # If this is a LOAD node, it's an operand
            if node.node_type == NodeType.LOAD:
                loads.append(node)
            # For operations that modify structure (permute, squeeze, etc.), keep the operation
            elif node.node_type in [NodeType.PERMUTE3, NodeType.SQUEEZE, NodeType.UNSQUEEZE] and depth == 0:
                # For structure-modifying operations at the top level, keep the whole operation
                loads.append(node)
            else:
                # Recursively search children
                for child in node.children:
                    if isinstance(child, ASTNode):
                        loads.extend(find_load_nodes(child, depth + 1))

            return loads

        # Handle different operation types
        if computation.node_type == NodeType.ADD:
            # For addition, extract operands from each child
            for child in computation.children:
                if isinstance(child, ASTNode):
                    if child.node_type == NodeType.MUL and len(child.children) >= 2:
                        # For (x 1 (load ...)), extract the load
                        if child.children[0].node_type == NodeType.NUM and child.children[0].value == 1:
                            loads = find_load_nodes(child.children[1])
                            operands.extend(loads)
                        else:
                            loads = find_load_nodes(child)
                            operands.extend(loads)
                    elif child.node_type == NodeType.MATMUL:
                        # For matrix multiplication within ADD
                        loads = find_load_nodes(child)
                        operands.extend(loads)
                    else:
                        loads = find_load_nodes(child)
                        operands.extend(loads)

        elif computation.node_type == NodeType.MATMUL:
            # For matrix multiplication, get both operands
            if len(computation.children) >= 2:
                left_operand = computation.children[0]
                right_operand = computation.children[1]

                # Keep structure-preserving operations like PERMUTE3 intact
                if right_operand.node_type == NodeType.PERMUTE3:
                    operands.append(right_operand)
                else:
                    operands.extend(find_load_nodes(right_operand))

                operands.extend(find_load_nodes(left_operand))

        elif computation.node_type == NodeType.MUL:
            # For multiplication (often scalar multiplication)
            for child in computation.children:
                if isinstance(child, ASTNode) and child.node_type != NodeType.NUM:
                    loads = find_load_nodes(child)
                    operands.extend(loads)

        elif computation.node_type in [NodeType.EXP, NodeType.RSUM, NodeType.BCAST]:
            # For unary operations
            if computation.children:
                operands.extend(find_load_nodes(computation.children[0]))

        elif computation.node_type in [NodeType.PERMUTE3, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            # For structure operations, extract the input
            if computation.children:
                operands.extend(find_load_nodes(computation.children[0]))

        else:
            # Default: find all LOAD nodes
            operands = find_load_nodes(computation)

        return operands

    def _is_weight_tensor(self, tensor_name: str) -> bool:
        """Check if tensor is a weight (parameter) tensor"""
        # Simple heuristic: check if name contains W
        return 'W' in tensor_name

    def _wrap_gradients_naive(self, ops: List[ASTNode]) -> Optional[ASTNode]:
        """Wrap gradient operations in naive loop structures."""
        if not ops:
            return None

        wrapped_ops: List[ASTNode] = []
        for op in ops:
            target_name = ""
            if op.node_type == NodeType.STORE and op.children:
                target_name = self._extract_tensor_name(op.children[0])

            op_clone = clone_ast(op)

            if self._is_weight_gradient(target_name):
                inner = self._make_loop(0, 4544, 'tile_k', 'k', op_clone)
                wrapped = self._make_loop(0, 4544, 'tile_n', 'n', inner)
            else:
                wrapped = self._make_loop(0, 4544, 'tile_n', 'n', op_clone)

            wrapped_ops.append(wrapped)

        return self._chain_seq(wrapped_ops)

    def _make_loop(self, start: int, end: int, tile_name: str, var_name: str, body: ASTNode) -> ASTNode:
        """Create a naive sequential loop node"""
        return ASTNode(NodeType.LOOP, [
            ASTNode(NodeType.NUM, [], start),
            ASTNode(NodeType.NUM, [], end),
            ASTNode(NodeType.VAR, [], tile_name),
            ASTNode(NodeType.VAR, [], var_name),
            body
        ])

    def _chain_seq(self, nodes: List[ASTNode]) -> ASTNode:
        """Create right-associated seq chain from list of nodes"""
        if not nodes:
            return ASTNode(NodeType.SEQ, [])
        if len(nodes) == 1:
            return nodes[0]

        result = nodes[-1]
        for node in reversed(nodes[:-1]):
            result = ASTNode(NodeType.SEQ, [node, result])
        return result

    def _is_weight_gradient(self, target_name: str) -> bool:
        if not target_name:
            return False
        names = [name.strip() for name in target_name.split(',') if name.strip()]
        return bool(names) and all(name.startswith('dW') for name in names)

    def _gradient_name(self, name: str) -> str:
        if name.startswith('d'):
            return name
        return f"d{name}"

    def _adjust_gradient_io(self, node: ASTNode) -> ASTNode:
        if node is None:
            return node

        new_children: List[ASTNode] = []
        for child in node.children:
            if isinstance(child, ASTNode):
                new_children.append(self._adjust_gradient_io(child))
            else:
                new_children.append(child)

        if node.node_type in [NodeType.TENSOR, NodeType.INPUT, NodeType.OUTPUT]:
            names = [child.value for child in node.children if child.node_type == NodeType.VAR]
            if names and all(name in self.backward_input_names for name in names):
                return ASTNode(NodeType.INPUT, new_children, node.value)
            if names and all(name in self.backward_output_names for name in names):
                return ASTNode(NodeType.OUTPUT, new_children, node.value)

        return ASTNode(node.node_type, new_children, node.value)

    def _contributes_to_loss(self, tensor_name: str) -> bool:
        """Check if tensor contributes to loss"""
        # For backward pass generation, assume all tensors contribute to loss
        # This ensures gradients are computed for all intermediate tensors
        return True

    def _ast_to_ir_string(self, node: ASTNode, indent: int = 0) -> str:
        """Convert AST back to IR string format"""
        indent_str = "  " * indent

        if node.node_type == NodeType.NUM:
            return str(node.value)
        elif node.node_type == NodeType.VAR:
            return str(node.value)
        elif node.node_type == NodeType.FULLTILE:
            return "fulltile"
        elif node.node_type == NodeType.DUMMY:
            return "dummy"
        else:
            # Convert node type to operation string
            op_map = {
                NodeType.LOOP: "loop",
                NodeType.PLOOP: "ploop",
                NodeType.SLOOP: "sloop",
                NodeType.SEQ: "seq",
                NodeType.STORE: "store",
                NodeType.LOAD: "load",
                NodeType.ADD: "+",
                NodeType.SUB: "-",
                NodeType.MUL: "x",
                NodeType.MATMUL: "*",
                NodeType.DIV: "/",
                NodeType.EXP: "exp",
                NodeType.TENSOR: "tensor",
                NodeType.INPUT: "input",
                NodeType.OUTPUT: "output",
                NodeType.INDEX: "index",
                NodeType.TILE: "tile",
                NodeType.ELEM: "elem",
                NodeType.TRANSPOSE: "transpose",
                NodeType.PERMUTE3: "permute3",
                NodeType.SQUEEZE: "squeeze",
                NodeType.UNSQUEEZE: "unsqueeze",
                NodeType.RSUM: "rsum",
                NodeType.BCAST: "bcast",
                NodeType.CONST_TILE: "const_tile",
            }

            op = op_map.get(node.node_type, str(node.node_type.value))

            if not node.children:
                return f"({op})"

            # Format children
            children_strs = []
            for child in node.children:
                if isinstance(child, ASTNode):
                    child_str = self._ast_to_ir_string(child, indent + 1)
                    children_strs.append(child_str)
                else:
                    children_strs.append(str(child))

            # Special formatting for certain node types
            if node.node_type in [NodeType.TENSOR, NodeType.INPUT, NodeType.OUTPUT]:
                # Handle comma-separated names
                names = []
                for child in node.children:
                    if child.node_type == NodeType.VAR:
                        names.append(child.value)
                if names:
                    return f"({op} {','.join(names)})"

            # Multi-line formatting for complex nodes
            if node.node_type in [NodeType.STORE, NodeType.SEQ, NodeType.PLOOP, NodeType.SLOOP]:
                result = f"({op}"
                for i, child_str in enumerate(children_strs):
                    if i == 0 and node.node_type != NodeType.SEQ:
                        result += f" {child_str}"
                    else:
                        result += f"\n{indent_str}  {child_str}"
                result += ")"
                return result
            else:
                # Single line for simple operations
                return f"({op} {' '.join(children_strs)})"

def convert_group(file_path):
    expressions = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    id_part = parts[0]
                    try:
                        ir_id = int(id_part)
                        ir_expr = parts[1].strip()
                        if 'dummydata' not in ir_expr:
                            expressions.append((ir_id, ir_expr))
                    except:
                        continue
    
    return expressions

def bwd_generate(ir_expr: str, ir_id: int):
    generator = BackwardIRGenerator()
    
    # Analyze forward IR
    print("Analyzing forward IR...")
    generator.analyze_forward_ir(ir_expr)

    # Print analysis results
    print(f"\nFound {len(generator.tape)} operations")
    print(f"Found {len(generator.tensor_info)} tensors")

    print("\nTensor storage analysis:")
    for name, info in sorted(generator.tensor_info.items()):
        storage = "OFF-CHIP" if info.is_offchip else "ON-CHIP"
        kernel = f"kernel {info.kernel_id}" if info.kernel_id else "no kernel"
        print(f"  {name:30s} -> {storage:8s} ({kernel})")

    # Generate backward IR
    print("\nGenerating backward IR...")
    backward_ir = format_lisp_with_rules(generator.generate_backward())

    # Save result
    output_file = f'seq16_bwd{ir_id}.txt'
    with open(output_file, 'w') as f:
        f.write(backward_ir)

    print(f"Backward IR saved to '{output_file}'")
    for idx, computation in enumerate(generator.tape):
        print(f"[{idx}]: {computation}")

def main():
    """Main function"""

    argparser = argparse.ArgumentParser(description="Convert FWD IR to BWD IR")
    argparser.add_argument("--n", type=int, help="Case number to convert")
    argparser.add_argument("--file", action="store_true", help="Convert entire tile IR list file")
    args = argparser.parse_args()

    num = args.n
    file = args.file

    # Read forward IR
    if not file:
        with open(f'seq16_fwd{num}.txt', 'r') as f:
            forward_ir = f.read()
        bwd_generate(forward_ir, num)
    else:
        expressions = convert_group('seq16_vanilla.txt')
        for id, expr in expressions:
            bwd_generate(expr, id)

if __name__ == "__main__":
    main()
