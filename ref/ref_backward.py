import torch
import torch.nn as nn

class AttentionBackward(nn.Module):
    """
    PyTorch reference implementation for attention backward pass.
    Forward: X -> Q1, K1, V1 -> Attention -> O2
    Backward: dO2 -> dWQ, dWK, dWV
    """
    def __init__(self, M, N, D, H):
        super().__init__()
        self.M = M  # Batch size
        self.N = N  # Feature dimension
        self.D = D  # Head dimension (D = N // H for this case)
        self.H = H  # Number of heads

    def forward(self, X, WQ, WK, WV):
        """
        Forward pass matching the IR.

        Args:
            X: (M, N)
            WQ, WK, WV: (N, N)

        Returns:
            O2: (M, N)
        """
        M, N, D, H = self.M, self.N, self.D, self.H

        # Project to Q1, K1, V1
        Q1 = X @ WQ  # (M, N)
        K1 = X @ WK  # (M, N)
        V1 = X @ WV  # (M, N)

        # Reshape to (M, H, D) then permute to (H, M, D)
        Q = Q1.view(M, H, D).permute(1, 0, 2)  # (H, M, D)
        K = K1.view(M, H, D).permute(1, 0, 2)  # (H, M, D)
        V = V1.view(M, H, D).permute(1, 0, 2)  # (H, M, D)

        # Attention computation
        # C_exp = exp(Q @ K^T)  # (H, M, M)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (H, M, M)
        C_exp = torch.exp(scores)  # (H, M, M)

        # C_sum = sum(C_exp, dim=-1)  # (H, M)
        C_sum = C_exp.sum(dim=-1, keepdim=False)  # (H, M)

        # O = (C_exp @ V) / C_sum  # (H, M, D)
        O = torch.matmul(C_exp, V)  # (H, M, D)
        O = O / C_sum.unsqueeze(-1)  # (H, M, D)

        # Reshape back to O2
        # O2 = squeeze(permute(O, 1, 0, 2), 1)
        O_permuted = O.permute(1, 0, 2)  # (M, H, D)
        O2 = O_permuted.reshape(M, N)  # (M, N)

        return O2, Q1


def compute_gradients(M, N, D, H, X, WQ, WK, WV, dO2, C_exp=None, device='cuda:2', dtype=torch.float16):
    """
    Compute gradients using PyTorch autograd or manual computation with C_exp.

    Args:
        C_exp: If provided, uses manual backward computation instead of autograd

    Returns:
        dWQ, dWK, dWV: (N, N) gradients
    """
    # Ensure requires_grad
    WQ = WQ.clone().detach().requires_grad_(True)
    WK = WK.clone().detach().requires_grad_(True)
    WV = WV.clone().detach().requires_grad_(True)
    X = X.clone().detach()

    # if C_exp is None:
    # Standard autograd path
    model = AttentionBackward(M, N, D, H).to(device).to(dtype)
    O2, Q1 = model(X, WQ, WK, WV)
    O2.backward(dO2)
    return WQ.grad, WK.grad, WV.grad, Q1
    # else:
    #     # Manual backward computation using provided C_exp
    #     # Forward pass (recompute needed intermediate values)
    #     Q1 = X @ WQ  # (M, N)
    #     K1 = X @ WK  # (M, N)
    #     V1 = X @ WV  # (M, N)

    #     Q = Q1.view(M, H, D).permute(1, 0, 2)  # (H, M, D)
    #     K = K1.view(M, H, D).permute(1, 0, 2)  # (H, M, D)
    #     V = V1.view(M, H, D).permute(1, 0, 2)  # (H, M, D)

    #     # Backward computation
    #     # dO2 -> dO
    #     dO = dO2.view(M, H, D).permute(1, 0, 2)  # (H, M, D)

    #     # C_exp normalization
    #     C_sum = C_exp.sum(dim=-1, keepdim=True)  # (H, M, 1)
    #     C_normalized = C_exp / C_sum  # (H, M, M)

    #     # dL/dV = C_normalized^T @ dO
    #     dV = torch.matmul(C_normalized.transpose(-2, -1), dO)  # (H, M, D)

    #     # dL/dC_normalized = dO @ V^T
    #     dC_normalized = torch.matmul(dO, V.transpose(-2, -1))  # (H, M, M)

    #     # Backward through normalization (C_exp / C_sum)
    #     # dL/dC_exp = dL/dC_normalized * (1/C_sum - C_exp/(C_sum^2) * sum_j(dL/dC_normalized))
    #     dC_sum = -(dC_normalized * C_normalized).sum(dim=-1, keepdim=True)  # (H, M, 1)
    #     dC_exp = dC_normalized / C_sum + dC_sum * C_exp / (C_sum ** 2)  # (H, M, M)

    #     # Backward through exp (assuming C = exp(scores))
    #     dC = dC_exp * C_exp  # (H, M, M)

    #     # dL/dQ = dC @ K
    #     dQ = torch.matmul(dC, K)  # (H, M, D)

    #     # dL/dK = dC^T @ Q
    #     dK = torch.matmul(dC.transpose(-2, -1), Q)  # (H, M, D)

    #     # Reshape gradients back to (M, N)
    #     dQ1 = dQ.permute(1, 0, 2).reshape(M, N)  # (M, N)
    #     dK1 = dK.permute(1, 0, 2).reshape(M, N)  # (M, N)
    #     dV1 = dV.permute(1, 0, 2).reshape(M, N)  # (M, N)

    #     # Compute weight gradients
    #     dWQ = X.T @ dQ1  # (N, N)
    #     dWK = X.T @ dK1  # (N, N)
    #     dWV = X.T @ dV1  # (N, N)

    #     return dWQ, dWK, dWV


def test_backward():
    """Test backward pass computation."""
    # Falcon config
    M = 16
    D = 64
    N = 4544
    H = 71

    device = torch.device('cuda:2')
    dtype = torch.float16

    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Generate inputs
    std = 0.01
    X = torch.randn((M, N), device=device, dtype=dtype) * std
    WQ = torch.randn((N, N), device=device, dtype=dtype) * std
    WK = torch.randn((N, N), device=device, dtype=dtype) * std
    WV = torch.randn((N, N), device=device, dtype=dtype) * std
    dO2 = torch.ones((M, N), device=device, dtype=dtype)

    # Compute gradients
    dWQ, dWK, dWV = compute_gradients(M, N, D, H, X, WQ, WK, WV, dO2, device, dtype)

    print("PyTorch Reference Results:")
    print(f"dWQ shape: {dWQ.shape}")
    print(f"dWQ sample:\n{dWQ[:5, :5]}")
    print(f"\ndWK shape: {dWK.shape}")
    print(f"dWK sample:\n{dWK[:5, :5]}")
    print(f"\ndWV shape: {dWV.shape}")
    print(f"dWV sample:\n{dWV[:5, :5]}")


if __name__ == "__main__":
    test_backward()
