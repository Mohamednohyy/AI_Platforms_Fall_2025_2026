import torch
from torch import nn


class ManualLinear(nn.Module):
    """A tiny fully-connected layer implemented with explicit Parameters.

    Uses weight shape (out_features, in_features) and bias shape (out_features,).
    Forward computes x @ W^T + b. Accepts scalar or 1D input and returns 1D output.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Small random initialization; gradients flow via autograd
        weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.5
        bias = torch.zeros(out_features, dtype=torch.float32)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is 1D: (in_features,) — allow scalar by unsqueezing
        if x.ndim == 0:
            x = x.unsqueeze(0)
        return torch.matmul(self.weight, x) + self.bias


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class Tanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class TwoBranchComputationalGraph(nn.Module):
    """Network following the described structure using manual ops and activations.

    Structure:
      - Branch A: Linear(1->3) → ReLU → sum to scalar
      - Branch B: Linear(1->2) → Sigmoid → sum to scalar
      - Combine: scalar_a + scalar_b → Tanh
      - Output: Linear(1->1) (no activation)
    """

    def __init__(self):
        super().__init__()
        self.layer1 = ManualLinear(in_features=1, out_features=3)
        self.act1 = ReLU()

        self.layer2 = ManualLinear(in_features=1, out_features=2)
        self.act2 = Sigmoid()

        self.combine_act = Tanh()

        # Output linear layer that consumes a single scalar feature
        self.output_layer = ManualLinear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor, *, print_intermediates: bool = False) -> torch.Tensor:
        # Expect scalar input; ensure requires_grad is preserved
        if x.ndim == 1 and x.numel() == 1:
            x_scalar = x.squeeze(0)
        elif x.ndim == 0:
            x_scalar = x
        else:
            raise ValueError("This example expects a scalar input tensor.")

        # Branch A: 1 -> 3 with ReLU
        a_linear = self.layer1(x_scalar)
        a_activated = self.act1(a_linear)
        a_sum = a_activated.sum()

        # Branch B: 1 -> 2 with Sigmoid
        b_linear = self.layer2(x_scalar)
        b_activated = self.act2(b_linear)
        b_sum = b_activated.sum()

        combined = a_sum + b_sum
        combined_act = self.combine_act(combined)

        # Output layer expects a length-1 vector; convert scalar feature accordingly
        output = self.output_layer(combined_act)

        if print_intermediates:
            print("Input x:", x_scalar.detach().item())
            print("Layer1 linear (3,):", a_linear.detach())
            print("Layer1 ReLU (3,):", a_activated.detach())
            print("Layer1 sum -> scalar:", a_sum.detach().item())

            print("Layer2 linear (2,):", b_linear.detach())
            print("Layer2 Sigmoid (2,):", b_activated.detach())
            print("Layer2 sum -> scalar:", b_sum.detach().item())

            print("Combined sum (scalar):", combined.detach().item())
            print("Tanh(combined) (scalar):", combined_act.detach().item())
            print("Output linear (1,):", output.detach())

        # Return scalar tensor by squeezing the single output feature
        return output.squeeze(0)


def main() -> None:
    torch.manual_seed(42)

    # Create input as a scalar with gradient tracking
    x = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)

    model = TwoBranchComputationalGraph()

    # Forward pass with intermediate prints
    y = model(x, print_intermediates=True)
    print("Final output (scalar):", y.detach().item())

    # Backward pass to compute gradients of output w.r.t. input and parameters
    y.backward()

    print("\nGradients:")
    print("dOutput/dx:", x.grad.item())

    # List parameter gradients to verify autograd path
    for name, param in model.named_parameters():
        if param.grad is None:
            grad_info = None
        else:
            grad_info = param.grad
        print(f"{name} grad:\n{grad_info}\n")


if __name__ == "__main__":
    main()


