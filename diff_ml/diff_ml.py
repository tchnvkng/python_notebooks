import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Approximator(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        learning_rate=0.001,
        layer_sizes=[],
        activation=nn.Softplus(beta=10.0),
    ):
        super().__init__()

        self.activation = activation
        self.layer_sizes = layer_sizes
        all_sizes = [n_inputs] + layer_sizes
        self.layers = np.ravel(
            [
                [nn.Linear(all_sizes[i], all_sizes[i + 1]), self.activation]
                for i in range(len(all_sizes) - 1)
            ]
        ).tolist() + [nn.Linear(all_sizes[-1], 1)]
        self.layers = nn.Sequential(*self.layers)

        self.learning_rate_ = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate_)

    @property
    def learning_rate(self):
        return self.learning_rate_

    @learning_rate.setter
    def learning_rate(self, value):
        self.learning_rate_ = value
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate_)

    def forward(self, x):
        return self.layers(x)

    def train(self, n_epochs, x_train, y_train, dy_train, lambda_grad=0.7):
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            x_train.requires_grad_(True)
            y_pred = self(x_train)
            grad_outputs = torch.ones_like(y_pred)
            grad_pred = torch.autograd.grad(
                outputs=y_pred,
                inputs=x_train,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # Compute loss: MSE on value and gradient
            loss_val = nn.functional.mse_loss(y_pred, y_train)
            loss_grad = nn.functional.mse_loss(grad_pred, dy_train)
            loss = (1 - lambda_grad) * loss_val + lambda_grad * loss_grad

            loss.backward()
            self.optimizer.step()
            if epoch % 200 == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss.item():.6f}, Value Loss: {loss_val.item():.6f}, Grad Loss: {loss_grad.item():.6f}"
                )

    def predict(self, x_test):
        x_test.requires_grad_(True)
        y_pred = self(x_test)
        grad_pred_test = torch.autograd.grad(
            outputs=y_pred,
            inputs=x_test,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        return (
            y_pred.cpu().detach().numpy().squeeze(),
            grad_pred_test.cpu().detach().numpy().squeeze(),
        )
