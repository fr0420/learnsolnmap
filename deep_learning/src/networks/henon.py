import torch
from torch import nn
from networks.basics import MLP


def get_activation_derivative(activation):
    if isinstance(activation, nn.Tanh):
        return lambda x: 1 - torch.tanh(x)**2
    elif isinstance(activation, nn.ELU):
        return lambda x: (x<=0) * activation.alpha * torch.exp(x) + (x>0) * 1.
    elif isinstance(activation, nn.ReLU):
        return lambda x: (x<=0) * 0. + (x>0) * 1.
    elif isinstance(activation, nn.LeakyReLU):
        return lambda x: (x<=0) * activation.negative_slope + (x>0) * 1.
    elif isinstance(activation, nn.Sigmoid):
        return lambda x: torch.sigmoid(x) - torch.sigmoid(x)**2
    else:
        return None


def calc_jacobian(inputs, net, act_deriv=None):
    """Calculate jacobian of a feedforward network with one hidden layer w.r.t. inputs.
    
    y = W2 * a
    a = activation(z)
    z = W1 * x + b1 
    
    -> dydx = dyda * dadz * dzdx 
            = W2 * dadz * W1
    """
    # inputs shape: (bs, n_in)
    # net(inputs) shape: (bs, n_out)
    # net: nn.Sequential(nn.Linear(n_in, n_hidden), activation, nn.Linear(n_hidden, n_out))

    layer1, act, layer2 = net[0], net[1], net[2]
    z = layer1(inputs)  # (bs, n_hidden)
    
    if act_deriv is not None:
        dadz = act_deriv(z)  # (bs, n_hidden)
        dadz = torch.diag_embed(dadz)  # (bs, n_hidden, n_hidden)
    else:  # compute using autodiff, about 10 times slower than explicit computation
        dadz = torch.func.vmap(torch.func.jacrev(act))(z)  # (bs, n_hidden, n_hidden)

    jac = torch.matmul(layer2.weight, dadz)  # (n_out, n_hidden) @ (bs, n_hidden, n_hidden) -> (bs, n_out, n_hidden)
    jac = torch.matmul(jac, layer1.weight)  # (bs, n_out, n_hidden) @ (n_hidden, n_in) -> (bs, n_out, n_in)
    return jac  # (bs, n_out, n_in)


def calc_jacobian_autodiff(inputs, net):
    """Calculate jacobian of a feedforward network w.r.t. inputs using automatic differentiation."""
    # inputs shape: (bs, n_in)
    # net(inputs) shape: (bs, n_out)
    jac = torch.func.jacrev(net)(inputs)  # (bs, n_out, bs, n_in)
    jac = torch.diagonal(jac, dim1=0, dim2=2)  # (n_out, n_in, bs)
    jac = torch.permute(jac, (2, 0, 1))  # (bs, n_out, n_in)
    return jac
    

class HenonMap(nn.Module):
    
    def __init__(self, dof, hidden_dim, activation, n_hidden_layers=1, epsilon=1.0):
        super(HenonMap, self).__init__()
        self.shift = nn.Parameter(torch.zeros(dof), requires_grad=True)
        # self.potential = MLP([dof]+[hidden_dim]*n_hidden_layers+[1], activation)
        self.potential = nn.Sequential(
            nn.Linear(dof, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1, bias=False)
        )  # (bs, dof) -> (bs, 1)
        self.act_deriv = get_activation_derivative(activation)
        if self.act_deriv is None:
            print(f"Derivative function for {activation} is undefined. \
                  Autodiff will be used instead, which will slow down jacobian computation.")
        self.epsilon = epsilon

    def forward(self, p, q, inverse_mode=False):
        # p shape: (bs, dof)
        # q shape: (bs, dof)

        if inverse_mode:
            # grad_V = calc_jacobian_autodiff(q-self.shift, self.potential)  # (bs, 1, dof) <-- too slow!
            grad_V = calc_jacobian(q-self.shift, self.potential, self.act_deriv)  # (bs, 1, dof)
        
            grad_V = grad_V.squeeze(dim=-2)  # (bs, dof)
            p_new = q - self.shift 
            q_new = -p + self.epsilon * grad_V
        else:
            # grad_V = calc_jacobian_autodiff(p, self.potential)  # (bs, 1, dof) <-- too slow!
            grad_V = calc_jacobian(p, self.potential, self.act_deriv)  # (bs, 1, dof)
            
            grad_V = grad_V.squeeze(dim=-2)  # (bs, dof)
            p_new = -q + self.epsilon * grad_V
            q_new = p + self.shift

        return p_new, q_new


class HenonLayer(nn.Module):
    
    def __init__(self, dof, hidden_dim, activation, n_hidden_layers=1, epsilon=1.0):
        super(HenonLayer, self).__init__()
        self.henon_map = HenonMap(dof, hidden_dim, activation, n_hidden_layers, epsilon)

    def forward(self, u, inverse_mode=False):
        p, q = u.chunk(2, dim=-1)
        for _ in range(4):
            p, q = self.henon_map(p, q, inverse_mode)
        return torch.cat((p, q), dim=-1)


class HenonNet(nn.Module):
    
    def __init__(self, dof, hidden_dim, n_henon_layers, activation, n_hidden_layers=1, epsilon=1.0):
        super(HenonNet, self).__init__()
        self.henon_layers = nn.ModuleList(
            [HenonLayer(dof, hidden_dim, activation, n_hidden_layers, epsilon) for _ in range(n_henon_layers)]
        )

    def forward(self, u, inverse_mode=False):
        if inverse_mode:
            for layer in self.henon_layers[::-1]:
                u = layer(u, inverse_mode)
        else:
            for layer in self.henon_layers:
                u = layer(u, inverse_mode)
        return u


class SymplecticGyroceptron(nn.Module):

    def __init__(self, dof, NIHenon_hidden_dim, NIHenon_n_henon_layers, 
                 Henon_hidden_dim, Henon_n_henon_layers, activation, epsilon, 
                 NIHenon_n_hidden_layers=1, Henon_n_hidden_layers=1):
        super(SymplecticGyroceptron, self).__init__()

        # Define the near-identity Henon net
        self.NIHenon = HenonNet(dof, NIHenon_hidden_dim, NIHenon_n_henon_layers, activation, 
                                NIHenon_n_hidden_layers, epsilon)
        
        # Define the Henon net
        self.Henon = HenonNet(dof, Henon_hidden_dim, Henon_n_henon_layers, activation, 
                              Henon_n_hidden_layers, 1.0)
        
        # Define the circle action layer
        self.CALayer = CircleActionLayer()

    def forward(self, u):
        u = self.Henon(u, inverse_mode=True)
        u = self.CALayer(u)
        u = self.Henon(u)
        u = self.NIHenon(u)
        return u


class CircleActionLayer(nn.Module):
    
    def __init__(self):
        super(CircleActionLayer, self).__init__()
        self.theta = nn.Parameter(torch.randn(1), requires_grad=True)
    
    def forward(self, u):
        assert u.shape[-1] == 4, "Input must have shape (..., 4)"
        p1, p2, q1, q2 = u[..., (0,)], u[..., (1,)], u[..., (2,)], u[..., (3,)]

        # Rotate (q1, p1) in phase space by theta
        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)
        new_q1 = cos_theta * q1 + sin_theta * p1
        new_p1 = -sin_theta * q1 + cos_theta * p1
        
        return torch.cat((new_p1, p2, new_q1, q2), dim=-1)
    