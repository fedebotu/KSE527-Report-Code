import torch; import torch.nn as nn
from torch import sin, pow
from torch.nn import Parameter
from torch.distributions.exponential import Exponential
import numpy as np


class Snake(nn.Module):
    '''         
    Implementation of the serpentine-like sine-based periodic activation function
    
    .. math::
         Snake_a := beta x + \frac{1}{a} sin^2(ax) = x - \frac{1}{2a}cos{2ax} + \frac{1}{2a}
    
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
        - We implement the slope parameter inspired by this:
        https://arxiv.org/abs/2112.01579
        Be careful though: although the authors claim its monotonicity, it does not seem so.
        
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, a=None, beta=1, trainable=True):
        '''
        Initialization.
        Args:
            in_features: shape of the input
            a: trainable parameter
            beta: slope parameter. Defaults to 1, 0.5 may be better according to https://arxiv.org/abs/2112.01579. Not learnable for now
            trainable: sets `a` as a trainable parameter
            
            `a` is initialized to 1 by default, higher values = higher-frequency, 
            5-50 is a good starting point if you already think your data is periodic, 
            consider starting lower e.g. 0.5 if you think not, but don't worry, 
            `a` will be trained along with the rest of your model. 
        '''
        super(Snake,self).__init__()
        self.in_features = in_features if isinstance(in_features, list) or isinstance(in_features, tuple) else [in_features]

        # Initialize `a`
        if a is not None:
            self.a = Parameter(torch.ones(self.in_features) * a) # create a tensor out of alpha
        else:            
            m = Exponential(torch.tensor([0.1]))
            self.a = Parameter((m.rsample(self.in_features)).squeeze()) # random init = mix of frequencies

        self.a.requiresGrad = trainable # Set the training to true
        self.beta = beta # slope is not trainable for now, we may do an extension though

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= beta*x + 1/a* sin^2 (xa)
        '''
        return  self.beta*x + (1.0/self.a) * pow(sin(x * self.a), 2)