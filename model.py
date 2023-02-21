import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


POOL_FUNS = {'max': mx.nd.max, 'mean': mx.nd.mean, 'sum': mx.nd.sum}
INF = 6.5e4
CACHE_SIZE = 2 ** 20

class DCN(gluon.Block):
    """Deep & Cross Network:

            x = dense(x)* x0 + x

    Parameters
    ----------
    input_dim: input dimension
        Should be equal to number of features
    num_layers : int, default is 2
        The number of DCN layers
    """
    def __init__(self, input_dim, num_layers=2, **kwargs):
        super(DCN, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Sequential()
            for i in range(num_layers):
                self.dense.add(nn.Dense(input_dim, use_bias=True))

    def forward(self, x):
        x1 = x
        for dense in self.dense:
            s = dense(x1)
            x1 = s*x + x1
        return x1


class MLP(gluon.Block):
    """Customized Multilayer Perceptron Network with DCN layers.
        The network is actually MLP followed by DCN.

    Parameters
    ----------
    input_dim: input dimension
        Should be equal to number of features
    dcn : int, default is 2
        The number of DCN layers
    hiddens: a list of [int]
        The numbers of hidden units of normal dense layers
    batchnorm: boolean, default is True
        Whether to add batch normalization layer at the input
    dropout: float, default is 0.2
        Add a dropout layer between DCN and MLP
    """
    def __init__(self, input_dim, output_dim=1, dcn=2, hiddens=[10], batchnorm=True, dropout=0.2, activation=None, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.mlp = nn.Sequential()
            if batchnorm:
                self.mlp.add(nn.BatchNorm())
            self.mlp.add(DCN(input_dim=input_dim, num_layers=dcn))
            self.mlp.add(nn.Dropout(dropout))
            for hidden in hiddens:
                self.mlp.add(nn.Dense(hidden, activation="relu"))
            if activation:
                self.dense = nn.Dense(output_dim, activation=activation)
            else:
                self.dense = nn.Dense(output_dim)

    def forward(self, x):
        """
        :param x: input feature
        :return: output score
        """
        x = self.mlp(x)
        return self.dense(x)
