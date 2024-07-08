from config.templates import encoder, layers
from config.templates.experiment import Experiment
from config.templates.dataset import Dataset

# ======================================================================================================================
# RegDB_C
# ======================================================================================================================

REGDB_A_ENCODERS_large = [
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_large"),
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_large"),
]
REGDB_A_DECODERS_large = [
    encoder.Encoder(input_size=None, layers="cnn_large_decoder"),
    encoder.Encoder(input_size=None, layers="cnn_large_decoder"),
]

REGDB_A_ENCODERS = [
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_small"),
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_small"),
]
REGDB_A_DECODERS = [
    encoder.Encoder(input_size=None, layers="cnn_small_decoder_"),
    encoder.Encoder(input_size=None, layers="cnn_small_decoder_"),
]


# ======================================================================================================================
# RegDB
# ======================================================================================================================
REGDB_ENCODERS_big_large = [
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_big_large"),
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_big_large"),
]
REGDB_DECODERS_big_large = [
    encoder.Encoder(input_size=None, layers="cnn_big_large_decoder"),
    encoder.Encoder(input_size=None, layers="cnn_big_large_decoder"),
]

REGDB_ENCODERS_large = [
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_large"),
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_large"),
]
REGDB_DECODERS_large = [
    encoder.Encoder(input_size=None, layers="cnn_large_decoder_"),
    encoder.Encoder(input_size=None, layers="cnn_large_decoder_"),
]

REGDB_ENCODERS = [
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_small"),
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_small"),
]
REGDB_DECODERS = [
    encoder.Encoder(input_size=None, layers="cnn_small_decoder_"),
    encoder.Encoder(input_size=None, layers="cnn_small_decoder_"),
]

REGDB_ENCODERS_tiny = [
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_tiny"),
    encoder.Encoder(input_size=(3, 128, 64), layers="cnn_tiny"),
]
REGDB_DECODERS_tiny = [
    encoder.Encoder(input_size=None, layers="cnn_tiny_decoder_"),
    encoder.Encoder(input_size=None, layers="cnn_tiny_decoder_"),
]

class REGDB_B_Experiment(Experiment):
    dataset_config: Dataset = Dataset(name="regdb_b")
    n_views: int = 2
    n_clusters: int = 50


class REGDB_A_Experiment(Experiment):
    dataset_config: Dataset = Dataset(name="regdb_a")
    n_views: int = 2
    n_clusters: int = 50

class REGDBExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="regdb")
    n_views: int = 2
    n_clusters: int = 206

# Default encoders for MNIST-like datasets
_MNIST_ENCODERS = [
    encoder.Encoder(input_size=(1, 28, 28), layers="cnn_small"),
    encoder.Encoder(input_size=(1, 28, 28), layers="cnn_small"),
]
_MNIST_DECODERS = [
    encoder.Encoder(input_size=None, layers="cnn_small_decoder"),
    encoder.Encoder(input_size=None, layers="cnn_small_decoder"),
]

# ======================================================================================================================
# Blobs (debugging dataset)
# ======================================================================================================================
BLOBS_ENCODERS = [
    encoder.Encoder(layers="dense_2d", input_size=(2,)),
    encoder.Encoder(layers="dense_2d", input_size=(2,)),
]
BLOBS_DECODERS = [
    encoder.Encoder(layers="dense_2d_decoder", input_size=None),
    encoder.Encoder(layers="dense_2d_decoder", input_size=None),
]


class BlobsExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="blobs_dep")
    n_views: int = 2
    n_clusters: int = 3
    n_runs: int = 1
    n_epochs: int = 20


# ======================================================================================================================
# Noisy MNIST
# ======================================================================================================================
NOISY_MNIST_ENCODERS = _MNIST_ENCODERS
NOISY_MNIST_DECODERS = _MNIST_DECODERS


class NoisyMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="noisymnist")
    n_views: int = 2
    n_clusters: int = 10

class NoisyMNISTExperiment_half(Experiment):
    dataset_config: Dataset = Dataset(name="noisymnist")
    n_views: int = 2
    n_clusters: int = 5

# ======================================================================================================================
# Noisy FashionMNIST
# ======================================================================================================================
NOISY_FASHIONMNIST_ENCODERS = _MNIST_ENCODERS
NOISY_FASHIONMNIST_DECODERS = _MNIST_DECODERS


class NoisyFashionMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="noisyfashionmnist")
    n_views: int = 2
    n_clusters: int = 10

class NoisyFashionMNISTExperiment_half(Experiment):
    dataset_config: Dataset = Dataset(name="noisyfashionmnist")
    n_views: int = 2
    n_clusters: int = 5
# ======================================================================================================================
# Edge MNIST
# ======================================================================================================================
EDGE_MNIST_ENCODERS = _MNIST_ENCODERS
EDGE_MNIST_DECODERS = _MNIST_DECODERS


class EdgeMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="edgemnist")
    n_views: int = 2
    n_clusters: int = 10


# ======================================================================================================================
# Edge FashionMNIST
# ======================================================================================================================
EDGE_FASHIONMNIST_ENCODERS = _MNIST_ENCODERS
EDGE_FASHIONMNIST_DECODERS = _MNIST_DECODERS


class EdgeFashionMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="edgefashionmnist")
    n_views: int = 2
    n_clusters: int = 10


# ======================================================================================================================
# Caltech101
# ======================================================================================================================

def _caltech_decoder(out_dim):
    return encoder.Encoder(layers=[
        layers.Dense(n_units=256),
        layers.BatchNormalization(),
        layers.RELU,
        layers.Dense(n_units=1024),
        layers.BatchNormalization(),
        layers.RELU,
        layers.Dense(n_units=1024),
        layers.BatchNormalization(),
        layers.RELU,
        layers.Dense(n_units=1024),
        layers.BatchNormalization(),
        layers.RELU,
        layers.Dense(n_units=out_dim, activation="sigmoid"),
    ])


CALTECH_ENCODERS = [
    encoder.Encoder(layers="dense_5", input_size=(48,)),
    encoder.Encoder(layers="dense_5", input_size=(40,)),
    encoder.Encoder(layers="dense_5", input_size=(254,)),
    encoder.Encoder(layers="dense_5", input_size=(1984,)),
    encoder.Encoder(layers="dense_5", input_size=(512,)),
    encoder.Encoder(layers="dense_5", input_size=(928,)),
]
CALTECH_DECODERS = [
    _caltech_decoder(48),
    _caltech_decoder(40),
    _caltech_decoder(254),
    _caltech_decoder(1984),
    _caltech_decoder(512),
    _caltech_decoder(928),
]


class Caltech7Experiment(Experiment):
    dataset_config: Dataset = Dataset(name="caltech7")
    n_views: int = 6
    n_clusters: int = 7


class Caltech20Experiment(Experiment):
    dataset_config: Dataset = Dataset(name="caltech20")
    n_views: int = 6
    n_clusters: int = 20


# ======================================================================================================================
# COIL-20
# ======================================================================================================================

COIL20_ENCODERS = [encoder.Encoder(input_size=(1, 64, 64), layers="cnn_large") for _ in range(3)]
COIL20_DECODERS = [encoder.Encoder(input_size=None, layers="cnn_large_decoder") for _ in range(3)]


class COIL20Experiment(Experiment):
    dataset_config: Dataset = Dataset(name="coil20")
    n_views: int = 3
    n_clusters: int = 20


# ======================================================================================================================
# PatchedMNIST
# ======================================================================================================================

# PATCHED_MNIST_ENCODERS = _MNIST_ENCODERS
# PATCHED_MNIST_DECODERS = _MNIST_DECODERS

PATCHED_MNIST_ENCODERS = [
    encoder.Encoder(input_size=(1, 28, 28), layers="cnn_small"),
    encoder.Encoder(input_size=(1, 28, 28), layers="cnn_small"),
    encoder.Encoder(input_size=(1, 28, 28), layers="cnn_small"),

]
PATCHED_MNIST_DECODERS = [
    encoder.Encoder(input_size=None, layers="cnn_small_decoder"),
    encoder.Encoder(input_size=None, layers="cnn_small_decoder"),
    encoder.Encoder(input_size=None, layers="cnn_small_decoder"),
]

class PatchedMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="patchedmnist")
    n_views: int = 3  # 12
    n_clusters: int = 3
