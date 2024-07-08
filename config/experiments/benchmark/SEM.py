from config.experiments.base_experiments import *  # encoder/decoder dataset_config
from config.templates.models.simvc_comvc import CoMVC  # model framework

regdb_a_SEM = REGDB_A_Experiment(
    model_config=CoMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)

regdb_SEM = REGDBExperiment(
    model_config=CoMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=32,
)

noisymnist_SEM = NoisyMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)

noisyfashionmnist_SEM = NoisyFashionMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

patchedmnist_SEM = PatchedMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)
