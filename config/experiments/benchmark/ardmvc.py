from config.experiments.base_experiments import *
from config.templates.models.simvc_comvc import CoMVC


noisymnist_ardmvc_am = NoisyMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
    ),
    batch_size=100,
)

noisyfashionmnist_ardmvc_am = NoisyFashionMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

patchedmnist_ardmvc_am = PatchedMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)

# no_kl
noisymnist_ardmvc = NoisyMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
    ),
    batch_size=100,
)

noisyfashionmnist_ardmvc  = NoisyFashionMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

patchedmnist_ardmvc  = PatchedMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)