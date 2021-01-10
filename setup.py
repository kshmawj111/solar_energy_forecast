from setuptools import setup, find_packages

setup(
    name             = 'solar_energy_forecast',
    version          = '1.0',
    description      = 'Python wrapper for liquibase',
    url              = 'https://github.com/kshmawj111/solar_energy_forcast',
    install_requires = [
                        'gluonts',
                        'mxnet',
                        'pandas',
                        'numpy',
                        'bayesian-optimization'

    ],
    packages = find_packages(),
    python_requires  = '>=3.7',
    zip_safe=False,
)