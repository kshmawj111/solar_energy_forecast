from setuptools import setup, find_packages

setup(
    name             = 'solar_energy_forecast',
    version          = '1.0',
    description      = 'Python wrapper for liquibase',
    author           = 'Won Jun Kim',
    author_email     = 'glistering96@gmail.com',
    url              = 'https://github.com/rampart81/pyquibase',
    download_url     = 'https://githur.com/rampart81/pyquibase/archive/1.0.tar.gz',
    install_requires = [ ],
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['liquibase', 'db migration'],
    python_requires  = '>=3',
    package_data     =  {
        'pyquibase' : [
            'db-connectors/sqlite-jdbc-3.18.0.jar',
            'db-connectors/mysql-connector-java-5.1.42-bin.jar',
            'liquibase/liquibase.jar'
    ]},
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ]
)