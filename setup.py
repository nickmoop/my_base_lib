from distutils.core import setup

setup(
    name='base_lib',
    version='0.1',
    description='Python data transformation and visualization utilites',
    author='Nick',
    author_email='wsaad@mail.ru',
    url='https://github.com/nickmoop/my_base_lib/',
    packages=['base_lib'],
    install_requires = [
        'scikit-learn',
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib'
    ]
)
