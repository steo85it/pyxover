import setuptools

setuptools.setup(
    name='pyxover',
    version='0.0.1',
    author='Stefano Bertone',
    # author_email='you@yourdomain.com',
    description='Python package to process altimetry crossovers for planetary geodesy',
    platforms='Posix; MacOS X; Windows',
    # packages=setuptools.find_packages(where='src'),
    #packages=['src'],
    package_dir={
        '': 'src',
    },
    include_package_data=True,
    install_requires=(
        'numpy',
    ),
    setup_requires=(
        'pytest-runner',
    ),
    tests_require=(
        'pytest-cov',
    ),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
