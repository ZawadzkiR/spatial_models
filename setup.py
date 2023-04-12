    from setuptools import setup, find_packages
     
    setup(name='spatial_models',
          version='1.0',
          author='ZawadzkiR',
          author_email='r.zawadzki96@gmail.com',
          license='For internal use',
          description='Extension of sklearn models to spatial data. Inspired by the SpatialML library https://github.com/cran/SpatialML',
          packages=['.spatial_models', 'spatial_models.forests'],
          package_data = {'.spatial_models': ['*'],'.spatial_models.forests': ['*'] },
          install_requires=['sklearn','scipy', 'numpy', 'pandas']
          )
