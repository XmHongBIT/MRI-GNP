try:
    from setuptools import setup, find_namespace_packages
except ImportError:
    raise ImportError(
        '\n\nSome errors occurred when importing "setuptools". Maybe you should '
        'upgrade "setuptools" and "pip" using \n"pip install -U setuptools"\n"pip install -U pip".\n'
        'When "setuptools" and "pip" is updated successfully, try to install this package again :-).\n\n')

#script_dir = sys.path[0]
info_dict = {}
exec(open('digicare/pkginfo.py').read(), info_dict)

setup(
    name='digicare',
    packages=find_namespace_packages(include=["digicare", "digicare.*"]),
    version=info_dict['__version__'],
    
    description='',
    
    author=info_dict['__author__'],
    author_email='chenghao1652@126.com, 3120215380@bit.edu.cn',
    
    # need Python > 3.5, because os.mkdir(...) is not thread-safe before this version
    # Python 3.7 just works fine.

    #
    # NOTE: these package settings are only tested on Ubuntu-16.04
    #
    python_requires='>=3.7.1', 
    install_requires=[
        'numpy==1.20.3',
        'nibabel==3.2.1',
        'scikit-image==0.18.1',
        'scipy==1.7.3',
        'matplotlib==3.4.2',
        'psutil==5.8.0',
        'imageio==2.13.2',
        'openpyxl==3.0.9', # read xlsx
        'xlsxwriter==1.4.3', # write xlsx
        #'pingouin==0.5.1', # for correlation analysis
        'statsmodels==0.13.2',
        'reportlab==3.6.9',
        'svglib==1.2.1',
        'sklearn'
    ],
    entry_points={
        'console_scripts':[
        ]
    },
    keywords=[''
    ]
)
