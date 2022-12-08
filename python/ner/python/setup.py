from typing import List

from setuptools import setup, find_packages
import os

from itertools import chain


def get_dependency_from_requirements(filename, folder="requirements") -> List[str]:
    with open(os.path.join(folder, filename)) as f:
        content = f.readlines()
    return [x.strip() for x in content if not x.startswith("#")]


install_requires = get_dependency_from_requirements("requirements.txt")

# extras_require = {
#     "analyze": get_dependency_from_requirements("requirements_analyze.txt"),
#     "train": get_dependency_from_requirements("requirements_train.txt"),
# }

# extras_require['all'] = list(chain(extras_require.values()))
#
# print(extras_require)

dependency_links = [
    "https://download.pytorch.org/whl/cu113/torch_stable.html"
]


setup(
    name="lab_ner",
    version="0.0.1",
    description="개체명 인식 모델",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    zip_safe=False,
    install_requires=install_requires,
    # extras_require=extras_require,
)
