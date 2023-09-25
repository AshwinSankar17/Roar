import codecs
import importlib.util
import os
import subprocess
from distutils import cmd as distutils_cmd
from distutils import log as distutils_log
from itertools import chain

import setuptools

spec = importlib.util.spec_from_file_location("package_info", "roar/package_info.py")
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)


__contact_emails__ = package_info.__contact_emails__
__contact_names__ = package_info.__contact_names__
__description__ = package_info.__description__
__keywords__ = package_info.__keywords__
__license__ = package_info.__license__
__package_name__ = package_info.__package_name__
__repository_url__ = package_info.__repository_url__
__version__ = package_info.__version__


if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    long_description_content_type = "text/markdown"


def req_file(filename):
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")


class StyleCommand(distutils_cmd.Command):
    __ISORT_BASE = "isort"
    __BLACK_BASE = "black"
    description = "Checks overall project code style."
    user_options = [
        ("scope=", None, "Folder of file to operate within."),
        ("fix", None, "True if tries to fix issues in-place."),
    ]

    def __call_checker(self, base_command, scope, check):
        command = list(base_command)

        command.append(scope)

        if check:
            command.extend(["--check", "--diff"])

        self.announce(
            msg="Running command: %s" % str(" ".join(command)),
            level=distutils_log.INFO,
        )

        return_code = subprocess.call(command)

        return return_code

    def _isort(self, scope, check):
        return self.__call_checker(
            base_command=self.__ISORT_BASE.split(),
            scope=scope,
            check=check,
        )

    def _black(self, scope, check):
        return self.__call_checker(
            base_command=self.__BLACK_BASE.split(),
            scope=scope,
            check=check,
        )

    def _pass(self):
        self.announce(msg="\033[32mPASS\x1b[0m", level=distutils_log.INFO)

    def _fail(self):
        self.announce(msg="\033[31mFAIL\x1b[0m", level=distutils_log.INFO)

    # noinspection PyAttributeOutsideInit
    def initialize_options(self):
        self.scope = "."
        self.fix = ""

    def run(self):
        scope, check = self.scope, not self.fix
        isort_return = self._isort(scope=scope, check=check)
        black_return = self._black(scope=scope, check=check)

        if isort_return == 0 and black_return == 0:
            self._pass()
        else:
            self._fail()
            exit(isort_return if isort_return != 0 else black_return)

    def finalize_options(self):
        pass


setuptools.setup(
    name=__package_name__,
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url=__repository_url__,
    author=", ".join(__contact_names__),  # type: ignore
    author_email=", ".join(__contact_emails__),  # type: ignore
    license=__license__,
    maintainer=", ".join(__contact_names__),  # type: ignore
    maintainer_email=", ".join(__contact_emails__),  # type: ignore
    classifiers=[
        #  1 - Planning
        #  2 - Pre-Alpha
        #  3 - Alpha
        #  4 - Beta
        #  5 - Production/Stable
        #  6 - Mature
        #  7 - Inactive
        "Development Status :: 3 - Alpha",
        # Who is this toolkit for?
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # Project Domain
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Speech Synthesis",
        "Topic :: Scientific/Engineering :: Speech Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        # License
        "License :: OSI Approved :: Apache Software License",
        # Supported Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        # Additional Settings
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    dependency_links=["https://download.pytorch.org/whl/cu118"],
    python_requires=">=3.10",
    include_package_data=True,
    exclude=["tools", "tests"],
    package_data={"": ["*.txt", "*.md", "*.rst"]},
    zip_safe=False,
    keywords=__keywords__,
    cmdclass={"style": StyleCommand},
)
