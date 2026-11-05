from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop

package_name = "aic_ros_recorder"


class ColconCompatibleDevelop(_develop):
    """Accept legacy colcon's extra --editable flag during setup.py develop."""

    user_options = _develop.user_options + [
        ("editable", None, "Ignored compatibility flag passed by colcon"),
        ("script-dir=", None, "ROS compatibility option from setup.cfg"),
    ]
    boolean_options = list(getattr(_develop, "boolean_options", [])) + ["editable"]

    def initialize_options(self):
        super().initialize_options()
        self.editable = None
        self.script_dir = None


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/aic_ros_recorder"]),
        ("share/aic_ros_recorder", ["package.xml"]),
    ],
    install_requires=["setuptools", "lerobot", "pyyaml"],
    zip_safe=True,
    maintainer="enrique",
    maintainer_email="enrique@intrinsic.ai",
    description="AIC ROS recorder: bag + lerobot parquet",
    license="Apache-2.0",
    tests_require=[],
    cmdclass={
        "develop": ColconCompatibleDevelop,
    },
    entry_points={
        "console_scripts": [
            "aic_ros_recorder_node = aic_ros_recorder.recorder_node:main",
            "aic_ros_bag_to_parquet = aic_ros_recorder.bag_to_parquet:main",
            "aic_ros_postprocess_scoring = aic_ros_recorder.postprocess_scoring:main",
            "aic_ros_recover_scoring = aic_ros_recorder.recover_scoring_from_bags:main",
        ],
    },
)
