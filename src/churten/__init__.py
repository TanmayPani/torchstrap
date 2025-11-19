from beartype.claw import beartype_all, beartype_this_package
from beartype import BeartypeConf
beartype_this_package()
#beartype_all(conf=BeartypeConf(violation_type=UserWarning))

from . import optimizer
from . import utils
from . import stateless
from . import history
from . import callbacks


__all__ = [
    "optimizer",
    "utils",
    "stateless",
    "history",
    "callbacks",
]


def hello() -> str:
    return "Hello from churten!"
