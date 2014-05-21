"""
Simple logging class with no module dependencies.

Use set_level(TRACE|DEBUG|INFO\WARN|ERROR) to indicate which level of log
messages should be printed. To generate log messages, use trace(string),
debug(string), info(string), etc.
"""

TRACE = 0
DEBUG = 1
INFO = 2
WARN = 3
ERROR = 4


__level__ = DEBUG
__level_strings__ = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"]


def set_level(level):
    global __level__
    __level__ = level


def __log__(msg):
    print("{}: {}".format(__level_strings__[__level__], msg))


def trace(msg):
    if __level__ <= TRACE:
        __log__(msg)


def debug(msg):
    if __level__ <= DEBUG:
        __log__(msg)


def info(msg):
    if __level__ <= INFO:
        __log__(msg)


def warn(msg):
    if __level__ <= WARN:
        __log__(msg)


def error(msg):
    if __level__ <= ERROR:
        __log__(msg)



