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


def is_trace_enabled():
    return __level__ <= TRACE


def is_debug_enabled():
    return __level__ <= DEBUG


def is_info_enabled():
    return __level__ <= INFO


def is_warn_enabled():
    return __level__ <= WARN


def is_error_enabled():
    return __level__ <= ERROR


def trace(msg):
    if is_trace_enabled():
        __log__(msg)


def debug(msg):
    if is_debug_enabled():
        __log__(msg)


def info(msg):
    if is_info_enabled():
        __log__(msg)


def warn(msg):
    if is_warn_enabled():
        __log__(msg)


def error(msg):
    if is_error_enabled():
        __log__(msg)



