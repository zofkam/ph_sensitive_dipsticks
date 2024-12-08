#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logger = logging.root

DEFAULT_FORMATTER = '%(asctime)s - %(levelname)s - %(message)s'

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

FILE_HANDLER = logging.FileHandler
SCREEN_HANDLER = logging.StreamHandler


class Logger(object):
    """
    The Logger class is a simple wrapper around the standard `logging` module.
    It allows you to set up different types of logging handlers (e.g. FileHandler, StreamHandler, ...) and offers
     the same logging levels and the same corresponding methods as the standard `logging` module: debug, info, warning,
     exception, error and critical.

    """

    def __init__(self, name, handlers=None):
        """
        Initializes the logger by name, sets the minimal logging level and initializes requested handlers.

        :param name: name to use for the logger
        :type name: str
        :param handlers: is a list of dictionaries with information about the handler, the keys are:
                          - handler_class: handler class, either `FILE_HANDLER` or `SCREEN_HANDLER` (mandatory)
                          - formatter: formatting string to use for log entries (mandatory)
                          - level: logging level to use (optional, if not specified, the logger level (INFO) is used)
                          - filename: name of the log file with absolute path
                                       (optional - only for handler_class=FILE_HANDLER)
                          - encoding: encoding to use for the log file (optional - only for handler_class=FILE_HANDLER)
        :type handlers: list

        """
        # initialize the logger or returns an existing logger with the given name
        self._logger = logging.getLogger(name=name)
        # set logging level
        self._logger.setLevel(level=DEBUG)

        for handler in handlers:
            self.add_handler(handler=handler)

    def is_logging_enabled(self, handler_class):
        """
        Returns an indicator if the given handler is already active.

        :param handler_class: handler class (`FILE_HANDLER` or `SCREEN_HANDLER`)
        :type handler_class: class

        :return: indicator
        :rtype: bool

        """
        return any(isinstance(logger_handler, handler_class) for logger_handler in self._logger.handlers[:])

    def add_handler(self, handler):
        """
        Adds a logger handler.

        :param handler: logging handler options
        :type handler: dict

        """
        if not self.is_logging_enabled(handler_class=handler['handler_class']):
            self._logger.addHandler(hdlr=self._initialize_handler(handler=handler))

    def close(self):
        """
        Closes all open log handlers.

        """
        if self._logger:
            handlers = self._logger.handlers[:]

            for handler in handlers:
                handler.close()
                self._logger.removeHandler(handler)

    def _initialize_handler(self, handler):
        """
        Initialize a handler.

        :param handler: handler class
        :type handler: class

        :return: initialized handler object
        :rtype: Handler

        """
        # initialize the handler
        console_handler = handler['handler_class'](**{param: handler[param]
                                                      for param in ['filename', 'encoding'] if param in handler.keys()})

        # set the handler logging level
        console_handler.setLevel(level=handler['level'] if 'level' in handler.keys() else self._get_effective_level())

        # set handler formatting string
        console_handler.setFormatter(logging.Formatter(handler['formatter']))

        return console_handler

    def _get_effective_level(self):
        """
        Returns the currently set logging level for the logger.

        :return: ID of the level
        :rtype int

        """
        return self._logger.getEffectiveLevel()

    def debug(self, msg):
        """
        Log debug info.

        :param msg: message to log
        :type msg: Any

        """
        self._logger.debug(msg=msg)

    def info(self, msg):
        """
        Log info.

        :param msg: message to log
        :type msg: Any

        """
        self._logger.info(msg=msg)

    def warning(self, msg):
        """
        Log a warning.

        :param msg: message to log
        :type msg: Any

        """
        self._logger.warning(msg=msg)

    def error(self, msg):
        """
        Log an error.

        :param msg: message to log
        :type msg: Any

        """
        self._logger.error(msg=msg)

    def exception(self, msg):
        """
        Log an exception.

        :param msg: message to log
        :type msg: Any

        """
        self._logger.exception(msg=msg)

    def critical(self, msg):
        """
        Log a critical error.

        :param msg: message to log
        :type msg: Any

        """
        self._logger.critical(msg=msg)
