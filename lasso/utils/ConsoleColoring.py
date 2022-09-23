
import subprocess


class ConsoleColoring:
    ''' Holds coloring escape sequences for command line shells
    '''

    # text coloring
    LIGHT_GREEN = '\033[92m'
    LIGHT_RED = '\033[91m'
    LIGHT_CYAN = '\033[96m'
    LIGHT_BLUE = '\033[94m'
    LIGHT_PURPLE = '\033[95m'
    LIGHT_YELLOW = '\033[93m'

    PURPLE = '\033[95m'
    RED = '\033[91m'
    GREEN = '\u001b[32m'
    CYAN = '\u001b[36m'
    WHITE = '\u001b[37m'
    BLACK = '\u001b[30m'
    BLUE = '\u001b[34m'
    ORANGE = '\u001b[33m'

    # special stuff
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    REVERSED = '\u001b[7m'

    # ends coloring
    RESET = '\033[0m'

    @staticmethod
    def purple(msg, light=False):
        ''' Format a string in purple

        Parameters
        ----------
        msg : `str`
            string to format
        light : `bool`
            whether to use light coloring

        Returns
        -------
        formatted_msg : `str`
            string colored for console output
        '''
        if light:
            return ConsoleColoring.LIGHT_PURPLE + msg + ConsoleColoring.RESET
        else:
            return ConsoleColoring.PURPLE + msg + ConsoleColoring.RESET

    @staticmethod
    def yellow(msg, light=False):
        ''' Format a string in yellow

        Parameters
        ----------
        msg : `str`
            string to format
        light : `bool`
            whether to use light coloring

        Returns
        -------
        formatted_msg : `str`
            string colored for console output
        '''
        if light:
            return ConsoleColoring.LIGHT_YELLOW + msg + ConsoleColoring.RESET
        else:
            return ConsoleColoring.ORANGE + msg + ConsoleColoring.RESET

    @staticmethod
    def red(msg, light=False):
        ''' Format a string in red

        Parameters
        ----------
        msg : `str`
            string to format
        light : `bool`
            whether to use light coloring

        Returns
        -------
        formatted_msg : `str`
            string colored for console output
        '''
        if light:
            return ConsoleColoring.LIGHT_RED + msg + ConsoleColoring.RESET
        else:
            return ConsoleColoring.RED + msg + ConsoleColoring.RESET

    @staticmethod
    def green(msg, light=False):
        ''' Format a string in green

        Parameters
        ----------
        msg : `str`
            string to format
        light : `bool`
            whether to use light coloring

        Returns
        -------
        formatted_msg : `str`
            string colored for console output
        '''
        if light:
            return ConsoleColoring.LIGHT_GREEN + msg + ConsoleColoring.RESET
        else:
            return ConsoleColoring.GREEN + msg + ConsoleColoring.RESET

    @staticmethod
    def blue(msg, light=False):
        ''' Format a string in green

        Parameters
        ----------
        msg : `str`
            string to format
        light : `bool`
            whether to use light coloring

        Returns
        -------
        formatted_msg : `str`
            string colored for console output
        '''
        if light:
            return ConsoleColoring.LIGHT_BLUE + msg + ConsoleColoring.RESET
        else:
            return ConsoleColoring.BLUE + msg + ConsoleColoring.RESET
