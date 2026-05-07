"""
For Ref: https://stackoverflow.com/questions/42260912/how-to-get-filename-of-second-item-deep-in-exception-traceback
"""

import sys


class ApplicationException(Exception):
    """
    Custom exception that gives error messages with contextual debugging information such as the file name and line number where the original exception occurred.

    Attributes:
        - lineno (int | str): Line number where the exception was raised
        - filename (str): Name of the file where the exception occurred

    Args:
        error_message (_type_): readable error message
        sys_error_details (Exception | None, optional): Original exception instance. Defaults to None.
    """

    def __init__(self, error_message, sys_error_details: Exception | None = None):
        super().__init__(error_message)

        traceback = sys_error_details.__traceback__ if sys_error_details else None

        self.lineno = traceback.tb_lineno if traceback else "NA"
        self.filename = traceback.tb_frame.f_code.co_filename if traceback else "NA"

    def __str__(self):
        """
        Return a readable string representation of the exception.

        Format:
            filename:lineno | message

        Returns:
           str: readable string representation of the exception
        """
        return f"{self.filename}:{self.lineno} | {self.args[0]}"
