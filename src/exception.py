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

    def __init__(self, error_message, sys_error_details = None):
        super().__init__(error_message)

        if sys_error_details is None:
            _, exc, tb = sys.exc_info()
        elif isinstance(sys_error_details, Exception):
            tb = sys_error_details.__traceback__
            while tb and tb.tb_next:
                tb = tb.tb_next
        else:
            tb = None

        self.lineno = tb.tb_lineno if tb else "NA"
        self.filename = tb.tb_frame.f_code.co_filename if tb else "NA"

    def __str__(self):
        """
        Return a readable string representation of the exception.

        Format:
            filename:lineno | message

        Returns:
           str: readable string representation of the exception
        """
        return f"{self.filename}:{self.lineno} | {self.args[0]}"
