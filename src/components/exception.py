# src/exception.py
import sys
import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        error_message = str(error_message)
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.line_number = exc_tb.tb_lineno
        else:
            self.file_name = "<unknown>"
            self.line_number = -1
        self.error_message = error_message

    def __str__(self):
        # Red color for error message
        return (f"\033[91mFile: {self.file_name}\n"
                f"Line: {self.line_number}\n"
                f"Message: {self.error_message}\033[0m")

# Example usage:
# try:
#     ...
# except Exception as e:
#     raise CustomException(str(e), sys)
