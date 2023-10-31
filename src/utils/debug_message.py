import inspect

def Assert(condition, message):
    if not condition:
        message = message + " [{function_name}]"
        current_function_name = inspect.currentframe().f_back.f_code.co_name
        raise AssertionError(message.format(function_name = current_function_name))

