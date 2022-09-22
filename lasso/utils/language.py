
def get_var(name, context, default=None):
    ''' Get a variable from a dict context

    Parameters
    ----------
    name : str or list(str)
        variable name
    context : dict
        variable dict
    default:
        default value or function which creates the default
        value if failed to find var in context
    '''
    if not isinstance(name, (list, tuple)):
        name = (name,)

    current_context = context
    for current_name in name:
        if current_name in current_context:
            current_context = current_context[current_name]
        else:
            if callable(default):
                return default()
            else:
                return default

    return current_context


def set_var(name, value, context):
    ''' Set a viarble in a dict context

    Parameters
    ----------
    name : str or list(str)
        variable name or path as list
    value : object
        anything to set
    context : dict
        variable dict
    '''

    if not isinstance(name, (list, tuple)):
        name = (name,)

    current_context = context
    for i_name, current_name in enumerate(name):

        # at last level set var
        if i_name == len(name)-1:
            current_context[current_name] = value
        # otherwise iterate into next level
        else:
            if current_name in current_context:
                current_context = current_context[current_name]
            else:
                new_level = {}
                current_context[current_name] = new_level
                current_context = new_level
