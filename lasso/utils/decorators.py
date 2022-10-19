import inspect


# Source: https://www.fast.ai/2019/08/06/delegation/


def delegates(to_function=None, keep=False):
    """Decorator: replace `**kwargs` in signature with params from `to`"""

    def _f(f_att):
        if to_function is None:
            to_f, from_f = f_att.__base__.__init__, f_att.__init__
        else:
            to_f, from_f = to_function, f_att
        sig = inspect.signature(from_f)
        sig_dict = dict(sig.parameters)
        k = sig_dict.pop("kwargs")
        s2_dict = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sig_dict
        }
        sig_dict.update(s2_dict)
        if keep:
            sig_dict["kwargs"] = k
        # noinspection PyTypeChecker
        from_f.__signature__ = sig.replace(parameters=sig_dict.values())
        return f_att

    return _f


def custom_dir(custom_c, add):
    """Custom dir *add description?"""
    return dir(type(custom_c)) + list(custom_c.__dict__.keys()) + add


class GetAttr:

    """
    Base class for attr accesses in `self._xtra` passed down to `self.default
    """

    @property
    def _xtra(self):
        return [o for o in dir(self.default) if not o.startswith("_")]

    def __getattr__(self, k):
        if k in self._xtra:
            return getattr(self.default, k)
        raise AttributeError(k)

    def __dir__(self):
        return custom_dir(self, self._xtra)
