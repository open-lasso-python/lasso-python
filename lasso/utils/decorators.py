
import inspect

# Source: https://www.fast.ai/2019/08/06/delegation/

def delegates(to=None, keep=False):
    """ Decorator: replace `**kwargs` in signature with params from `to`"""

    def _f(f):
        if to is None: to_f,from_f = f.__base__.__init__,f.__init__
        else:          to_f,from_f = to,f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f


def custom_dir(c, add): return dir(type(c)) + list(c.__dict__.keys()) + add

class GetAttr:
    "Base class for attr accesses in `self._xtra` passed down to `self.default`"
    @property
    def _xtra(self): return [o for o in dir(self.default) if not o.startswith('_')]
    def __getattr__(self,k):
        if k in self._xtra: return getattr(self.default, k)
        raise AttributeError(k)
    def __dir__(self): return custom_dir(self, self._xtra)

