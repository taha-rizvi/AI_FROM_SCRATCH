from linear_layer import Linear
class Module:
    def __init__(self):
        self._submodules={}
    
    def __setattr__(self, name, value):
        # ONLY call super() AFTER checking - avoid recursion!
        if hasattr(value, 'parameters') and callable(value.parameters):
            if not hasattr(self, '_submodules'):
                self._submodules = {}
            self._submodules[name] = value
        super(Module, self).__setattr__(name, value)
    def parameters(self):
        params = []
        for submodule in getattr(self, '_submodules', {}).values():
            params.extend(submodule.parameters())
        return params