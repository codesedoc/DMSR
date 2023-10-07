# Singleton Pattern
def singleton(_class: type):
    class_names = {}
    name = repr(_class)

    def wrapped_class(*args, **kwargs):
        if name in class_names.keys():
            instance = class_names[name]
        else:
            instance = _class(*args, **kwargs)
            class_names[name] = instance
        return instance
    return wrapped_class


def init_argument_decorator(_decorator):
    def _decorator_wrap(cls):
        def _init_decorator(instance, *args, **kwargs):
            args, kwargs = _decorator(*args, **kwargs)
            return cls._origin__init__(instance, *args, **kwargs)
        cls._origin__init__ = cls.__init__
        cls.__init__ = _init_decorator
        return cls
    return _decorator_wrap


