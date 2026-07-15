from __future__ import annotations

import guidellm.utils.lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_extras(
    __name__,
    attrs={
        "use": lazy.ExtraAttr("matplotlib", alias="use"),
        "plt": lazy.ExtraAttr("matplotlib", alias="pyplot"),
        "Axes": lazy.ExtraAttr("matplotlib.axes", alias="Axes"),
    },
    error_message="Please install guidellm[plot] to use plot output features",
)
