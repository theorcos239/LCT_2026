# -*- coding: utf-8 -*-
"""Классификатор области DXA-снимка: lh (левое бедро) / rh (правое) / spine."""
from .features import CLASS_RU, CLASSES

__all__ = ['RegionClassifier', 'CLASSES', 'CLASS_RU']


def __getattr__(name):
    """Ленивый импорт: иначе `python -m region_clf.predict` грузит модуль дважды."""
    if name == 'RegionClassifier':
        from .predict import RegionClassifier
        return RegionClassifier
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
