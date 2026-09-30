"""Adapters around the unmodified SuperVLAD submodule.

Upstream only exposes ``parser.parse_arguments()`` (parses sys.argv), refuses existing
log directories, and checks a hard-coded GSV-Cities path at import time. These helpers
reuse the upstream code as-is by patching those spots for the duration of one call.
"""

import argparse
import os
import pathlib
from unittest import mock

import commons
import parser as parser_module


class _CapturedParser(Exception):
    def __init__(self, parser):
        self.parser = parser


def _raise_captured(parser, *args, **kwargs):
    raise _CapturedParser(parser)


def build_parser():
    with mock.patch.object(argparse.ArgumentParser, "parse_args", _raise_captured):
        try:
            parser_module.parse_arguments()
        except _CapturedParser as captured:
            parser = captured.parser
    parser.set_defaults(patience=5)
    return parser


def validate_arguments(args):
    with mock.patch.object(argparse.ArgumentParser, "parse_args", lambda *_, **__: args):
        return parser_module.parse_arguments()


def setup_logging(save_dir, allow_existing=False):
    if allow_existing and os.path.exists(save_dir):
        with mock.patch.object(commons.os.path, "exists", return_value=False):
            return commons.setup_logging(save_dir)
    return commons.setup_logging(save_dir)


def test(*args, **kwargs):
    """Upstream test.test, with loaders that do not use /dev/shm (src.data.PipeDataLoader)."""
    import test as test_module

    from .data import PipeDataLoader

    with mock.patch.object(test_module, "DataLoader", PipeDataLoader):
        return test_module.test(*args, **kwargs)


def import_gsv_cities_dataset():
    """Return upstream GSVCitiesDataset. Pass ``base_path`` as a str ending in "/"."""
    import pandas  # noqa: F401 -- import dependencies before Path.exists is patched
    import PIL.Image  # noqa: F401
    import torchvision.transforms  # noqa: F401

    with mock.patch.object(pathlib.Path, "exists", return_value=True):
        from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
    return GSVCitiesDataset
