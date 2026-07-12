import unittest

from torch import nn

from src.models import ModelAdapter, ModelBundle, get_model_adapter, model_names, register_model


class ModelRegistryTests(unittest.TestCase):
    def test_builtin_models_are_registered(self):
        self.assertIn("supervlad", model_names())
        self.assertIn("boq", model_names())
        self.assertEqual(get_model_adapter("supervlad").name, "supervlad")

    def test_unknown_model_reports_available_models(self):
        with self.assertRaisesRegex(ValueError, "Available models"):
            get_model_adapter("missing-model")

    def test_duplicate_registration_is_rejected(self):
        def add_arguments(parser):
            del parser

        def build(args):
            del args
            return ModelBundle(nn.Identity(), 1)

        def load_weights(model, path, args):
            del model, path, args

        adapter = get_model_adapter("supervlad")
        register_model(adapter)

        conflicting = ModelAdapter("supervlad", add_arguments, build, load_weights)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_model(conflicting)


if __name__ == "__main__":
    unittest.main()
