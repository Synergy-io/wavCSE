"""Unit tests for mtlkit/registry.py (Eng Review decision 3B)."""

import unittest

from mtlkit.registry import Registry


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry: Registry[str] = Registry("widget")

    def test_register_then_get_roundtrips(self):
        self.registry.register("a", "value-a")
        self.assertEqual(self.registry.get("a"), "value-a")

    def test_register_returns_the_registered_item(self):
        returned = self.registry.register("a", "value-a")
        self.assertEqual(returned, "value-a")

    def test_duplicate_key_raises_keyerror(self):
        self.registry.register("a", "value-a")
        with self.assertRaises(KeyError) as ctx:
            self.registry.register("a", "value-a-again")
        self.assertIn("already registered", str(ctx.exception))
        self.assertIn("widget", str(ctx.exception))

    def test_unknown_key_raises_keyerror_with_valid_keys_listed(self):
        self.registry.register("a", "value-a")
        self.registry.register("b", "value-b")
        with self.assertRaises(KeyError) as ctx:
            self.registry.get("z")
        message = str(ctx.exception)
        self.assertIn("Unknown widget 'z'", message)
        self.assertIn("a", message)
        self.assertIn("b", message)

    def test_unknown_key_on_empty_registry_says_none_registered(self):
        with self.assertRaises(KeyError) as ctx:
            self.registry.get("z")
        self.assertIn("(none registered)", str(ctx.exception))

    def test_try_get_returns_none_on_missing_key(self):
        self.assertIsNone(self.registry.try_get("missing"))

    def test_try_get_returns_value_on_present_key(self):
        self.registry.register("a", "value-a")
        self.assertEqual(self.registry.try_get("a"), "value-a")

    def test_list_returns_sorted_keys(self):
        self.registry.register("z", "value-z")
        self.registry.register("a", "value-a")
        self.assertEqual(self.registry.list(), ["a", "z"])

    def test_contains_and_len(self):
        self.registry.register("a", "value-a")
        self.assertIn("a", self.registry)
        self.assertNotIn("b", self.registry)
        self.assertEqual(len(self.registry), 1)

    def test_iter_yields_sorted_keys(self):
        self.registry.register("b", "value-b")
        self.registry.register("a", "value-a")
        self.assertEqual(list(self.registry), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
