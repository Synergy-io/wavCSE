"""Checkout resolution: explicit, discovered, or a loud failure — never silent."""

import os
import sys
import unittest

from improvements.compute import resolve as resolve_module
from improvements.compute.errors import ConfigurationError
from improvements.compute.tests.fakes import ComputeTestCase

SKILL_PATH = os.path.join(".agents", "skills", "wavcse-infra-operator", "SKILL.md")


class ResolutionTests(ComputeTestCase):
    def setUp(self):
        super(ResolutionTests, self).setUp()
        self._saved = {}
        for name in (resolve_module.CHECKOUT_ENV, resolve_module.CLI_ENV):
            self._saved[name] = os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        super(ResolutionTests, self).tearDown()

    def fake_checkout(self, *, with_cli=True):
        root = os.path.join(self.home, "infra-checkout")
        skill = os.path.join(root, SKILL_PATH)
        os.makedirs(os.path.dirname(skill), exist_ok=True)
        with open(skill, "w", encoding="utf-8") as handle:
            handle.write("# fake\n")
        if with_cli:
            binary = os.path.join(root, ".venv", "bin", "infra")
            os.makedirs(os.path.dirname(binary), exist_ok=True)
            with open(binary, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\nexit 0\n")
            os.chmod(binary, 0o755)
        return root

    def fake_embedded_subsystem(self, *, with_cli=True):
        root = os.path.join(self.home, "infra")
        package = os.path.join(root, "src", "wavcse_infra", "__init__.py")
        os.makedirs(os.path.dirname(package), exist_ok=True)
        with open(package, "w", encoding="utf-8") as handle:
            handle.write('"""fake embedded infra"""\n')
        with open(os.path.join(root, "pyproject.toml"), "w", encoding="utf-8") as handle:
            handle.write('[project]\nname = "wavcse-infra"\n')
        if with_cli:
            binary = os.path.join(root, ".venv", "bin", "infra")
            os.makedirs(os.path.dirname(binary), exist_ok=True)
            with open(binary, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\nexit 0\n")
            os.chmod(binary, 0o755)
        return root

    def test_embedded_subsystem_resolves_by_default(self):
        root = self.fake_embedded_subsystem()

        location = resolve_module.resolve()

        self.assertEqual(location.checkout, root)
        self.assertEqual(location.source, "monorepo")
        self.assertTrue(location.cli.endswith(os.path.join(".venv", "bin", "infra")))

    def test_embedded_subsystem_without_cli_fails_loudly(self):
        root = self.fake_embedded_subsystem(with_cli=False)

        with self.assertRaises(ConfigurationError) as caught:
            resolve_module.resolve()

        self.assertIn(root, str(caught.exception))
        self.assertIn("uv sync", str(caught.exception))

    def test_explicit_checkout_resolves(self):
        root = self.fake_checkout()
        os.environ[resolve_module.CHECKOUT_ENV] = root
        location = resolve_module.resolve()
        self.assertEqual(location.checkout, root)
        self.assertTrue(location.cli.endswith(os.path.join(".venv", "bin", "infra")))
        self.assertEqual(location.source, "environment")

    def test_explicit_checkout_without_marker_fails_loudly(self):
        os.environ[resolve_module.CHECKOUT_ENV] = os.path.join(self.home, "not-a-checkout")
        with self.assertRaises(ConfigurationError) as caught:
            resolve_module.resolve()
        self.assertIn(resolve_module.CHECKOUT_ENV, str(caught.exception))

    def test_explicit_cli_resolves(self):
        root = self.fake_checkout()
        binary = os.path.join(root, ".venv", "bin", "infra")
        os.environ[resolve_module.CLI_ENV] = binary
        location = resolve_module.resolve()
        self.assertEqual(location.cli, binary)
        self.assertEqual(location.checkout, root)

    def test_mismatched_checkout_and_cli_are_refused(self):
        first = self.fake_checkout()
        second = os.path.join(self.home, "other-infra")
        os.makedirs(os.path.join(second, ".venv", "bin"), exist_ok=True)
        binary = os.path.join(second, ".venv", "bin", "infra")
        with open(binary, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\nexit 0\n")
        os.chmod(binary, 0o755)
        os.environ[resolve_module.CHECKOUT_ENV] = first
        os.environ[resolve_module.CLI_ENV] = binary
        with self.assertRaises(ConfigurationError):
            resolve_module.resolve()

    def test_non_executable_cli_is_refused(self):
        path = os.path.join(self.home, "infra")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\n")
        os.environ[resolve_module.CLI_ENV] = path
        with self.assertRaises(ConfigurationError):
            resolve_module.resolve()

    def test_environment_override_beats_path(self):
        root = self.fake_checkout()
        os.environ[resolve_module.CHECKOUT_ENV] = root
        location = resolve_module.resolve()
        self.assertEqual(location.source, "environment")

    def test_path_discovery_recovers_the_checkout(self):
        root = self.fake_checkout()
        binary = os.path.join(root, ".venv", "bin", "infra")
        bindir = os.path.join(self.home, "bin")
        os.makedirs(bindir, exist_ok=True)
        link = os.path.join(bindir, "infra")
        os.symlink(binary, link)
        saved_path = os.environ.get("PATH")
        os.environ["PATH"] = bindir + os.pathsep + (saved_path or "")
        try:
            location = resolve_module.resolve()
            self.assertEqual(location.checkout, root)
            self.assertEqual(location.source, "path")
        finally:
            if saved_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = saved_path


if __name__ == "__main__":
    unittest.main()
