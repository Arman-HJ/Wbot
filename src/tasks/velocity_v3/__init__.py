"""Velocity v3: high-speed velocity tracking (v1-style, up to 3 m/s)."""

from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
