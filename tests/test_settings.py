"""
Tests for loading/configuring settings.
"""

import specster
from specster._settings import Settings, read_settings, write_settings


class TestBasic:
    """Tests for settings default behavior."""

    def test_global_setting_set(self):
        """specster.settings should be a settings instance."""
        assert isinstance(specster.settings, Settings)

    def test_read_write(self, tmp_path):
        """Ensure writing settings to disk works."""
        setting = Settings()
        path = tmp_path / "specster_settings.json"
        write_settings(path, setting)
        assert path.exists()
        loaded_settings = read_settings(path, use_settings=False)
        assert loaded_settings == setting
