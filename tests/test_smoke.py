from typer.testing import CliRunner

import handwriter
from handwriter.app.cli import app
from handwriter.config import a4_size_px, mm_to_px


def test_version():
    assert handwriter.__version__


def test_a4_size_at_300_dpi():
    assert a4_size_px(300) == (2480, 3508)


def test_mm_to_px():
    assert mm_to_px(25.4, 300) == 300


def test_cli_info():
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "2480 x 3508" in result.output
