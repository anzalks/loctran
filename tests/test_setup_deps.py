"""Tests for loctran.setup_deps — dependency detection and install logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestDetectPlatform:
    def test_returns_os_and_arch(self):
        from loctran.setup_deps import detect_platform

        result = detect_platform()
        assert "os" in result
        assert "arch" in result
        assert "pkg_manager" in result
        assert "in_docker" in result
        assert "is_root" in result

    def test_docker_detection_dockerenv(self):
        from loctran.setup_deps import is_docker

        with patch("loctran.setup_deps.Path") as mock_path:
            instance = MagicMock()
            instance.exists.return_value = True
            mock_path.return_value = instance
            assert is_docker() is True

    def test_docker_env_var(self):
        from loctran.setup_deps import is_docker

        with (
            patch("loctran.setup_deps.Path") as mock_path,
            patch.dict("os.environ", {"container": "docker"}),
        ):
            instance = MagicMock()
            instance.exists.return_value = False
            mock_path.return_value = instance
            assert is_docker() is True

    def test_docker_cgroup(self):
        from loctran.setup_deps import is_docker

        mock_dockerenv = MagicMock()
        mock_dockerenv.exists.return_value = False

        mock_cgroup = MagicMock()
        mock_cgroup.exists.return_value = True
        mock_cgroup.read_text.return_value = "12:blkio:/docker/abc123"

        def make_path(p):
            if str(p) == "/.dockerenv":
                return mock_dockerenv
            return mock_cgroup

        with (
            patch("loctran.setup_deps.Path", side_effect=make_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert is_docker() is True

    def test_not_docker(self):
        from loctran.setup_deps import is_docker

        with (
            patch("loctran.setup_deps.Path") as mock_path,
            patch.dict("os.environ", {}, clear=True),
        ):
            instance = MagicMock()
            instance.exists.return_value = False
            mock_path.return_value = instance
            assert is_docker() is False

    def test_linux_pkg_managers(self):
        from loctran.setup_deps import detect_platform

        def fake_which(name):
            return "/usr/bin/dnf" if name == "dnf" else None

        with (
            patch("platform.system", return_value="Linux"),
            patch("loctran.setup_deps.shutil.which", side_effect=fake_which),
            patch("loctran.setup_deps.is_docker", return_value=False),
            patch("loctran.setup_deps._is_root", return_value=False),
        ):
            result = detect_platform()
            assert result["os"] == "linux"
            assert result["pkg_manager"] == "dnf"

    def test_windows_pkg_managers(self):
        from loctran.setup_deps import detect_platform

        def fake_which(name):
            return r"C:\choco" if name == "choco" else None

        with (
            patch("platform.system", return_value="Windows"),
            patch("loctran.setup_deps.shutil.which", side_effect=fake_which),
            patch("loctran.setup_deps.is_docker", return_value=False),
            patch("loctran.setup_deps._is_root", return_value=False),
        ):
            result = detect_platform()
            assert result["os"] == "windows"
            assert result["pkg_manager"] == "choco"


class TestCheckFunctions:
    def test_check_tesseract(self):
        from loctran.setup_deps import check_tesseract

        with patch(
            "loctran.diagnostics._check_tesseract", return_value=(True, "5.3.4")
        ):
            r = check_tesseract()
            assert r["installed"] is True

    def test_check_ollama(self):
        from loctran.setup_deps import check_ollama

        with (
            patch(
                "loctran.diagnostics._check_ollama",
                return_value=(True, "running"),
            ),
            patch("loctran.setup_deps.shutil.which", return_value="/usr/bin/ollama"),
        ):
            r = check_ollama()
            assert r["installed"] is True
            assert r["running"] is True

    def test_check_model(self):
        from loctran.setup_deps import check_model

        with patch(
            "loctran.diagnostics._check_model",
            return_value=(True, "pulled (2.2 GB)"),
        ):
            r = check_model("glm-ocr")
            assert r["pulled"] is True


class TestCheckAll:
    def test_returns_all_keys(self):
        from loctran.setup_deps import check_all

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": True, "detail": "5.3"},
            ),
            patch(
                "loctran.setup_deps.check_ollama",
                return_value={"installed": True, "running": True, "detail": "running"},
            ),
            patch(
                "loctran.setup_deps.check_model",
                return_value={"pulled": True, "detail": "pulled"},
            ),
        ):
            result = check_all()
            assert "platform" in result
            assert "tesseract" in result
            assert "ollama" in result
            assert "models" in result
            assert result["all_ok"] is True

    def test_missing_tesseract_not_all_ok(self):
        from loctran.setup_deps import check_all

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "not found"},
            ),
            patch(
                "loctran.setup_deps.check_ollama",
                return_value={"installed": True, "running": True, "detail": "ok"},
            ),
            patch(
                "loctran.setup_deps.check_model",
                return_value={"pulled": True, "detail": "ok"},
            ),
        ):
            result = check_all()
            assert result["all_ok"] is False

    def test_ollama_not_running_skips_model_check(self):
        from loctran.setup_deps import check_all

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": True, "detail": "ok"},
            ),
            patch(
                "loctran.setup_deps.check_ollama",
                return_value={
                    "installed": False,
                    "running": False,
                    "detail": "not running",
                },
            ),
            patch("loctran.setup_deps.check_model") as mock_model,
        ):
            result = check_all()
            mock_model.assert_not_called()
            assert result["all_ok"] is False


class TestRunCmd:
    def test_successful_command(self):
        from loctran.setup_deps import _run_cmd

        ok, out = _run_cmd(["echo", "hello"])
        assert ok is True
        assert "hello" in out

    def test_failing_command(self):
        from loctran.setup_deps import _run_cmd

        ok, out = _run_cmd(["false"])
        assert ok is False

    def test_missing_command(self):
        from loctran.setup_deps import _run_cmd

        ok, out = _run_cmd(["nonexistent_binary_xyz"])
        assert ok is False
        assert "not found" in out.lower() or "Command not found" in out

    def test_progress_callback(self):
        from loctran.setup_deps import _run_cmd

        msgs = []
        ok, _ = _run_cmd(["echo", "test"], progress=lambda m, p: msgs.append(m))
        assert ok is True
        assert len(msgs) >= 1


class TestInstallTesseract:
    def test_already_installed(self):
        from loctran.setup_deps import install_tesseract

        with patch(
            "loctran.setup_deps.check_tesseract",
            return_value={"installed": True, "detail": "5.3"},
        ):
            result = install_tesseract()
            assert result["success"] is True

    def test_unsupported_platform(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "haiku", "pkg_manager": None, "in_docker": False},
            ),
        ):
            result = install_tesseract()
            assert result["success"] is False

    def test_docker_returns_instruction(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "linux", "pkg_manager": "apt", "in_docker": True},
            ),
        ):
            result = install_tesseract()
            assert result["success"] is False
            assert "Docker" in result["detail"]

    def test_macos_brew_success(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={
                    "os": "darwin",
                    "pkg_manager": "brew",
                    "in_docker": False,
                },
            ),
            patch("loctran.setup_deps._run_cmd", return_value=(True, "installed")),
        ):
            result = install_tesseract()
            assert result["success"] is True
            assert "Homebrew" in result["detail"]

    def test_macos_no_brew(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "darwin", "pkg_manager": None, "in_docker": False},
            ),
        ):
            result = install_tesseract()
            assert result["success"] is False
            assert "brew.sh" in result["detail"]

    def test_linux_apt_success(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "linux", "pkg_manager": "apt", "in_docker": False},
            ),
            patch("loctran.setup_deps._run_cmd", return_value=(True, "ok")),
            patch("loctran.setup_deps._is_root", return_value=False),
        ):
            result = install_tesseract()
            assert result["success"] is True

    def test_linux_apt_failure_gives_manual_cmd(self):
        from loctran.setup_deps import install_tesseract

        call_count = [0]

        def fake_run(*a, **kw):
            call_count[0] += 1
            if call_count[0] <= 1:
                return (True, "ok")  # apt-get update
            return (False, "permission denied")

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "linux", "pkg_manager": "apt", "in_docker": False},
            ),
            patch("loctran.setup_deps._run_cmd", side_effect=fake_run),
            patch("loctran.setup_deps._is_root", return_value=False),
        ):
            result = install_tesseract()
            assert result["success"] is False
            assert "manual_cmd" in result

    def test_linux_dnf(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "linux", "pkg_manager": "dnf", "in_docker": False},
            ),
            patch("loctran.setup_deps._run_cmd", return_value=(True, "ok")),
            patch("loctran.setup_deps._is_root", return_value=True),
        ):
            result = install_tesseract()
            assert result["success"] is True

    def test_linux_pacman(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={
                    "os": "linux",
                    "pkg_manager": "pacman",
                    "in_docker": False,
                },
            ),
            patch("loctran.setup_deps._run_cmd", return_value=(True, "ok")),
            patch("loctran.setup_deps._is_root", return_value=True),
        ):
            result = install_tesseract()
            assert result["success"] is True

    def test_linux_no_pkg_manager(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "linux", "pkg_manager": None, "in_docker": False},
            ),
        ):
            result = install_tesseract()
            assert result["success"] is False

    def test_windows_choco(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={
                    "os": "windows",
                    "pkg_manager": "choco",
                    "in_docker": False,
                },
            ),
            patch("loctran.setup_deps._run_cmd", return_value=(True, "ok")),
        ):
            result = install_tesseract()
            assert result["success"] is True

    def test_windows_winget(self):
        from loctran.setup_deps import install_tesseract

        call_count = [0]

        def fake_run(cmd, *a, **kw):
            call_count[0] += 1
            if "choco" in cmd:
                return (False, "not found")
            return (True, "ok")

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={
                    "os": "windows",
                    "pkg_manager": "winget",
                    "in_docker": False,
                },
            ),
            patch("loctran.setup_deps._run_cmd", side_effect=fake_run),
        ):
            result = install_tesseract()
            assert result["success"] is True

    def test_windows_no_pkg_manager(self):
        from loctran.setup_deps import install_tesseract

        with (
            patch(
                "loctran.setup_deps.check_tesseract",
                return_value={"installed": False, "detail": "missing"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "windows", "pkg_manager": None, "in_docker": False},
            ),
        ):
            result = install_tesseract()
            assert result["success"] is False
            assert "UB-Mannheim" in result["detail"]


class TestInstallOllama:
    def test_already_running(self):
        from loctran.setup_deps import install_ollama

        with patch(
            "loctran.setup_deps.check_ollama",
            return_value={"installed": True, "running": True, "detail": "ok"},
        ):
            result = install_ollama()
            assert result["success"] is True

    def test_macos_brew_install_and_start(self):
        from loctran.setup_deps import install_ollama

        call_count = [0]

        def fake_check():
            call_count[0] += 1
            return {
                "installed": call_count[0] > 1,
                "running": call_count[0] > 2,
                "detail": "ok",
            }

        with (
            patch("loctran.setup_deps.check_ollama", side_effect=fake_check),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={
                    "os": "darwin",
                    "pkg_manager": "brew",
                    "in_docker": False,
                },
            ),
            patch("loctran.setup_deps._run_cmd", return_value=(True, "ok")),
            patch("loctran.setup_deps.subprocess.Popen"),
            patch("loctran.setup_deps.time.sleep"),
        ):
            result = install_ollama()
            assert result["success"] is True

    def test_linux_install_script(self):
        from loctran.setup_deps import install_ollama

        call_count = [0]

        def fake_check():
            call_count[0] += 1
            return {
                "installed": call_count[0] > 1,
                "running": call_count[0] > 2,
                "detail": "ok",
            }

        with (
            patch("loctran.setup_deps.check_ollama", side_effect=fake_check),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "linux", "pkg_manager": "apt", "in_docker": False},
            ),
            patch("loctran.setup_deps._run_cmd", return_value=(True, "ok")),
            patch("loctran.setup_deps.subprocess.Popen"),
            patch("loctran.setup_deps.time.sleep"),
        ):
            result = install_ollama()
            assert result["success"] is True

    def test_windows_returns_download_link(self):
        from loctran.setup_deps import install_ollama

        with (
            patch(
                "loctran.setup_deps.check_ollama",
                return_value={
                    "installed": False,
                    "running": False,
                    "detail": "missing",
                },
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "windows", "pkg_manager": None, "in_docker": False},
            ),
        ):
            result = install_ollama()
            assert result["success"] is False
            assert "ollama.com" in result["detail"]

    def test_unsupported_platform(self):
        from loctran.setup_deps import install_ollama

        with (
            patch(
                "loctran.setup_deps.check_ollama",
                return_value={
                    "installed": False,
                    "running": False,
                    "detail": "missing",
                },
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={"os": "haiku", "pkg_manager": None, "in_docker": False},
            ),
        ):
            result = install_ollama()
            assert result["success"] is False

    def test_start_fails_binary_not_found(self):
        from loctran.setup_deps import install_ollama

        with (
            patch(
                "loctran.setup_deps.check_ollama",
                return_value={"installed": True, "running": False, "detail": "stopped"},
            ),
            patch(
                "loctran.setup_deps.detect_platform",
                return_value={
                    "os": "darwin",
                    "pkg_manager": "brew",
                    "in_docker": False,
                },
            ),
            patch(
                "loctran.setup_deps.subprocess.Popen",
                side_effect=FileNotFoundError("not found"),
            ),
        ):
            result = install_ollama()
            assert result["success"] is False
            assert "not found" in result["detail"]


class TestPullModels:
    def test_already_pulled(self):
        from loctran.setup_deps import pull_models

        with patch(
            "loctran.setup_deps.check_model",
            return_value={"pulled": True, "detail": "ok"},
        ):
            result = pull_models()
            assert result["success"] is True

    def test_pull_success(self):
        from loctran.setup_deps import pull_models

        mock_ollama = MagicMock()
        with (
            patch(
                "loctran.setup_deps.check_model",
                return_value={"pulled": False, "detail": "missing"},
            ),
            patch("loctran.setup_deps.ollama", mock_ollama, create=True),
        ):
            # Need to mock the import inside the function
            import sys

            sys.modules["ollama"] = mock_ollama
            try:
                result = pull_models()
                assert result["success"] is True
            finally:
                del sys.modules["ollama"]

    def test_pull_failure(self):
        from loctran.setup_deps import pull_models

        mock_ollama = MagicMock()
        mock_ollama.pull.side_effect = RuntimeError("network error")
        with patch(
            "loctran.setup_deps.check_model",
            return_value={"pulled": False, "detail": "missing"},
        ):
            import sys

            sys.modules["ollama"] = mock_ollama
            try:
                result = pull_models()
                assert result["success"] is False
            finally:
                del sys.modules["ollama"]


class TestInstallAll:
    def test_all_ok_skips_installs(self):
        from loctran.setup_deps import install_all

        with patch("loctran.setup_deps.check_all", return_value={"all_ok": True}):
            result = install_all()
            assert result["success"] is True

    def test_installs_everything(self):
        from loctran.setup_deps import install_all

        with (
            patch(
                "loctran.setup_deps.check_all",
                return_value={
                    "all_ok": False,
                    "tesseract": {"installed": False},
                    "ollama": {"running": False},
                    "models": {"m": {"pulled": False}},
                },
            ),
            patch(
                "loctran.setup_deps.install_tesseract",
                return_value={"success": True, "detail": "ok"},
            ),
            patch(
                "loctran.setup_deps.install_ollama",
                return_value={"success": True, "detail": "ok"},
            ),
            patch(
                "loctran.setup_deps.pull_models",
                return_value={"success": True, "models": {}},
            ),
        ):
            result = install_all()
            assert result["success"] is True

    def test_skips_models_when_ollama_fails(self):
        from loctran.setup_deps import install_all

        with (
            patch(
                "loctran.setup_deps.check_all",
                return_value={
                    "all_ok": False,
                    "tesseract": {"installed": True},
                    "ollama": {"running": False},
                    "models": {},
                },
            ),
            patch(
                "loctran.setup_deps.install_ollama",
                return_value={"success": False, "detail": "failed"},
            ),
        ):
            result = install_all()
            assert result["success"] is False
            assert result["results"]["models"]["success"] is False
            assert "Skipped" in result["results"]["models"]["detail"]

    def test_progress_callback(self):
        from loctran.setup_deps import install_all

        msgs = []

        with (
            patch(
                "loctran.setup_deps.check_all",
                return_value={
                    "all_ok": False,
                    "tesseract": {"installed": True},
                    "ollama": {"running": True},
                    "models": {"m": {"pulled": False}},
                },
            ),
            patch(
                "loctran.setup_deps.pull_models",
                return_value={"success": True, "models": {}},
            ),
        ):
            install_all(progress=lambda m, p: msgs.append(m))
            assert len(msgs) >= 1
