# Loctran Development Rules & Standards

This document defines the industry-standard practices that must be followed for all development in this repository.

## 1. Architecture: Event-Driven over Polling
- **Standard**: Use persistent connections (WebSockets) for real-time state management and lifecycle control.
- **Prohibited**: Do NOT use `setInterval` polling (e.g., `fetch('/status')` every 1s) for critical application state or heartbeat monitoring.
- **Rationale**: Polling is inefficient, introduces latency, and "magic numbers" (timers). WebSockets provide instant, precise state synchronization.

## 2. Cross-Platform Compatibility
- **Standard**: All system interactions (file pickers, opening folders, subprocess calls) must explicitly handle macOS (`darwin`), Windows (`win32`), and Linux (`linux`).
- **Standard**: Use `pathlib.Path` for all file system operations. Avoid manual string concatenation for paths.
- **Standard**: For native dialogs, use platform-native tools (AppleScript for Mac, PowerShell for Windows, Zenity/Kdialog for Linux) or a cross-platform library if available/permitted.

## 3. Robustness & Lifecycle
- **Standard**: The application must not shut down unexpectedly during user interaction or active processing.
- **Standard**: **Graceful Shutdown**: The server should only shut down when:
    1. No clients are connected (Connections == 0).
    2. No background jobs are running (Active Jobs == 0).
    3. No blocking user interactions are pending (Dialogs == Closed).
    4. A grace period (e.g., 3s) has passed to allow for page reloads.

## 4. User Experience (UX)
- **Standard**: **Native Workflows**: Prefer native system dialogs for file/folder selection over web-based file inputs where path context is required.
- **Standard**: **Feedback**: Always provide immediate visual feedback for user actions (loading states, progress bars, toast notifications).
- **Standard**: **Resilience**: The UI must auto-reconnect if the server connection drops and recover gracefully from errors.

## 5. Security
- **Standard**: **Input Validation**: Sanitize all file names and paths. Use `werkzeug.utils.secure_filename` or equivalent logic.
- **Standard**: **Path Traversal**: Validate that all file operations occur within allowed directories (`OUTPUT_DIR`, `UPLOAD_DIR`, or user-selected paths).

## 6. Code Quality
- **Standard**: **Type Hinting**: Use Python type hints (`typing.List`, `typing.Optional`, etc.) for function arguments and return values.
- **Standard**: **Documentation**: Every endpoint and complex function must have a docstring explaining its purpose, arguments, and return values.
- **Standard**: **Asyncio**: Use `async/await` for I/O-bound operations (file uploads, network requests) to avoid blocking the main thread.

## 7. Scripting Standards
- **Standard**: All utility scripts (e.g., startup, cleanup) must be written in Python or Bash.
- **Standard**: Scripts must have error handling (`set -e` in bash, `try-except` in Python).
- **Standard**: Scripts must be located in a `scripts/` directory or root if essential (like `start.sh`).

## 8. CI/CD Pipeline Validation
- **Standard**: All code must pass linting (Flake8/Black for Python) before commit.
- **Standard**: GitHub Actions must be configured to run tests on every push/PR.
- **Standard**: No hardcoded secrets or absolute paths in committed code.

## 9. No Emoji Policy
- **Standard**: Do NOT use emojis (e.g., 🚀, 💀, ✅) in any part of the codebase, including:
    - Source code comments and docstrings.
    - Terminal output / logging messages.
    - User Interface (HTML/CSS) text content.
    - Documentation files.
- **Rationale**: Emojis can cause encoding issues, look unprofessional in logs, and distract from the core information. Use clear text or standard ASCII characters instead.

## 10. Logging Standards (Dev Mode)
- **Standard**: **User Mode (Default)**: The application must be silent in the terminal, showing ONLY fatal errors (`[ERROR]`).
- **Standard**: **Dev Mode**: Detailed logs (`[INFO]`, `[DEBUG]`, `[WARN]`) should only appear when Dev Mode is explicitly enabled (e.g., via `LOCTRAN_DEBUG=1` env var).
- **Rationale**: Users should not be overwhelmed by technical noise. Developers need full visibility for debugging.

## 11. AI Model Management
- **Standard**: **Lazy Loading**: Large AI models (like `deepseek-ocr:3b`) should only be pulled/loaded when explicitly requested by the user, not at startup.
- **Standard**: **Fallback**: Always provide a deterministic fallback (e.g., standard Tesseract OCR) if the AI model fails or is unavailable.
- **Standard**: **Transparency**: Explicitly inform the user (via logs or UI) when an AI model is being downloaded or used.

## 12. Mobile & PWA Standards
- **Standard**: **Responsive Design**: All UI components must be responsive and usable on small screens (< 600px). Use CSS media queries.
- **Standard**: **Touch Optimization**: Interactive elements (buttons, inputs) must have a minimum size of 44x44px for touch targets.
- **Standard**: **PWA Compliance**: The application must include a valid `manifest.json` and meta tags for theme color and mobile capability.
## 13. Commercial Viability & Licensing
- **Standard**: **License Compliance**: All third-party dependencies MUST be compatible with a proprietary/closed-source license.
    - **Allowed**: MIT, Apache 2.0, BSD, ISC.
    - **Prohibited**: GPL, AGPL, CC-BY-SA (any license with "copyleft" or "share-alike" clauses that force open-sourcing).
- **Standard**: **No Hidden Costs**: The application must not rely on paid external APIs (e.g., OpenAI, Google Cloud) unless explicitly authorized. Preference is for local, offline-capable solutions (Ollama, Tesseract, pypdfium2).
- **Standard**: **Vendor Lock-in**: Avoid dependencies that tie the core functionality to a specific cloud provider or paid service.
- **Requirement**: **Ghostscript Prohibition**: Do NOT use strictly AGPL tools like Ghostscript unless there is a commercial license in place. Use permissive alternatives like `pypdfium2` or `pdfium`.
