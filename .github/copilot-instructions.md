# Copilot Instructions

## Plot And Figure Rules

- When generating plots or guide images, use a 16:9 canvas by default.
- In matplotlib, use `figsize=(16, 9)` unless the user explicitly asks for a different ratio.
- Do not use `bbox_inches="tight"` when exact aspect ratio must be preserved.

## Python Execution Rules

- After editing Python scripts, run them in the project virtual environment and confirm output.
- Use this interpreter path for execution:
  `c:/programming/signal-integrity-notes/.venv/Scripts/python.exe`
- Report the saved output file path and whether execution completed successfully.
