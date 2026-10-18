"""Run every offline analysis in order."""
import runpy, pathlib
here = pathlib.Path(__file__).parent
for s in ["01_reproduce_frozen.py", "02_benchmark_and_statistics.py", "03_factorial_exact.py", "04_degradation_fallback.py"]:
    print(f"\n===== {s} =====")
    runpy.run_path(str(here / s), run_name="__main__")
print("\nAll analyses completed.")
