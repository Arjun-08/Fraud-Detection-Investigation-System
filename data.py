from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("=" * 80)
print("DATASET PATH DIAGNOSTIC")
print("=" * 80)

print(f"\nScript location:")
print(ROOT)

data_root = ROOT / "data"

print(f"\nData directory:")
print(data_root)

print(f"\nData directory exists: {data_root.exists()}")

print("\nEverything under data/:")
print("-" * 80)

for path in data_root.rglob("*"):
    print(
        f"{'[DIR ]' if path.is_dir() else '[FILE]'} "
        f"{path}"
    )

print("\n" + "=" * 80)
print("SEARCHING FOR PICKLE FILES")
print("=" * 80)

pkl_files = list(data_root.rglob("*.pkl"))

print(f"\nNumber of .pkl files: {len(pkl_files)}")

for file in pkl_files[:20]:
    print(file)

print("\n" + "=" * 80)
print("DIRECTORY CONTENT CHECK")
print("=" * 80)

dataset_root = data_root / "simulated-data-raw"

print(f"\nDataset root:")
print(dataset_root)

print(f"Exists: {dataset_root.exists()}")

if dataset_root.exists():

    print("\nImmediate contents:")

    for item in dataset_root.iterdir():

        print(
            f"{'[DIR ]' if item.is_dir() else '[FILE]'} "
            f"{item.name}"
        )

    nested_data = dataset_root / "data"

    print("\nNested data directory:")
    print(nested_data)

    print(f"Exists: {nested_data.exists()}")

    if nested_data.exists():

        print("\nNested data contents:")

        for item in list(nested_data.iterdir())[:20]:

            print(
                f"{'[DIR ]' if item.is_dir() else '[FILE]'} "
                f"{item.name}"
            )