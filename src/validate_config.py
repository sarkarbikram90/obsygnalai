import sys
from pathlib import Path

REQUIRED_FILES = [
    Path("src/app.py"),
    Path("src/start.sh"),
    Path("Dockerfile"),
    Path("requirements.txt"),
    Path("tests/test_app.py"),
    Path("src/validate_config.py"),
    Path("firestore.indexes.json"),
    Path(".github/workflows/deploy.yml"),
    Path("infra/deploy.sh"),
    Path("infra/setup_gcp.sh"),
    Path("README.md"),
]

def validate_paths() -> bool:
    print("--- [1/2] Verifying Workspace File Structure ---")
    all_valid = True
    for file_path in REQUIRED_FILES:
        if not file_path.exists():
            print(f"  [MISSING] {file_path}")
            all_valid = False
        else:
            print(f"  [FOUND]   {file_path}")
    return all_valid

def validate_dockerfile() -> bool:
    print("\n--- [2/2] Verifying Dockerfile Syntax Configuration Rules ---")
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("  [FAIL] Dockerfile does not exist.")
        return False
    
    content = dockerfile_path.read_text(encoding="utf-8")
    if "EXPOSE 8080" not in content:
        print("  [FAIL] Dockerfile missing mandatory 'EXPOSE 8080' directive.")
        return False
    
    print("  [PASS] Mandatory rule verified: 'EXPOSE 8080' is present.")
    return True

if __name__ == "__main__":
    paths_ok = validate_paths()
    dockerfile_ok = validate_dockerfile()

    if paths_ok and dockerfile_ok:
        print("\n[SUCCESS] All structural paths and configuration rules validated successfully.")
        sys.exit(0)
    else:
        print("\n[ERROR] Configuration validation failed.")
        sys.exit(1)
