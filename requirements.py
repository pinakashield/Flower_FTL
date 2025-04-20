import os
import re

def find_imports_in_project(directory):
    imports = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read()
                    matches = re.findall(r'^\s*(?:import|from)\s+([\w\.]+)', content, re.MULTILINE)
                    imports.update(matches)
    return imports

def main():
    project_dir = "/Users/peenalgupta/PinakaShield/GitHub/Flower_FTL_IDS"
    imports = find_imports_in_project(project_dir)
    with open(os.path.join(project_dir, "requirements.txt"), "w", encoding="utf-8") as req_file:
        for lib in sorted(imports):
            req_file.write(f"{lib}\n")
    print("requirements.txt has been updated with the detected libraries.")

if __name__ == "__main__":
    main()