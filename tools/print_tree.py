import os
from pathlib import Path

def print_tree(directory, prefix=""):
    """
    Recursively prints the folder structure in a tree format.
    """
    # Folders to ignore to keep the output clean
    ignore_list = {".git", "__pycache__", ".ipynb_checkpoints", ".DS_Store"}
    
    try:
        # Get all items in the directory and sort them
        items = sorted([item for item in os.listdir(directory) if item not in ignore_list])
    except PermissionError:
        return
    
    for i, item in enumerate(items):
        path = os.path.join(directory, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "

        if os.path.isdir(path):
            # Count only files (not recursing into subfolders)
            try:
                num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            except Exception:
                num_files = 0
            
            # If a folder is massive (like data/processed), summarize it
            if num_files > 15:
                display_name = "raw_images" if os.path.basename(path) == "images" else os.path.basename(path)
                print(f"{prefix}{connector}{display_name} [{num_files} files]")
                continue
            else:
                # Print the directory name and recurse inside
                print(f"{prefix}{connector}{item}")
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(path, new_prefix)
        else:
            # It's just a file
            print(f"{prefix}{connector}{item}")

if __name__ == "__main__":
    # --- PATH RESOLUTION ---
    # Looks one level up from the /tools folder to guarantee we capture the project root
    root_path = Path(__file__).resolve().parent.parent
    
    print(f"\n[ROOT] {root_path.name}/")
    print_tree(str(root_path))
    print("\n--- Structure Report Complete ---")