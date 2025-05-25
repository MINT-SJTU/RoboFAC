import os
import shutil
import site

def replace_info_raw(src_path):
    dst_path = os.path.expanduser("~/.maniskill/data/assets/mani_skill2_ycb/info_raw.json")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copyfile(src_path, dst_path)
    print(f"Replaced info_raw.json at: {dst_path}")

def find_scene_builder_path():
    for path in site.getsitepackages():
        candidate = os.path.join(path, "mani_skill", "utils", "scene_builder", "table", "scene_builder.py")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Could not locate scene_builder.py in ManiSkill installation.")

def replace_scene_builder(src_path):
    dst_path = find_scene_builder_path()
    shutil.copyfile(src_path, dst_path)
    print(f"Replaced scene_builder.py at: {dst_path}")
    return os.path.dirname(dst_path)

def update_init_py(scene_builder_dir):
    init_file = os.path.join(scene_builder_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            pass  # Create empty file

    with open(init_file, "r") as f:
        content = f.read()

    line = "from .scene_builder import noTableSceneBuilder"
    if line not in content:
        with open(init_file, "a") as f:
            f.write(f"\n{line}\n")
        print(f"Added '{line}' to {init_file}")
    else:
        print(f"Line already exists in {init_file}")
    
    line = "from .scene_builder import noTableSceneBuilder_microwave"
    if line not in content:
        with open(init_file, "a") as f:
            f.write(f"\n{line}\n")
        print(f"Added '{line}' to {init_file}")
    else:
        print(f"Line already exists in {init_file}")

if __name__ == "__main__":
    print("Starting RoboFAC environment configuration...\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")

    info_raw_file = os.path.join(config_dir, "info_raw.json")
    scene_builder_file = os.path.join(config_dir, "scene_builder.py")

    if not os.path.isfile(info_raw_file):
        print(f"Missing file: {info_raw_file}")
    else:
        replace_info_raw(info_raw_file)

    if not os.path.isfile(scene_builder_file):
        print(f"Missing file: {scene_builder_file}")
    else:
        scene_builder_dir = replace_scene_builder(scene_builder_file)
        update_init_py(scene_builder_dir)

    print("\nConfiguration complete!")
