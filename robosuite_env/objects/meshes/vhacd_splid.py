import os
import trimesh

def export_convex_parts(obj_path, out_dir, prefix="part"):
    os.makedirs(out_dir, exist_ok=True)
    mesh = trimesh.load(obj_path, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    parts = mesh.split(only_watertight=False)

    files = []
    for i, part in enumerate(parts):
        filename = f"{prefix}{i}.obj"
        fullpath = os.path.join(out_dir, filename)
        part.export(fullpath)
        files.append(fullpath)

    print(f"✔ Exported {len(files)} convex parts to '{out_dir}'")
    return files

object_name = "wood_block"
current_dir = os.path.dirname(os.path.abspath(__file__))
name_in = os.path.join(current_dir, f"{object_name}_vhacd.obj")
name_out = os.path.join(current_dir, f"{object_name}_parts")
export_convex_parts(name_in, name_out, prefix=object_name)
