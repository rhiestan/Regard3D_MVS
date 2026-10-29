"""Minimal PLY reader / writer (numpy) for the evaluation scripts.

Reads ascii and binary PLY files: every element becomes a numpy structured
array; list properties (faces) are supported when all lists of the element
have the same length (triangle meshes), which is what PoissonRecon, MVE and
OpenMVS write.
"""

import numpy as np

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_header(f):
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt = None
    elements = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("truncated PLY header")
        words = line.decode("ascii", "replace").split()
        if not words or words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "format":
            fmt = words[1]
        elif words[0] == "element":
            elements.append((words[1], int(words[2]), []))
        elif words[0] == "property":
            if words[1] == "list":
                elements[-1][2].append((words[4], _TYPES[words[2]], _TYPES[words[3]]))
            else:
                elements[-1][2].append((words[2], _TYPES[words[1]], None))
        elif words[0] == "end_header":
            return fmt, elements


def read_ply(path):
    """Returns {element name: structured array}; a uniform list property
    `p` of length k becomes a field `p` of shape (k,)."""
    out = {}
    with open(path, "rb") as f:
        fmt, elements = _read_header(f)
        if fmt == "ascii":
            for name, count, props in elements:
                rows = [f.readline().split() for _ in range(count)]
                out[name] = _from_ascii(rows, props)
            return out
        order = "<" if fmt == "binary_little_endian" else ">"
        for name, count, props in elements:
            if all(p[2] is None for p in props):
                dtype = np.dtype([(p[0], order + p[1]) for p in props])
                out[name] = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count)
                continue
            # A list property: read the first row to get the list length, assume it is uniform.
            start = f.tell()
            fields = []
            row_size = 0
            for pname, ptype, itype in props:
                if itype is None:
                    fields.append((pname, order + ptype))
                    row_size += np.dtype(ptype).itemsize
                else:
                    f.seek(start + row_size)
                    k = int(np.frombuffer(f.read(np.dtype(ptype).itemsize), dtype=order + ptype)[0])
                    fields.append((pname + "_count", order + ptype))
                    fields.append((pname, order + itype, (k,)))
                    row_size += np.dtype(ptype).itemsize + k * np.dtype(itype).itemsize
            f.seek(start)
            dtype = np.dtype(fields)
            arr = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count)
            for pname, ptype, itype in props:
                if itype is not None and count > 0 and not np.all(arr[pname + "_count"] == arr[pname].shape[1]):
                    raise ValueError(f"list property {pname} of {name} has varying lengths")
            out[name] = arr
    return out


def _from_ascii(rows, props):
    fields = []
    for pname, ptype, itype in props:
        if itype is None:
            fields.append((pname, ptype))
        else:
            k = int(rows[0][len(fields)]) if rows else 0
            fields.append((pname + "_count", ptype))
            fields.append((pname, itype, (k,)))
    arr = np.zeros(len(rows), dtype=np.dtype(fields))
    flat = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 0))
    col = 0
    for field in fields:
        width = field[2][0] if len(field) == 3 else 1
        values = flat[:, col:col + width]
        arr[field[0]] = values if width > 1 else values[:, 0]
        col += width
    return arr


def write_points(path, xyz, normals=None, colors=None, scalars=None):
    """Binary little-endian point PLY: float x y z [nx ny nz] [uchar rgb] [float scalars]."""
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    for name in (scalars or {}):
        fields.append((name, "<f4"))
    arr = np.empty(n, dtype=np.dtype(fields))
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if normals is not None:
        arr["nx"], arr["ny"], arr["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    if colors is not None:
        arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    for name, values in (scalars or {}).items():
        arr[name] = values
    names = {"<f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property {names[t]} {name}" for name, t in fields]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(arr.tobytes())


_NAMES = {np.dtype(t).str[1:]: name for name, t in
          [("char", "i1"), ("uchar", "u1"), ("short", "i2"), ("ushort", "u2"), ("int", "i4"), ("uint", "u4"),
           ("float", "f4"), ("double", "f8")]}


def write_ply(path, elements, comments=()):
    """Binary little-endian PLY of {name: structured array} as returned by
    read_ply (list fields come with their "<name>_count" field)."""
    header = ["ply", "format binary_little_endian 1.0"] + [f"comment {c}" for c in comments]
    blobs = []
    for name, arr in elements.items():
        header.append(f"element {name} {len(arr)}")
        fields = []
        for field in arr.dtype.names:
            if field.endswith("_count") and field[:-6] in arr.dtype.names:
                continue
            sub = arr.dtype[field]
            if sub.shape:
                count = arr.dtype[field + "_count"]
                header.append(f"property list {_NAMES[count.str[1:]]} {_NAMES[sub.base.str[1:]]} {field}")
                fields += [(field + "_count", count.newbyteorder("<")), (field, sub.base.newbyteorder("<"), sub.shape)]
            else:
                header.append(f"property {_NAMES[sub.str[1:]]} {field}")
                fields.append((field, sub.newbyteorder("<")))
        out = np.empty(len(arr), dtype=np.dtype(fields))
        for field in out.dtype.names:
            out[field] = arr[field]
        blobs.append(out.tobytes())
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for blob in blobs:
            f.write(blob)


def positions(vertex):
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def triangles(ply):
    face = ply.get("face")
    if face is None:
        return np.zeros((0, 3), dtype=np.int64)
    name = "vertex_indices" if "vertex_indices" in face.dtype.names else "vertex_index"
    tri = np.asarray(face[name])
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("only triangle meshes are supported")
    return tri.astype(np.int64)
