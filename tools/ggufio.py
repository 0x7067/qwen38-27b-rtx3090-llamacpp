"""Minimal pure-Python GGUF v3 reader/writer helpers (no numpy, no gguf-py).

Only what the drafter-truncation surgery needs: read a header, keep the KV block
as an opaque byte range so metadata copies bit-identically, and lay out a new
tensor table.

Layout constraint that decides whether llama.cpp accepts the file
(ggml/src/gguf.cpp:779): tensor data offsets must equal the running sum of
GGML_PAD(nbytes, alignment) in tensor-info order. Info order and data order are
therefore the same list.
"""

import struct

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# ggml type id -> (name, type_size, block_size)
GGML_TYPES = {
    0:  ("F32",   4,  1),
    1:  ("F16",   2,  1),
    2:  ("Q4_0",  18, 32),
    3:  ("Q4_1",  20, 32),
    6:  ("Q5_0",  22, 32),
    7:  ("Q5_1",  24, 32),
    8:  ("Q8_0",  34, 32),
    10: ("Q2_K",  84, 256),
    11: ("Q3_K",  110, 256),
    12: ("Q4_K",  144, 256),
    13: ("Q5_K",  176, 256),
    14: ("Q6_K",  210, 256),
    24: ("I8",    1,  1),
    25: ("I16",   2,  1),
    26: ("I32",   4,  1),
    27: ("I64",   8,  1),
    28: ("F64",   8,  1),
    30: ("BF16",  2,  1),
}

GGML_TYPE_I64 = 27

# GGUF metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = 0, 1, 2, 3, 4, 5
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}


def nbytes(type_id, dims):
    name, type_size, block_size = GGML_TYPES[type_id]
    ne = 1
    for d in dims:
        ne *= d
    if ne % block_size:
        raise ValueError("tensor of %d elements is not a whole number of %s blocks" % (ne, name))
    return ne // block_size * type_size


def pad_to(x, alignment):
    return (x + alignment - 1) // alignment * alignment


class _Cursor:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt):
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))

    def string(self):
        (n,) = self.unpack("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t):
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            (et,) = self.unpack("<I")
            (n,) = self.unpack("<Q")
            return [self.value(et) for _ in range(n)]
        return self.unpack(_SCALAR_FMT[t])[0]


class Tensor:
    __slots__ = ("name", "dims", "type_id", "offset")

    def __init__(self, name, dims, type_id, offset):
        self.name = name
        self.dims = list(dims)
        self.type_id = type_id
        self.offset = offset

    @property
    def type_name(self):
        return GGML_TYPES[self.type_id][0]

    @property
    def nbytes(self):
        return nbytes(self.type_id, self.dims)

    def info_bytes(self):
        raw = self.name.encode("utf-8")
        out = [struct.pack("<Q", len(raw)), raw, struct.pack("<I", len(self.dims))]
        out += [struct.pack("<Q", d) for d in self.dims]
        out.append(struct.pack("<I", self.type_id))
        out.append(struct.pack("<Q", self.offset))
        return b"".join(out)

    def __repr__(self):
        return "Tensor(%r, %s, %s, off=%d)" % (self.name, self.dims, self.type_name, self.offset)


class GGUF:
    """Parsed GGUF header. `kv_raw` is the verbatim KV byte block."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            c = _Cursor(f)
            magic, version = c.unpack("<4sI")
            if magic != GGUF_MAGIC:
                raise ValueError("%s: not a GGUF file (magic %r)" % (path, magic))
            if version != GGUF_VERSION:
                raise ValueError("%s: unsupported GGUF version %d" % (path, version))
            (n_tensors,) = c.unpack("<Q")
            (n_kv,) = c.unpack("<Q")

            kv_start = f.tell()
            self.kv = {}
            for _ in range(n_kv):
                k = c.string()
                (t,) = c.unpack("<I")
                self.kv[k] = c.value(t)
            kv_end = f.tell()

            f.seek(kv_start)
            self.kv_raw = f.read(kv_end - kv_start)

            self.tensors = []
            for _ in range(n_tensors):
                name = c.string()
                (nd,) = c.unpack("<I")
                dims = [c.unpack("<Q")[0] for _ in range(nd)]
                (tt,) = c.unpack("<I")
                (off,) = c.unpack("<Q")
                self.tensors.append(Tensor(name, dims, tt, off))

            self.header_end = f.tell()

        self.n_kv = n_kv
        self.alignment = self.kv.get("general.alignment", DEFAULT_ALIGNMENT)
        self.data_start = pad_to(self.header_end, self.alignment)

    def tensor(self, name):
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError("%s: no tensor named %r" % (self.path, name))

    def abs_offset(self, name):
        return self.data_start + self.tensor(name).offset

    def check_offsets(self):
        """Mirror gguf.cpp's sequential-offset requirement. Returns list of errors."""
        errors = []
        running = 0
        for t in self.tensors:
            if t.offset != running:
                errors.append("%s: offset %d, expected %d" % (t.name, t.offset, running))
            running += pad_to(t.nbytes, self.alignment)
        return errors, running


def header_bytes(n_tensors, kv_raw, n_kv, tensors):
    out = [struct.pack("<4sI", GGUF_MAGIC, GGUF_VERSION),
           struct.pack("<Q", n_tensors),
           struct.pack("<Q", n_kv),
           kv_raw]
    out += [t.info_bytes() for t in tensors]
    return b"".join(out)


def layout(tensors, alignment):
    """Assign sequential, padded offsets in list order."""
    running = 0
    for t in tensors:
        t.offset = running
        running += pad_to(t.nbytes, alignment)
    return running
