# fcecodec_blender.py
# Copyright (C) 2024 and later Benjamin Futasz <https://github.com/bfut>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
fcecodec_blender.py - Blender (.fce) Import/Export Add-on
Supports Blender 4.x, 4.2 LTS, and 3.6 LTS on Windows, Linux, and macOS.

INSTALLATION:
    * Edit > Preferences > Add-ons > Install... > fcecodec_blender.py
        - requires an active internet connection while Blender installs Python modules "fcecodec" and "tinyobjloader" and "unvivtool"

USAGE:
    * File > Import > Need For Speed (.fce)
        - load selected (.fce) file and optional (.tga) texture file
        - alternatively, select (.viv) archive
            1. Hit "Select from (.viv)" button
            2. Select (.fce) file and optional (.tga) from the lists
            3. Hit "Import FCE" button

    * File > Export > Need For Speed (.fce)
        - export as (.fce) file
        - alternatively, select (.viv) archive
            1. Hit "Select from (.viv)" button
            2. Select (.fce) file from the list
            3. Hit "Export FCE" button

TUTORIAL:
    The following tutorial describes how to use Blender to:
        * create/modify damage models
        * edit part centers
        * set vertice animation flags
        * edit dummies (light / fx objects)
        * edit triangle flags
        * edit texture pages

    https://github.com/bfut/fcecodec/tree/main/scripts/doc_Obj2Fce.md
"""

bl_info = {
    "name": "fcecodec_blender",
    "author": "Benjamin Futasz",
    "version": (3, 4),
    "blender": (3, 6, 0),
    "location": "File > Import/Export > Need For Speed (.fce)",
    "description": "Imports & Exports Need For Speed (.fce) files, powered by fcecodec",
    "category": "Import-Export",
    "url": "https://github.com/bfut/fcecodec_blender",
}
DEV_MODE = False

import colorsys
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

def pip_install(package, upgrade=False, pre=False, version: str | None = None):
    _call = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        _call.append("-U")
    if pre:
        _call.append("--pre")
    if version:
        _call.append(f"{package}>={version}")
    else:
        _call.append(package)
    subprocess.check_call(_call)

min_fcecodec_version = "1.7"
try:
    import fcecodec as fc
    if fc.__version__ < min_fcecodec_version:
        raise ImportError
except ImportError:
    pip_install("fcecodec", upgrade=True, version=min_fcecodec_version)
    import fcecodec as fc

min_unvivtool_version = "3.0"
try:
    import unvivtool as uvt
    if uvt.__version__ < min_fcecodec_version:
        raise ImportError
except ImportError:
    pip_install("unvivtool", upgrade=True, version=min_unvivtool_version)
    import unvivtool as uvt

try:
    import numpy as np
except ImportError:
    pip_install("numpy")
    import numpy as np

try:
    import tinyobjloader
except ImportError:
    pip_install("tinyobjloader", upgrade=True, pre=True)
    import tinyobjloader

import bpy
from bpy.props import (StringProperty,
                       BoolProperty,
                       EnumProperty,
                       IntProperty,
                       IntVectorProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       CollectionProperty,
                       PointerProperty)
from bpy.types import (Operator,
                       PropertyGroup,
                       UIList)
from bpy_extras.io_utils import ImportHelper, ExportHelper


# wrappers
def ReorderTriagsTransparentDetachedAndToLast(mesh, pid_opaq):
    """ Copy original part, delete semi-transparent triags in original,
        delete opaque triags in copy, clean-up unreferenced verts, merge parts,
        delete temporary & original, move merged part to original index """
    pid_transp = mesh.OpCopyPart(pid_opaq)
    mesh.OpDeletePartTriags(pid_opaq, np.where(mesh.PGetTriagsFlags(pid_opaq) & 0x8 == 0x8)[0])
    mesh.OpDeletePartTriags(pid_transp, np.where(mesh.PGetTriagsFlags(pid_transp) & 0x8 < 0x8)[0])
    mesh.OpDelUnrefdVerts()
    new_pid = mesh.OpMergeParts(pid_opaq, pid_transp)  # last part idx
    mesh.PSetName(new_pid, mesh.PGetName(pid_opaq))
    mesh.OpDeletePart(pid_transp)
    mesh.OpDeletePart(pid_opaq)
    new_pid -= 2  # last part idx is now smaller
    while new_pid > pid_opaq:
        new_pid = mesh.OpMovePart(new_pid)  # move merged part to original index
    return mesh

def HiBody_ReorderTriagsTransparentToLast(mesh, version):
    """ Not implemented for FCE4M because windows are separate parts """
    if version in ("3", 3):
        mesh = ReorderTriagsTransparentDetachedAndToLast(mesh, 0)  # high body
        if mesh.MNumParts > 12:
            mesh = ReorderTriagsTransparentDetachedAndToLast(mesh, 12)  # high headlights
    elif version in ("4", 4):
        for partname in (":HB", ":OT", ":OL"):
            pid = GetMeshPartnameIdx(mesh, partname)
            if pid >= 0:
                mesh = ReorderTriagsTransparentDetachedAndToLast(mesh, pid)
    return mesh

def GetFceVersion(path):
    with open(path, "rb") as f:
        version = fc.GetFceVersion(f.read(0x2038))
        assert version > 0
        return version

def PrintFceInfo(path):
    with open(path, "rb") as f:
        buf = f.read()
        fc.PrintFceInfo(buf)
        # assert fc.ValidateFce(buf) == 1

def LoadFce(mesh, path):
    with open(path, "rb") as f:
        mesh.IoDecode(f.read())
        assert mesh.MValid() is True
        return mesh

def WriteFce(version, mesh, path, center_parts=False, mesh_function=None):
    if mesh_function is not None:  # e.g., HiBody_ReorderTriagsTransparentToLast
        mesh = mesh_function(mesh, version)
    with open(path, "wb") as f:
        if version in ("3", 3):
            buf = mesh.IoEncode_Fce3(center_parts)
        elif version in ("4", 4):
            buf = mesh.IoEncode_Fce4(center_parts)
        else:
            buf = mesh.IoEncode_Fce4M(center_parts)
        assert fc.ValidateFce(buf) == 1
        f.write(buf)

def ExportObj(mesh, objpath, mtlpath, texname,
              print_damage, print_dummies, use_part_positions, print_part_positions,
              filter_triagflags_0xfff=True):
    mesh.IoExportObj(str(objpath), str(mtlpath), str(texname),
                     print_damage, print_dummies, use_part_positions, print_part_positions,
                     filter_triagflags_0xfff)

def GetMeshPartnames(mesh):
    return [mesh.PGetName(pid) for pid in range(mesh.MNumParts)]

def GetMeshPartnameIdx(mesh, partname):
    for pid in range(mesh.MNumParts):
        if mesh.PGetName(pid) == partname:
            return pid
    print(f"GetMeshPartnameIdx: Warning: cannot find '{partname}'")
    return -1

def GetPartGlobalOrderVidxs(mesh, pid):
    map_verts = mesh.MVertsGetMap_idx2order
    part_vidxs = mesh.PGetTriagsVidx(pid)
    for i in range(part_vidxs.shape[0]):
        # print(part_vidxs[i], map_verts[part_vidxs[i]])
        part_vidxs[i] = map_verts[part_vidxs[i]]
    return part_vidxs

def FilterTexpageTriags(mesh, drop_texpages: int | list | None = None, select_texpages: int | list | None = None):
    # Delete triangles with texpage in drop_texpages
    if drop_texpages is not None and select_texpages is None:
        if not isinstance(drop_texpages, list):
            drop_texpages = [drop_texpages]
        for pid in reversed(range(mesh.MNumParts)):
            texp = mesh.PGetTriagsTexpages(pid)
            mask = np.isin(texp, drop_texpages)
            x = mask.nonzero()[0]
            assert mesh.OpDeletePartTriags(pid, x)

    # Delete triangles with texpage not in select_texpages
    elif drop_texpages is None and select_texpages is not None:
        print(f"FilterTexpageTriags: select_texpages={select_texpages}")
        if not isinstance(select_texpages, list):
            select_texpages = [select_texpages]
        for pid in reversed(range(mesh.MNumParts)):
            print(f"before mesh.PNumTriags(pid)={mesh.PNumTriags(pid)}")
            texp = mesh.PGetTriagsTexpages(pid)
            mask = np.isin(texp, select_texpages)
            mask = np.invert(mask)
            x = mask.nonzero()[0]
            print(mesh.PGetName(pid), texp.shape, x.shape, np.unique(texp))
            assert mesh.OpDeletePartTriags(pid, x)
            print(f"after: mesh.PNumTriags(pid)={mesh.PNumTriags(pid)}")

    else:
        raise ValueError("FilterTexpageTriags: call with either drop_texpages or select_texpages, not both")

    # assert mesh.OpDelUnrefdVerts()
    return mesh

def DeleteEmptyParts(mesh):
    for pid in reversed(range(mesh.MNumParts)):
        if mesh.PNumTriags(pid) == 0:
            mesh.OpDeletePart(pid)
    return mesh


#########################################################
# -------------------------------------- bfut_Obj2Fce3


# -------------------------------------- tinyobjloader wrappers
def LoadObj(filename):
    """src: https://github.com/tinyobjloader/tinyobjloader/blob/master/python/sample.py"""
    reader = tinyobjloader.ObjReader()
    config = tinyobjloader.ObjReaderConfig()
    config.triangulate = False
    ret = reader.ParseFromFile(str(filename), config)
    if ret is False:
        print("Failed to load : ", filename)
        print("Warn:", reader.Warning())
        print("Err:", reader.Error())
        sys.exit(-1)
    if reader.Warning():
        print("Warn:", reader.Warning())
    return reader

def GetVerts(reader):  # xyzxyzxyz
    attrib = reader.GetAttrib()
    arr = attrib.numpy_vertices()
    print(f"GetVerts: {type(arr)}, {arr.shape}, {arr.ndim} min,max={np.min(arr)};{np.max(arr)}")
    return arr

def GetNormals(reader):  # xyzxyzxyz
    attrib = reader.GetAttrib()
    arr = np.array(attrib.normals)
    if arr.shape[0] == 0:
        print(f"GetNormals: {type(arr)}, {arr.shape}, {arr.ndim} min,max=n/a;n/a")
    else:
        print(f"GetNormals: {type(arr)}, {arr.shape}, {arr.ndim} min,max={np.min(arr)};{np.max(arr)}")
    return arr

def GetTexcoords(reader):  # uvuvuv
    attrib = reader.GetAttrib()
    arr = np.array(attrib.texcoords)
    if arr.shape[0] == 0:
        print(f"GetTexcoords: {type(arr)}, {arr.shape}, {arr.ndim} min,max=n/a;n/a")
    else:
        print(f"GetTexcoords: {type(arr)}, {arr.shape}, {arr.ndim} min,max={np.min(arr)};{np.max(arr)}")
    return arr

def PrintShapes(reader):
    shapes = reader.GetShapes()
    print("Num shapes: ", len(shapes))
    for shape in shapes:
        print(shape.name,
              f"faces={int(shape.mesh.numpy_indices().shape[0] / (3*3))}")

def GetShapeNames(reader):
    shapenames = []
    shapes = reader.GetShapes()
    for i in range(len(shapes)):
        shapenames += [shapes[i].name]
    return shapenames

def GetShapeFaces(reader, vertices, normals, texcoords, shapename):
    shape = None
    shapes = reader.GetShapes()
    for s in shapes:
        if s.name == shapename:
            shape = s
            break
    if shape is None:
        print("GetShapeFaces: cannot find specified shapename", shapename)
        return None
    s_NumFaces = int(shape.mesh.numpy_indices()[0::3].shape[0] / 3)

    s_faces = shape.mesh.numpy_indices()[0::3]
    normals_idxs = shape.mesh.numpy_indices()[1::3]
    texcoord_idxs = shape.mesh.numpy_indices()[2::3]

    print(shape.name, f"faces={int(s_faces.shape[0] / 3)}")
    print(shape.name, f"normals_idxs={int(normals_idxs.shape[0])} min,max={np.min(normals_idxs)};{np.max(normals_idxs)}")
    print(shape.name, f"texcoord_idxs={int(texcoord_idxs.shape[0])} min,max={np.min(texcoord_idxs)};{np.max(texcoord_idxs)}")

    # cannot use np.unique(), as shape may have unreferenced verts
    # example: mcf1/car.viv->car.fce :HB
    vert_selection = np.arange(np.amin(s_faces), np.amax(s_faces) + 1)
    s_verts = vertices.reshape(-1, 3)[ vert_selection ].flatten()

    # Get normals (use vert positions, if no normals for shape)
    # obj: number of verts and normals may differ; fce: each vert has a normal
    print("GetShapeFaces: Get normals")
    norm_selection = np.empty(vert_selection.shape[0], dtype=int)
    map_v_t = np.copy(vert_selection)
    print(f"GetShapeFaces: for i in range{map_v_t.shape[0]}")
    for i in range(map_v_t.shape[0]):
        argwhere = np.argwhere(s_faces == map_v_t[i])
        if len(argwhere) == 0:
            map_v_t[i] = -1
        else:
            map_v_t[i] = argwhere[0]
    print(f"GetShapeFaces: for i in range{map_v_t.shape[0]}")
    for i in range(map_v_t.shape[0]):
        if map_v_t[i] < 0:
            norm_selection[i] = np.copy(vert_selection[i])
        else:
            norm_selection[i] = np.copy(normals_idxs[map_v_t[i]])
    if np.amax(norm_selection) <= int(normals.shape[0] / 3):
        print("norm_selection")
        s_norms = normals.reshape(-1, 3)[ norm_selection ].flatten()  # normals[normals_idxs]
    else:
        print("shape has no normals... use vert positions as normals")
        s_norms = np.copy(s_verts)

    # Get tex coordinate (set 0.0f, if not enough texcoords)
    print("GetShapeFaces: uvuvuv... -> uuuvvv...")
    s_texcs = np.zeros(s_NumFaces * 6, dtype=float)
    if texcoord_idxs.shape[0] == s_NumFaces * 3:
        for i in range(s_NumFaces):
            for j in range(3):
                s_texcs[i*6 + 0*3 + j] = texcoords[texcoord_idxs[i*3 + j] * 2 + 0]
                s_texcs[i*6 + 1*3 + j] = texcoords[texcoord_idxs[i*3 + j] * 2 + 1]
    else:
        print(f"shape has missing texcoords... set 0.0f (texcoord_idxs.shape {texcoord_idxs.shape[0]} != {s_NumFaces * 6} s_NumFaces*6)")

    s_matls = shape.mesh.numpy_material_ids()

    return s_faces, s_verts, s_norms, s_texcs, s_matls


# -------------------------------------- more wrappers
def LocalizeVertIdxs(faces):
    return faces - np.amin(faces)

def GetFlagFromTags(tags):
    flag = 0x0
    for t in tags:
        # if t == "FDEF":
        #     flag += 0x000
        if t == "FMAT":
            flag += 0x001
        elif t == "FHIC":
            flag += 0x002
        elif t == "FNOC":
            flag += 0x004
        elif t == "FSET":
            flag += 0x008
        elif t == "FUN1":
            flag += 0x010
        elif t == "FALW":
            flag += 0x020
        elif t == "FFRW":
            flag += 0x040
        elif t == "FLEW":
            flag += 0x080
        elif t == "FBAW":
            flag += 0x100
        elif t == "FRIW":
            flag += 0x200
        elif t == "FBRW":
            flag += 0x400
        elif t == "FUN2":
            flag += 0x800
    return flag

def GetTexPageFromTags(tags):
    txp = 0x0
    r = re.compile("T[0-9]", re.IGNORECASE)
    for t in tags:
        if r.match(t) is not None:
            try:
                txp = int(t[1:])
            except ValueError:
                print(f"Cannot convert tag {t} to texpage")
    return txp

def ShapeToPart(reader,
                mesh, objverts, objnorms, objtexcoords, request_shapename,
                material2texpage, material2triagflag):
    s_faces, s_verts, s_norms, s_texcs, s_matls = GetShapeFaces(reader,
        objverts, objnorms, objtexcoords, request_shapename)
    print(f"faces:{int(s_faces.shape[0] / 3)}")
    print(f"vert idx range: [{np.amin(s_faces)},{np.amax(s_faces)}]")
    print(f"vertices:{int(s_verts.shape[0] / 3)}")
    print(f"normals:{s_norms.shape[0]}")
    print(f"texcoords:{s_texcs.shape[0]}->{int(s_texcs.shape[0] / 6)}")

    # print(s_faces)
    s_faces = LocalizeVertIdxs(s_faces)

    s_verts[2::3] = -s_verts[2::3]  # flip sign in Z-coordinate
    s_norms[2::3] = -s_norms[2::3]  # flip sign in Z-coordinate
    mesh.IoGeomDataToNewPart(s_faces, s_texcs, s_verts, s_norms)
    mesh.PSetName(mesh.MNumParts - 1, request_shapename)  # shapename to partname

    # map faces material IDs to triangles texpages
    if material2texpage == 1:
        print("mapping faces material names to triangles texpages...")
        num_arts_warning = False
        materials = reader.GetMaterials()
        texps = mesh.PGetTriagsTexpages(mesh.MNumParts - 1)
        for i in range(texps.shape[0]):
            # Blender may change the name of the material to "<name>.001" from "<name>"
            mat_ = materials[s_matls[i]].name
            mat_ = re.sub(r"\.(.*)", "", mat_)
            if mat_[:2] == "0x":
                texps[i] = int(mat_, base=16)
                # print(f"{mat_} -> {texps[i]} (0x{texps[i]:0x})")
            else:
                tags = mat_.split("_")
                texps[i] = GetTexPageFromTags(tags)
                # print(f"{mat_} -> {tags} -> {texps[i]} (0x{texps[i]:0x})")
            if texps[i] > 0:
                num_arts_warning = True
        # print(type(texps), texps.dtype, texps.shape)
        if num_arts_warning:
            print("Warning: texpage greater than zero is present. FCE3/FCE4 require amending Mesh.NumArts value")
        mesh.PSetTriagsTexpages(mesh.MNumParts - 1, texps)

    # map faces material names to triangles flags iff
    # all material names are integer hex values (strings of the form '0xiii')
    if material2triagflag == 1:
        print("mapping faces material names to triangles flags...")
        materials = reader.GetMaterials()
        tflags = mesh.PGetTriagsFlags(mesh.MNumParts - 1)

        # if material name is hex value, map straight to triag flag
        # if it isn't, treat as string of tags
        for i in range(tflags.shape[0]):
            # Blender may change the name of the material to "<name>.001" from "<name>"
            mat_ = materials[s_matls[i]].name
            mat_ = re.sub(r"\.(.*)", "", mat_)
            if mat_[:2] == "0x":
                tflags[i] = int(mat_, base=16)
                # print(f"{mat_} -> {tflags[i]} (0x{tflags[i]:0x})")
            else:
                tags = mat_.split("_")
                tflags[i] = GetFlagFromTags(tags)
                # print(f"{mat_} -> {tags} -> {tflags[i]} (0x{tflags[i]:0x})")
        mesh.PSetTriagsFlags(mesh.MNumParts - 1, tflags)

    return mesh

def CopyDamagePartsVertsToPartsVerts(mesh):
    """
    Copy verts/norms of DAMAGE_<partname> to damaged verts/norms of <partname>
    """
    damgd_pids = []
    part_names = np.array(GetMeshPartnames(mesh), dtype="U64")
    mesh.PrintInfo()
    for damgd_pid in range(mesh.MNumParts):
        if part_names[damgd_pid][:7] == "DAMAGE_":
            damgd_pids += [damgd_pid]

            pid = np.argwhere(part_names == part_names[damgd_pid][7:])
            # print(damgd_pid, part_names[damgd_pid], pid, len(pid))
            if len(pid) < 1:
                continue
            pid = pid[0][0]

            if mesh.PNumTriags(damgd_pid) != mesh.PNumTriags(pid):
                print(f"discarding '{part_names[damgd_pid]}', because number of triags differ to {part_names[pid]}")
                continue
            if mesh.PNumVerts(damgd_pid) != mesh.PNumVerts(pid):
                print(f"discarding '{part_names[damgd_pid]}', because number of verts differ to {part_names[pid]}")
                continue

            print(f"copy verts/norms of {part_names[damgd_pid]} to damaged verts/norms of {part_names[pid]}")
            damgd_part_vidxs = GetPartGlobalOrderVidxs(mesh, damgd_pid)
            part_vidxs = GetPartGlobalOrderVidxs(mesh, pid)

            # cannot use np.unique(), as shape may have unreferenced verts
            # example: mcf1/car.viv->car.fce :HB
            # requires that OBJ parts verts are ordered in non-overlapping
            # ranges and that verts 0 and last, resp., are referenced
            damgd_part_vidxs = np.arange(np.amin(damgd_part_vidxs), np.amax(damgd_part_vidxs) + 1)
            part_vidxs = np.arange(np.amin(part_vidxs), np.amax(part_vidxs) + 1)

            dn = mesh.MVertsDamgdNorms.reshape((-1, 3))
            dv = mesh.MVertsDamgdPos.reshape((-1, 3))
            dn[part_vidxs] = mesh.MVertsNorms.reshape((-1, 3))[damgd_part_vidxs]
            dv[part_vidxs] = mesh.MVertsPos.reshape((-1, 3))[damgd_part_vidxs]
            mesh.MVertsDamgdNorms = dn.flatten()
            mesh.MVertsDamgdPos = dv.flatten()
    for i in sorted(damgd_pids, reverse=True):
        mesh.OpDeletePart(i)
    print(damgd_pids)
    return mesh

def PartsToDummies(mesh):
    """
    From shapes named DUMMY_##_<dummyname> create dummies at centroids.
    """
    pids = []
    dms = []
    dms_pos = np.empty(0)
    r = re.compile("DUMMY_[0-9][0-9]_", re.IGNORECASE)
    for i in range(mesh.MNumParts):
        if r.match(mesh.PGetName(i)[:9]) is not None:
            print(f"convert part {i} '{mesh.PGetName(i)}' to dummy {mesh.PGetName(i)[9:]}")
            pids += [i]
            dms += [mesh.PGetName(i)[9:]]
            mesh.OpCenterPart(i)
            dms_pos = np.append(dms_pos, mesh.PGetPos(i))
    for i in reversed(pids):
        mesh.OpDeletePart(i)
    print(pids, dms, dms_pos)
    mesh.MSetDummyNames(dms)
    mesh.MSetDummyPos(dms_pos)
    return mesh

def SetAnimatedVerts(mesh):
    """
    Set <partname> verts movable iff contained in ANIMATED_##_<partname> cuboid
    hull, where # is digit
    """
    print("SetAnimatedVerts(mesh):")
    vpos = mesh.MVertsPos.reshape((-1, 3))
    animation_flags = mesh.MVertsAnimation
    anim_pids = []
    part_names = GetMeshPartnames(mesh)
    r = re.compile("ANIMATED_[0-9][0-9]_", re.IGNORECASE)
    for i, part_name in zip(range(mesh.MNumParts), part_names):
        part_anim_pids = []
        if r.match(part_name[:12]) is None:
            print(i, part_name)
            for j, anim_name in zip(range(mesh.MNumParts), part_names):
                if anim_name[12:] == part_name and r.match(anim_name[:12]) is not None:
                    anim_pids += [j]
                    part_anim_pids += [j]
                    print("  ", j, anim_name)
            print("animation flag maps:", part_anim_pids)
            if len(part_anim_pids) > 0:
                part_vidxs = GetPartGlobalOrderVidxs(mesh, i)
                # cannot use np.unique(), as shape may have unreferenced verts
                # example: mcf1/car.viv->car.fce :HB
                # requires that OBJ parts verts are ordered in non-overlapping
                # ranges and that verts 0 and last, resp., are referenced
                part_vidxs = np.arange(np.amin(part_vidxs), np.amax(part_vidxs) + 1)
                part_animation_flags = animation_flags[part_vidxs]
                part_vpos = vpos[part_vidxs]
                part_animation_flags[:] = 0x4
                print(part_vidxs.shape, part_animation_flags.shape)
                for j in part_anim_pids:
                    part_anim_vidxs = GetPartGlobalOrderVidxs(mesh, j)
                    part_anim_vidxs = np.arange(np.amin(part_anim_vidxs), np.amax(part_anim_vidxs) + 1)
                    anim_vpos = vpos[part_anim_vidxs]
                    print(anim_vpos.shape, np.amin(anim_vpos, axis=0), np.amax(anim_vpos, axis=0))
                    cuboid_min = np.amin(anim_vpos, axis=0)
                    cuboid_max = np.amax(anim_vpos, axis=0)
                    for n in range(part_vpos.shape[0]):
                        # part_vpos is ndarray, but make static analysis happy
                        if False not in np.array(part_vpos[n] > cuboid_min) \
                        and False not in np.array(part_vpos[n] < cuboid_max):
                            part_animation_flags[n] = 0x0
                            print(n, part_vpos[n], part_animation_flags[n])
                    animation_flags[part_vidxs] = part_animation_flags
                print(np.unique(animation_flags[part_vidxs]))
    print(anim_pids, np.unique(animation_flags))
    mesh.MVertsAnimation = animation_flags
    for i in sorted(anim_pids, reverse=True):
        mesh.OpDeletePart(i)
    return mesh

def CenterParts(mesh):
    """
    Center part <partname> either to centroid, or if present to centroid of part POSITION_<partname>
    """
    pos_pids = []
    part_names = GetMeshPartnames(mesh)
    pos_parts = {}
    for name, i in zip(part_names, range(mesh.MNumParts)):
        if name[:9] == "POSITION_":
            pos_parts[name[9:]] = i
            pos_pids += [i]
    print(pos_parts)
    for pid in range(mesh.MNumParts):
        if part_names[pid][:9] != "POSITION_":
            if part_names[pid] not in pos_parts:  # POSITION_<partname> not available
                print(f"center {part_names[pid]} to local centroid")
                mesh.OpCenterPart(pid)
            else:                                 # POSITION_<partname> is available
                pos_pid = pos_parts[part_names[pid]]
                print(f"center {part_names[pid]} to centroid of {part_names[pos_pid]}")
                mesh.OpCenterPart(pos_pid)
                mesh.OpSetPartCenter(pid, mesh.PGetPos(pos_pid))
    for i in sorted(pos_pids, reverse=True):
        mesh.OpDeletePart(i)
    return mesh

def FixPartDummyNames(mesh):
    """
    Change any "<name>.001" -> "<name>"

    Blender may export partnames/dummynames such as "<name>.001"
    """
    dm = mesh.MGetDummyNames()
    for i in range(len(dm)):
        tmp_1 = dm[i]
        tmp_ = re.sub(r"\.(.*)", "", tmp_1)
        print(f"{tmp_1}->{tmp_}")
        dm[i] = tmp_
    mesh.MSetDummyNames(dm)
    for pid in range(mesh.MNumParts):
        tmp_1 = mesh.PGetName(pid)
        tmp_ = re.sub(r"\.(.*)", "", tmp_1)
        print(f"{tmp_1}->{tmp_}")
        mesh.PSetName(pid, tmp_)
    return mesh


#
def workload_Obj2Fce(filepath_obj_input, filepath_fce_output, CONFIG):
    print(CONFIG)

    # Import OBJ
    reader = LoadObj(filepath_obj_input)
    attrib = reader.GetAttrib()
    print(f"attrib.vertices = {len(attrib.vertices)}, {int(len(attrib.vertices) / 3)}")
    print(f"attrib.normals = {len(attrib.normals)}")
    print(f"attrib.texcoords = {len(attrib.texcoords)}")
    objverts = GetVerts(reader)
    objnorms = GetNormals(reader)
    objtexcoords = GetTexcoords(reader)
    PrintShapes(reader)
    shapenames = GetShapeNames(reader)

    # Transform geometric data, load as fc.Mesh
    mesh = fc.Mesh()
    for i in range(len(shapenames)):
        print("s_name", shapenames[i])
        mesh = ShapeToPart(reader,
                           mesh, objverts, objnorms, objtexcoords, shapenames[i],
                           CONFIG["material2texpage"], CONFIG["material2triagflag"])
    mesh = FixPartDummyNames(mesh)
    mesh = CopyDamagePartsVertsToPartsVerts(mesh)
    mesh = PartsToDummies(mesh)
    mesh = SetAnimatedVerts(mesh)
    if CONFIG["center_parts"] == 1:
        mesh = CenterParts(mesh)
    if CONFIG["normals2vertices"] == 1:
        # replace verts with normals, preserve part positions
        mesh.MVertsPos = mesh.MVertsNorms
        mesh.MVertsDamgdPos = mesh.MVertsDamgdNorms

    # Write FCE
    WriteFce(CONFIG["fce_version"], mesh, filepath_fce_output, center_parts=False)
    print(flush=True)
    # PrintFceInfo(filepath_fce_output)
    print(f"filepath_fce_output={filepath_fce_output}")

# -------------------------------------- bfut_Obj2Fce3
#########################################################


#########################################################
# -------------------------------------- bfut_SortPartsToFce3Order

# -------------------------------------- script functions
def PrintMeshParts_order(mesh, part_names_sorted):
    print("pid  IS                          SHOULD")
    for pid in range(mesh.MNumParts):
        print(f"{pid:<2} {mesh.PGetName(pid):<12} {part_names_sorted[pid]:<12}")

def AssertPartsOrder(mesh, part_names_sorted, onlywarn=False):
    for pid in range(mesh.MNumParts):
        if mesh.PGetName(pid) != part_names_sorted[pid]:
            PrintMeshParts_order(mesh, part_names_sorted)
            if not onlywarn:
                raise AssertionError (f"pid={pid} {mesh.PGetName(pid)} != {part_names_sorted[pid]}")
            print(f"AssertPartsOrder: Warning pid={pid} {mesh.PGetName(pid)} != {part_names_sorted[pid]}")


def workload_SortPartsToFce3Order(filepath_fce_input, fce_outversion):
    filepath_fce_output = filepath_fce_input

    mesh = fc.Mesh()
    mesh = LoadFce(mesh, filepath_fce_input)

    # sort
    if mesh.MNumParts > 1:
        priority_dic = {  # NB1: front wheel order differs for high body/medium body
            "high body": 0,
            "left front wheel": 1,
            "right front wheel": 2,
            "left rear wheel": 3,
            "right rear wheel": 4,
            "medium body": 5,
            "medium r front wheel": 6,
            "medium l front wheel": 7,
            "medium r rear wheel": 8,
            "medium l rear wheel": 9,
            "small body": 10,
            "tiny body": 11,
            "high headlights": 12,

            ":HB": 0,
            ":HLFW": 1,
            ":HRFW": 2,
            ":HLRW": 3,
            ":HRRW": 4,
            ":MB": 5,
            ":MRFW": 6,
            ":MLFW": 7,
            ":MRRW": 8,
            ":MLRW": 9,
            ":LB": 10,
            ":TB": 11,
            ":OL": 12,

            ":Hbody": 0,
        }

        part_names = []
        for pid in reversed(range(mesh.MNumParts)):
            part_names += [mesh.PGetName(pid)]
        part_names_sorted = sorted(part_names, key=lambda x: priority_dic.get(x, 64))

        for target_idx in range(0, len(part_names_sorted)):
            pname = part_names_sorted[target_idx]
            current_idx = GetMeshPartnameIdx(mesh, pname)
            # print(f" {pname} {current_idx} -> {target_idx}")
            while current_idx > target_idx:
                current_idx = mesh.OpMovePart(current_idx)
        AssertPartsOrder(mesh, part_names_sorted, onlywarn=True)
        # PrintMeshParts(mesh, part_names_sorted)


    WriteFce(fce_outversion, mesh, filepath_fce_output)
    PrintFceInfo(filepath_fce_output)
    # print(f"OUTPUT = {filepath_fce_output}", flush=True)

# -------------------------------------- bfut_SortPartsToFce3Order
#########################################################


#########################################################
# -------------------------------------- bfut_ConvertDummies (to Fce3) (Fce3 to Fce4)

def GetDummies(mesh):
    dms_pos = mesh.MGetDummyPos()
    dms_pos = np.reshape(dms_pos, (int(dms_pos.shape[0] / 3), 3))
    dms_names = mesh.MGetDummyNames()
    return dms_pos, dms_names

def SetDummies(mesh, dms_pos, dms_names):
    dms_pos = np.reshape(dms_pos, (int(dms_pos.shape[0] * 3)))
    dms_pos = dms_pos.astype("float32")
    mesh.MSetDummyPos(dms_pos)
    mesh.MSetDummyNames(dms_names)
    return mesh

def DummiesToFce3(dms_pos, dms_names):
    for i in range(len(dms_names)):
        x = dms_names[i]
        if len(str(x)) < 1:
            pass
        # """
        # elif x[0] in [":", "B", "I", "M", "P", "R"]:
        #     print(x, "->", dms_names[i])
        #     continue
        # # """
        elif x[:4] in ["HFLO", "HFRE", "HFLN", "HFRN", "TRLO", "TRRE", "TRLN",
                     "TRRN", "SMLN", "SMRN"]:
            pass  # keep canonical FCE3 names
        elif x[0] == "B":
            # dms_names[i] = "TRLO"  # convert brake lights to taillights?
            pass
        elif x[0] in ["H", "I"]:
            if x[3] == "O":
                dms_names[i] = "HFLO"
            elif x[3] == "E":
                dms_names[i] = "HFRE"
            elif dms_pos[i, 0] < 0:  # left-hand
                dms_names[i] = "HFLN"
            else:
                dms_names[i] = "HFRN"
            """
            # T - fce4 taillights seemingly work for fce3
            elif x[0] == "T":
                if x[3] == "O":
                    dms_names[i] = "TRLO"
                elif x[3] == "E":
                    dms_names[i] = "TRRE"
                elif dms_pos[i, 0] < 0:  # left-hand
                    dms_names[i] = "TRLN"
                else:
                    dms_names[i] = "TRRN"
            # """
        elif x[0] == "S":
            if x[1] == "B":
                dms_names[i] = "SMLN"  # blue
            else:
                dms_names[i] = "SMRN"  # red
        print(x, "->", dms_names[i])
    return dms_pos, dms_names

def workload_ConvertDummies_to_Fce3_(filepath_fce_input, fce_outversion):
    filepath_fce_output = filepath_fce_input

    mesh = fc.Mesh()
    mesh = LoadFce(mesh, filepath_fce_input)

    dms_pos, dms_names = GetDummies(mesh)
    dms_pos, dms_names = DummiesToFce3(dms_pos, dms_names)
    mesh = SetDummies(mesh, dms_pos, dms_names)

    WriteFce(fce_outversion, mesh, filepath_fce_output)

def DummiesFce3ToFce4(dms_pos, dms_names):
    for i in range(len(dms_names)):
        x = dms_names[i]
        tmp = []
        if len(str(x)) < 1:
            pass
        elif bool(re.search(r"\d", x)) or x[0] == ":":  # name contains integer
            pass  # do not convert canonical FCE4/FCE4M names
        elif x[0] == "H":
            tmp.append("HWY")  # kind, color, breakable
            if len(x) > 3:
                tmp.append(x[3])  # flashing
            else:
                tmp.append("N")
            tmp.append("5")  # intensity
            if len(x) > 3 and x[3] != "N":
                tmp.append("5")  # time
                if x[2] == "L":
                    tmp.append("0")  # delay
                else:  # == "R"
                    tmp.append("5")  # delay
            dms_names[i] = "".join(tmp)
        elif x[0] == "T":
            tmp.append("TRYN5")  # kind, color, breakable, flashing, intensity
            dms_names[i] = "".join(tmp)
        elif x[0] == "M":
            tmp.append("S")  # kind
            if len(x) > 2 and x[2] == "L":  # left
                tmp.append("RNO535")  # color, breakable, flashing, intensity, time, delay
            else:  # == "R"  # right
                tmp.append("BNE530")
            dms_names[i] = "".join(tmp)
        print(x, "->", dms_names[i])
    return dms_pos, dms_names

def workload_ConvertDummies_Fce3_to_Fce4_(filepath_fce_input, fce_outversion):
    filepath_fce_output = filepath_fce_input

    mesh = fc.Mesh()
    mesh = LoadFce(mesh, filepath_fce_input)

    dms_pos, dms_names = GetDummies(mesh)
    dms_pos, dms_names = DummiesFce3ToFce4(dms_pos, dms_names)
    mesh = SetDummies(mesh, dms_pos, dms_names)

    WriteFce(fce_outversion, mesh, filepath_fce_output)

# -------------------------------------- bfut_ConvertDummies (to Fce3) (Fce3 to Fce4)
#########################################################


#########################################################
# -------------------------------------- bfut_ConvertPartnames (Fce4 to Fce4M) (Fce3 to Fce4)

def Partnames3to4_car(mesh):
    # car.fce
    # NB1: front wheel order is different for high body/medium body
    pnames_map = [
        ":HB",  # high body
        ":HLFW",  # left front wheel
        ":HRFW",  # right front wheel
        ":HLRW",  # left rear wheel
        ":HRRW",  # right rear wheel
        ":MB",  # medium body
        ":MRFW",  # medium r front wheel
        ":MLFW",  # medium l front wheel
        ":MRRW",  # medium r rear wheel
        ":MLRW",  # medium l rear wheel
        ":LB",  # small body
        ":TB",  # tiny body
        ":OL",  # high headlights
    ]
    for pid in range(min(len(pnames_map), mesh.MNumParts)):
        mesh.PSetName(pid, pnames_map[pid])
        print(f"renaming part {pid} -> '{pnames_map[pid]}'")
    return mesh

def Partnames4to4M_car(mesh):
    # car.fce
    # FCE4M loads meshes for wheels, drivers, and enhanced parts from central files.
    pnames_map = {
        ":HB": ":Hbody",
        # ":MB": ,
        # ":LB": ,
        # ":TB": ,
        ":OT": ":Hconvertible",
        ":OL": ":Hheadlight",
        ":OS": ":PPspoiler",
        ":OLB": ":Hlbrake",
        ":ORB": ":Hrbrake",
        ":OLM": ":Hlmirror",
        ":ORM": ":Hrmirror",
        ":OC": ":Hinterior",
        ":ODL": ":Hdashlight",
        # ":OH": ,
        ":OD": ":PPdriver",
        # ":OND": ,
        ":HLFW": ":PPLFwheel",
        ":HRFW": ":PPRFwheel",
        # ":HLMW": ,
        # ":HRMW": ,
        ":HLRW": ":PPLRwheel",
        ":HRRW": ":PPRRwheel",
        # ":MLFW": ,
        # ":MRFW": ,
        # ":MLMW": ,
        # ":MRMW": ,
        # ":MLRW": ,
        # ":MRRW": ,
    }
    for pid in range(mesh.MNumParts):
        pname = mesh.PGetName(pid)
        new_pname = pnames_map.get(pname, None)
        if not new_pname is None:
            mesh.PSetName(pid, new_pname)
            print(f"renaming part {pid} '{pname}' -> '{new_pname}'")
    return mesh

def AddPartAtPositionForLicenseDummy(mesh):
    """
    If license dummy exists, add FCE4M ":PPlicense"-part  at dummy position
    Do nothing, if ":PPlicense" already exists.
    """
    # lic_list = [":LICENSE_EURO", ":LICENSE EURO", ":LICENSE", ":LICMED", ":LICLOW"]
    lic_list = [":LICENS", ":LICMED", ":LICLOW"]
    dms_names = mesh.MGetDummyNames()
    dms_names = [name[:7] for name in dms_names]
    for lic in lic_list:
        if GetMeshPartnameIdx(mesh, ":PPlicense") >= 0:
            break
        if lic in dms_names:
            didx = dms_names.index(lic)
            dms_pos = mesh.MGetDummyPos()[3*didx:3*didx + 3]
            new_pid = mesh.OpAddHelperPart(":PPlicense", dms_pos)
            print(f"adding dummy part {new_pid} ':PPlicense'")
            break
    return mesh

def workload_ConvertPartnames_Fce3_to_Fce4_(mesh, fce_outversion):
    if fce_outversion not in ["3", 3]:
        return mesh
    # filepath_fce_output = filepath_fce_input

    # # Load FCE
    # mesh = fc.Mesh()
    # mesh = LoadFce(mesh, filepath_fce_input)

    if mesh.MNumParts < 1:
        print("ConvertPartnames: FCE must have at least 1 part.")
        return

    # Convert partnames
    # if CONFIG["script_version"] == "34":
    mesh = Partnames3to4_car(mesh)
    # elif CONFIG["script_version"] == "4M":
    #     mesh = Partnames4to4M_car(mesh)
    #     mesh = AddPartAtPositionForLicenseDummy(mesh)

    # WriteFce(fce_outversion, mesh, filepath_fce_output)
    return mesh

def workload_ConvertPartnames_Fce4_to_Fce4M_(filepath_fce_input, fce_outversion):
    if fce_outversion not in ["4M", "5", 5]:
        return
    filepath_fce_output = filepath_fce_input

    # Load FCE
    mesh = fc.Mesh()
    mesh = LoadFce(mesh, filepath_fce_input)

    if mesh.MNumParts < 1:
        print("ConvertPartnames: FCE must have at least 1 part.")
        return

    # Convert partnames
    # if CONFIG["script_version"] == "34":
    #     mesh = Partnames3to4_car(mesh)
    # elif CONFIG["script_version"] == "4M":
    mesh = Partnames4to4M_car(mesh)
    mesh = AddPartAtPositionForLicenseDummy(mesh)

    WriteFce(fce_outversion, mesh, filepath_fce_output)

# -------------------------------------- bfut_ConvertPartnames (Fce4 to Fce4M) (Fce3 to Fce4)
#########################################################

def HeuristicTgaSearch(path, suffix=".tga"):
    """
    Heuristic search for TGA file in the same directory as the given file path.
    Returns "path/to/texname.tga" if found, else empty string.

    priority: <file>.tga <file>00.tga <any>.tga
    """
    path = pathlib.Path(path)
    suffix = str(suffix).lower()
    if not path.is_dir() and path.is_file():
        pdir = path.parent
    else:
        pdir = path
    texname = None
    pl = list(pdir.iterdir())
    pl.sort()
    for f in pl:
        fp = pathlib.Path(f.name)
        if fp.suffix.lower() != suffix:
            continue
        if fp.stem.lower() == path.stem.lower():
            texname = pdir / fp
            break
        if fp.stem.lower() == path.stem.lower() + "00":
            texname = pdir / fp
            break
    if not texname:
        for f in pl:
            fp = pathlib.Path(f.name)
            if fp.suffix.lower() == suffix:
                texname = pdir / fp
                break
    return str(texname) if texname else ""

#########################################################

def unvivtool_integration(path):
    path = pathlib.Path(path)
    ptn = time.process_time_ns()
    vd = dict(uvt.get_info(path))
    print(f"Decoding '{path.name}' with unvivtool {uvt.__version__} took {(float(time.process_time_ns() - ptn) / 1e6):.2f} ms")
    def get_fce_tga_list_from_viv(vd: dict):
        """
        Get indexes of valid files with .tga or .fce extension from BIGF viv archive.

        Returns (True, fce_idx, tga_idx) if at least one .fce file found, else False.
        """

        """
        {'format': 'BIGF', '__state': 14, 'size': 1613816,
        'count_dir_entries': 16, 'count_dir_entries_true': 16,
        'header_size': 304, 'header_size_true': 304,
        'files': ['carp.txt', 'car.fce', 'car00.tga', 'fedata.fre', 'fedata.ita', 'fedata.bri', 'fedata.ger', 'fedata.fsh', 'fedata.spa', 'fedata.swe', 'fedata.eng', 'car.bnk', 'ocard.bnk', 'ocar.bnk', 'scar.bnk', 'dash.qfs'],
        'files_offsets': [304, 5332, 153240, 415428, 416076, 416688, 417148, 417860, 548972, 549648, 550340, 551040, 777276, 842864, 908444, 1038516],
        'files_sizes': [5028, 147908, 262188, 646, 611, 459, 711, 131112, 673, 691, 697, 226236, 65588, 65580, 130072, 575300],
        'files_fn_lens': [8, 7, 9, 10, 10, 10, 10, 10, 10, 10, 10, 7, 9, 8, 8, 8],
        'files_fn_ofs': [24, 41, 57, 75, 94, 113, 132, 151, 170, 189, 208, 227, 243, 261, 278, 295],
        'validity_bitmap': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
        """
        if vd.get("format", None) != "BIGF":
            return False, None, None, False
        files = np.array(vd.get("files", []))
        if len(files) < 1:
            return False, None, None, False
        validity_bitmap = np.array(vd.get("validity_bitmap", []))
        fce_idx = np.where([f.lower().endswith(".fce") for f in files])[0]
        fce_idx = fce_idx[validity_bitmap[fce_idx] == 1]  # only valid files
        if len(fce_idx) < 1:
            return False, None, None, True
        tga_idx = np.where([f.lower().endswith(".tga") for f in files])[0]
        tga_idx = tga_idx[validity_bitmap[tga_idx] == 1]  # only valid files
        return True, fce_idx, tga_idx, True

    ret, fce_idx, tga_idx, valid = get_fce_tga_list_from_viv(vd)
    if not ret:
        return False, None, None, None, None, None, valid
    files = np.array(vd.get("files", []))
    # files_offsets = np.array(vd.get("files_offsets", []))
    # files_sizes = np.array(vd.get("files_sizes", []))
    # validity_bitmap = np.array(vd.get("validity_bitmap", []))

    return True, files[fce_idx], files[tga_idx], vd, fce_idx, tga_idx, valid

#########################################################

# classes
class VivItem(PropertyGroup):
    """Group of properties representing an item in the list."""
    name: StringProperty(name="Name")
    vivpath: StringProperty(name="vivpath")

# https://docs.blender.org/api/current/bpy.types.Operator.html
class FCEC_OT_UpdateUIList(Operator):
    """Update fce and tga UI lists."""
    bl_idname = "viv_files.update"
    bl_label = "Update fce and tga lists"

    filepath: StringProperty(default="")

    def decode(self, context, fp: pathlib.Path):
        ret, fce_files, tga_files, _, _, _, _ = unvivtool_integration(fp)
        if not ret:
            return
        valid_files = np.concatenate((fce_files, tga_files))
        print(f"valid_files: {valid_files}")
        for v in valid_files:
            self.add_value(context, v)

    def add_value(self, context, val: str):
        ext = pathlib.Path(val).suffix
        if ext.lower() == ".fce":
            fce_files = context.scene.fce_files
            item = fce_files.add()
            item.name = val
            item.vivpath = self.filepath
        elif ext.lower() == ".tga":
            tga_files = context.scene.tga_files
            item = tga_files.add()
            item.name = val
            item.vivpath = self.filepath
        else:
            print(f"Unknown file extension '{ext}'")

    def execute(self, context):
        context.scene.fce_files.clear()  # reset list
        context.scene.tga_files.clear()  # reset list
        fp = pathlib.Path(self.filepath)
        if fp.suffix.lower() == ".viv":
            print(f"Decoding archive... '{fp}'")
            self.decode(context, fp)
            self.filepath = ""  # avoid decoder loop
        return {'FINISHED'}

class FCEC_OT_UpdateUIListExport(Operator):
    """Update fce and tga UI lists."""
    bl_idname = "viv_files_export.update"
    bl_label = "Update fce and tga lists"

    filepath: StringProperty(default="")

    def decode(self, context, fp: pathlib.Path):
        ret, fce_files, tga_files, _, _, _, valid = unvivtool_integration(fp)
        if not ret and not valid:
            return
        elif ret:
            valid_files = np.concatenate((fce_files, tga_files))
            print(f"valid_files: {valid_files}")
            for v in valid_files:
                self.add_value(context, v)
        if valid:
            self.add_message(context, "foobar.fce", "<add file>")
            self.add_message(context, "foobar.tga", "<add file>")

    def add_value(self, context, val: str):
        ext = pathlib.Path(val).suffix
        if ext.lower() == ".fce":
            fce_files = context.scene.fce_files
            item = fce_files.add()
            item.name = val
            item.vivpath = self.filepath
        elif ext.lower() == ".tga":
            tga_files = context.scene.tga_files
            item = tga_files.add()
            item.name = val
            item.vivpath = self.filepath
        else:
            print(f"Unknown file extension '{ext}'")

    def add_message(self, context, val: str, msg: str):
        ext = pathlib.Path(val).suffix
        if ext.lower() == ".fce":
            fce_files = context.scene.fce_files
            item = fce_files.add()
            item.name = msg
            # item.vivpath = pathlib.Path(self.filepath)
            item.vivpath = self.filepath
        elif ext.lower() == ".tga":
            tga_files = context.scene.tga_files
            item = tga_files.add()
            item.name = msg
            item.vivpath = self.filepath
        else:
            print(f"Unknown file extension '{ext}'")

    def execute(self, context):
        context.scene.fce_files.clear()  # reset list
        context.scene.tga_files.clear()  # reset list
        fp = pathlib.Path(self.filepath)
        if fp.suffix.lower() == ".viv":
            print(f"Decoding archive... '{fp}'")
            self.decode(context, fp)
            self.filepath = ""  # avoid decoder loop
        return {'FINISHED'}

class FCEC_UL_stringlist(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        ob = data
        slot = item
        ma = slot.name
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if ma:
                # layout.prop(ma, "name", text="", emboss=False, icon_value=icon)
                layout.label(text=ma, translate=False, icon_value=icon)
            else:
                layout.label(text="", translate=False, icon_value=icon)

class FcecodecImport(Operator, ImportHelper):
    """Load an FCE file, powered by fcecodec"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "import_scene.fce"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Import FCE"         # Display name in the interface.
    bl_options = {"REGISTER", "UNDO"}  # Enable undo for the operator.

    filter_glob: StringProperty(default="*.fce;*.tga;*.viv", options={"HIDDEN"})

    addon_dev_mode: BoolProperty(
        name="Developer Mode",
        description="Keep temp files",
        default=False
    )

    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )

    print_damage: BoolProperty(
        name="Load damage model",
        description="Import parts damage models as extra objects DAMAGE_*",
        default=True
    )

    print_dummies: BoolProperty(
        name="Load dummies (light/fx objects)",
        description="Import dummies as extra objects DUMMY_*",
        default=True
    )

    use_part_positions: BoolProperty(
        name="Use part positions",
        description="Load relative part positions from the file and apply them to the parts",
        default=True
    )

    print_part_positions: BoolProperty(
        name="Load part positions as objects",
        description="Import parts positions as extra objects POSITION_*",
        default=True
    )

    fce_convertFCE3partnames: BoolProperty(
        name="Import as HS partnames (NFS3 only)",
        description="RECOMMENDED On import, convert NFS3 partnames to NFS:HS partnames. ",
        default=True
    )

    fce_filter_triagflags_0xfff: BoolProperty(
        name="Triangle flag 12-bit mask",
        description="Mask triangle flags. You may want to experiment with turning this off for MCO imports. ",
        default=True
    )

    fce_restrict_texpage: BoolProperty(
        name="Restrict to selected texpage",
        description="ONLY RECOMMENDED for MCO. Load triangles with selected texpage only.",
        default=False
    )

    fce_select_texpage: IntProperty(
        name="Texpage",#"Select texpage",
        description="ONLY RECOMMENDED for MCO (0, 1, 2). NFS 3 & HS officer models / road objects are in (0-8) ",
        default=0, min=0, max=16)


    # draws import dialog
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        # layout.use_property_decorate = False  # No animation.

        box = layout.box()
        box.prop(self, "print_damage")
        box.prop(self, "print_dummies")

        box = layout.box()
        box.prop(self, "print_part_positions")
        box.prop(self, "use_part_positions")

        box = layout.box()
        box.prop(self, "fce_convertFCE3partnames")
        box.prop(self, "fce_filter_triagflags_0xfff")

        box = layout.box()
        box.prop(self, "fce_restrict_texpage")
        sub = box.row()
        sub.enabled = self.fce_restrict_texpage
        sub.prop(self, "fce_select_texpage")

        # if VIV selected in import dialog, display contents of first VIV file
        if self.files:
            for f in self.files:
                fp = pathlib.Path(f.name)
                if fp.suffix.lower() == ".viv":
                    pdir = pathlib.Path(self.filepath).parent
                    fp = pdir / fp
                    if not fp.is_file() or self.current_viv_archive == fp:   # avoid decoder loop
                        break
                    self.current_viv_archive = str(fp)
                    break
                else:
                    self.current_num_fce_files = 0
        sce = context.scene
        row = layout.row()
        if pathlib.Path(self.current_viv_archive).suffix.lower() == ".viv":
            row.operator("viv_files.update", text="Select from (.viv)").filepath = self.current_viv_archive
            self.current_viv_archive = ""  # avoid decoder loop
        layout.template_list("FCEC_UL_stringlist", "", sce, "fce_files", sce, "active_fce_files_index")
        layout.template_list("FCEC_UL_stringlist", "", sce, "tga_files", sce, "active_tga_files_index")

        if len(sce.tga_files) > 0:
            active_tga = sce.tga_files[sce.active_tga_files_index].name
            active_tga_vivpath = sce.tga_files[sce.active_tga_files_index].vivpath
            active_tga_vivpath = pathlib.Path(active_tga_vivpath).parent
            row = layout.row()
            row.label(text=f"Exports (.tga) to:")
            row = layout.row()
            row.label(text=f"{active_tga_vivpath / active_tga}")

        if DEV_MODE:
            layout.prop(self, "addon_dev_mode")

    current_viv_archive = ""  # avoid decoder loop

    def execute(self, context):
        time_suffix = str(time.time())
        tdir = pathlib.Path(tempfile.gettempdir())
        pdir = pathlib.Path(self.filepath).parent

        vivpath = None
        path = None
        texname = ""
        vd = None
        fce_idx = None, None
        tempfiles = []  # will be unlinked
        if self.files:
            # if VIV selected in import dialog
            for f in self.files:
                fp = pathlib.Path(f.name)
                if fp.suffix.lower() == ".viv":
                    fp = pdir / fp
                    ret, _, _, vd, fce_idx, _, valid = unvivtool_integration(fp)
                    if not ret:
                        continue
                    if len(fce_idx) > 0:
                        vivpath = fp
                        break

            # if FCE selected in import dialog
            for f in self.files:
                fp = pathlib.Path(f.name)
                if fp.suffix.lower() == ".fce":
                    path = pdir / fp
                    vivpath = None  # fce takes precedence
                    break

            # if TGA selected in import dialog
            for f in self.files:
                fp = pathlib.Path(f.name)
                if fp.suffix.lower() == ".tga":
                    texname = pdir / fp
                    break


        print(f"path: {path}")
        print(f"texname: {texname}")
        print(f"vivpath: {vivpath}")


        # if FCE selected and no TGA selected in import dialog, heuristic TGA search
        if path and texname == "":
            texname = HeuristicTgaSearch(path)
            if texname == "":
                texname = HeuristicTgaSearch(path, ".png")
            if texname == "":
                texname = HeuristicTgaSearch(path, ".bmp")
            if texname == "":
                texname = HeuristicTgaSearch(path, ".jpg")


        #else if VIV selected in import dialog, import selected FCE and TGA
        # fce is mandatory, tga is optional
        elif vivpath and vd:
            def decode_viv(vivpath, path, texname, tempfiles, vd, tdir, time_suffix):
                vivpath = pathlib.Path(vivpath)
                # check that active files from UILists are valid in selected VIV archive
                sce = context.scene
                if len(sce.fce_files) < 1:
                    print("Warning: No FCE file selected from VIV archive")
                    return None, None, tempfiles

                files = np.array(vd.get("files", []))
                print(f"files: {files}")
                active_fce = sce.fce_files[sce.active_fce_files_index].name
                active_fce_idx = np.where(files == active_fce)
                if len(active_fce_idx) < 1:
                    print("Warning: Selected FCE file not found in selected VIV archive")
                    return None, None, tempfiles
                active_fce_idx = active_fce_idx[0]
                if len(sce.tga_files) > 0:
                    active_tga = sce.tga_files[sce.active_tga_files_index].name
                    active_tga_idx = np.where(files == active_tga)
                    if len(active_tga_idx) > 0:
                        active_tga_idx = active_tga_idx[0]
                else:
                    active_tga_idx = np.array([])

                print(f"active_fce_idx: {active_fce_idx}")
                print(f"active_tga_idx: {active_tga_idx}")

                # decode active files from selected VIV archive
                for idx in np.concatenate([active_fce_idx, active_tga_idx]):
                    tmp_ = pathlib.Path(tdir / files[idx])
                    # print(f"unviv(): {idx} to {tmp_}")
                    ret = uvt.unviv(vivpath, tdir, fileidx=idx+1, verbose=False)
                    if ret and tmp_.is_file():
                        # FCE to /temp
                        if tmp_.suffix.lower() == ".fce":
                            tmp_ = tmp_.rename(tmp_.with_stem(tmp_.stem + "_" + time_suffix))
                            path = tmp_
                            tempfiles.append(tmp_)
                        # TGA to path/to/vivpath.parent/<tex>.tga
                        if tmp_.suffix.lower() == ".tga":
                            texname = shutil.move(tmp_, vivpath.parent / tmp_.name)
                return path, texname, tempfiles

            path, texname, tempfiles = decode_viv(vivpath, path, texname, tempfiles, vd, tdir, time_suffix)

        print(f"path: {path}")
        print(f"texname: {texname}")
        print(f"vivpath: {vivpath}")

        if not path:
            for f in tempfiles:
                f = pathlib.Path(f).unlink()
            return {"CANCELLED"}

        # paths to temporary files
        path_obj = pathlib.Path(path.stem + "_" + time_suffix).with_suffix(".obj")
        path_mtl = pathlib.Path(path.stem + "_" + time_suffix).with_suffix(".mtl")
        path_obj = pathlib.Path(str(path_obj).replace(" ", "_"))
        path_mtl = pathlib.Path(str(path_mtl).replace(" ", "_"))
        path_obj = tdir / path_obj
        path_mtl = tdir / path_mtl

        tempfiles.append(path_obj)
        tempfiles.append(path_mtl)

        PrintFceInfo(path)

        # Load FCE as mesh
        ptn = time.process_time_ns()
        mesh = fc.Mesh()
        mesh = LoadFce(mesh, path)
        print(f"FCE import of '{path.name}' took {(float(time.process_time_ns() - ptn) / 1e6):.2f} ms")


        # Apply options to mesh
        ptn = time.process_time_ns()
        if self.fce_convertFCE3partnames:
            buf = path.read_bytes()
            ver = fc.GetFceVersion(buf)
            mesh = workload_ConvertPartnames_Fce3_to_Fce4_(mesh, ver)
            del buf
        if self.fce_restrict_texpage:
            mesh = FilterTexpageTriags(mesh, select_texpages=self.fce_select_texpage)
        mesh = DeleteEmptyParts(mesh)
        print(f"Applying options to '{path.name}' took {(float(time.process_time_ns() - ptn) / 1e6):.2f} ms")


        # Export mesh to temporary OBJ
        ptn = time.process_time_ns()
        # mesh.PrintInfo()
        ExportObj(mesh, path_obj, path_mtl, texname, self.print_damage, self.print_dummies, self.use_part_positions, self.print_part_positions,
                filter_triagflags_0xfff=self.fce_filter_triagflags_0xfff)
        print(f"OBJ export to '{path_obj.name}' took {(float(time.process_time_ns() - ptn) / 1e6):.2f} ms")


        # Import temporary OBJ to Blender
        # https://docs.blender.org/api/current/bpy.ops.wm.html#bpy.ops.wm.obj_import
        bpy.ops.wm.obj_import(filepath=str(path_obj))


        # cleanup
        if not self.addon_dev_mode:
            for f in tempfiles:
                    f = pathlib.Path(f).unlink()

        return {"FINISHED"}


class FcecodecExport(Operator, ExportHelper):
    """Save an FCE file, powered by fcecodec"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "export_scene.fce"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Export FCE"         # Display name in the interface.

    filename_ext = ""

    filter_glob: StringProperty(default="*.fce;*.viv", options={"HIDDEN"})

    addon_dev_mode: BoolProperty(
        name="Developer Mode",
        description="Keep temp files",
        default=False
    )

    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )

    export_selected_objects: BoolProperty(
        name="Limit export to selected objects",
        description="Ignore parts that are not selected. ",
        default=False
    )

    obj_rescale_factor: FloatProperty(
        name="Scale",
        description="Rescale model on write. ",
        default=1.0, min=0.1, max=10.0)

    obj_forward_axis: EnumProperty(
        name="Forward Axis",
        # nomenclatura per bpy.ops.wm.obj_export()
        items=(("Z", "Z",
                "Z axis"),
               ("NEGATIVE_Z", "-Z",
                "Negative Z axis"),),
        # description=(
        #     "description " ),
        default="NEGATIVE_Z"
    )

    obj_apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Apply modifiers to exported meshes ",
        default=True
    )

    fce_version: EnumProperty(
        name="Format",
        items=(("3", "NFS3 (.fce)",
                "Exports a single file, compatible with NFS3: Hot Pursuit. "
                "This format does not feature damage models."),
               ("4", "NFS:HS (.fce)",
                "Exports a single file, compatible with NFS: High Stakes. "
                "Includes damage models. Also a common FCE exchange format. "),
               ("4M", "MCO (.fce)",
                "Exports a single file, compatible with Motor City Online. "
                "Includes damage models."),),
        description=(
            "Output format. NFS3 and NFS:HS "
            "are the most common formats."
        ),
        default=0
    )

    fce_convertpartnames: BoolProperty(
        name="Convert partnames (HS to MCO only)",
        description="Try converting partnames to output FCE format; try adding :PPLicense part. ",
        default=True
    )

    fce_convertdummies: BoolProperty(
        name="Convert dummies (light/fx objects)",
        description="Try converting light & fx objects to output FCE format. ",
        default=True
    )

    fce_reordertriags: BoolProperty(
        name="Semi-transparency fix",
        description="RECOMMENDED for NFS3 & HS models with transparent triangles. ",
        default=True
    )

    fce_material2texpage: BoolProperty(
        name="Map materials to FCE texture pages",
        description="Maps face materials to FCE texpages. "
                     "ONLY check for NFS3 & HS officer models, NFS:HS road objects, and MCO",
        default=False
    )

    fce_modify_num_arts: BoolProperty(
        name="Export officer or road object",
        description="ONLY check for NFS3 & HS officer models, NFS:HS road objects. ",
        default=False
    )

    fce_write_color: BoolProperty(
        name="Color",
        description="Write color to FCE. ",
        default=True
    )

    fce_color_picker: FloatVectorProperty(
        name="Color picker",
        subtype = "COLOR",
        default = (0.0, 0.17, 1.0, 1.0),
        size=4,
        min = 0.0,
        max = 1.0,
        # step=100,
    )

    fce_normals2vertices: BoolProperty(
        name="Export Normals as Verts",
        description="Replace vert positions with normals ",
        default=False
    )

    tga_path: StringProperty(
        subtype="FILE_PATH"
    )


    # draws export dialog
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        # layout.use_property_decorate = False  # No animation.

        box = layout.box()
        box.prop(self, "export_selected_objects")
        box.prop(self, "obj_rescale_factor")
        box.prop(self, "obj_forward_axis")
        box.prop(self, "obj_apply_modifiers")

        box = layout.box()
        box.prop(self, "fce_version")
        box.prop(self, "fce_convertdummies")
        sub = box.row()
        sub.enabled = self.fce_version == "4M"
        sub.prop(self, "fce_convertpartnames")
        box.prop(self, "fce_reordertriags")

        box = layout.box()
        box.prop(self, "fce_material2texpage")
        box.prop(self, "fce_modify_num_arts")

        box = layout.box()
        box.prop(self, "fce_write_color")
        sub = box.row()
        sub.enabled = self.fce_write_color
        sub.prop(self, "fce_color_picker")

        box = layout.box()
        box.prop(self, "fce_normals2vertices")

        # if VIV selected in import dialog, display contents of first VIV file
        if self.files:
            for f in self.files:
                fp = pathlib.Path(f.name)
                if fp.suffix.lower() == ".viv":
                    pdir = pathlib.Path(self.filepath).parent
                    fp = pdir / fp
                    if not fp.is_file() or self.current_viv_archive == fp:   # avoid decoder loop
                        break
                    self.current_viv_archive = str(fp)
                    break
                else:
                    self.current_num_fce_files = 0
        sce = context.scene
        row = layout.row()
        if pathlib.Path(self.current_viv_archive).suffix.lower() == ".viv":
            row.operator("viv_files.update", text="Select in (.viv)").filepath = self.current_viv_archive
            # row.operator("viv_files_export.update", text="Select in (.viv)").filepath = self.current_viv_archive
            self.current_viv_archive = ""  # avoid decoder loop
        layout.template_list("FCEC_UL_stringlist", "", sce, "fce_files", sce, "active_fce_files_index")
        # layout.template_list("FCEC_UL_stringlist", "", sce, "tga_files", sce, "active_tga_files_index")

        if DEV_MODE:
            layout.prop(self, "addon_dev_mode")

    current_viv_archive = ""  # avoid decoder loop

    def execute(self, context):
        time_suffix = str(time.time())
        tdir = pathlib.Path(tempfile.gettempdir())
        pdir = pathlib.Path(self.filepath).parent

        vivpath = None
        path = None
        texname = ""
        tempfiles = []  # will be unlinked
        if self.files:
            # if VIV selected in dialog
            for f in self.files:
                fp = pathlib.Path(f.name)
                if fp.suffix.lower() == ".viv":
                    fp = pdir / fp
                    ret, _, _, vd, fce_idx, _, valid = unvivtool_integration(fp)
                    if not ret:
                        continue
                    if len(fce_idx) > 0:
                        path = fp.with_suffix(".fce")
                        vivpath = fp
                        break

            # if at least 1 FCE selected in dialog
            for f in self.files:
                fp = pathlib.Path(f.name)
                if fp.suffix.lower() == ".fce":
                    path = pdir / fp
                    vivpath = None  # fce takes precedence
                    break

            # if no file selected in dialog
            if not path and not vivpath:
                path = pathlib.Path(self.filepath)
                path = path.with_suffix(".fce")

        print(f"path: {path}")
        print(f"texname: {texname}")
        print(f"vivpath: {vivpath}")


        # if vivpath is not None: copy VIV archive to temp dir, create temp FCE, modify VIV archive in temp dir, move VIV to destination path, unlink temp FCE
        # else: create temp FCE, move to actual path


        path_actual = None
        if vivpath is None: path_actual = str(path)

        # paths to temporary files
        path = pathlib.Path(path.stem + "_" + time_suffix).with_suffix(".fce")
        path = tdir / path
        path_obj = pathlib.Path(path.stem + "_" + time_suffix).with_suffix(".obj")
        path_mtl = pathlib.Path(path.stem + "_" + time_suffix).with_suffix(".mtl")
        path_obj = pathlib.Path(str(path_obj).replace(" ", "_"))
        path_mtl = pathlib.Path(str(path_mtl).replace(" ", "_"))
        path_obj = tdir / path_obj
        path_mtl = tdir / path_mtl

        tempfiles.append(path_obj)
        tempfiles.append(path_mtl)
        if vivpath: tempfiles.append(path)

        print(f"path_actual: {path_actual}")
        print(f"path: {path}")
        print(f"texname: {texname}")
        print(f"vivpath: {vivpath}")


        # https://docs.blender.org/api/current/bpy.ops.wm.html#bpy.ops.wm.obj_export
        bpy.ops.wm.obj_export(filepath=str(path_obj),
                              forward_axis=self.obj_forward_axis,
                              global_scale=self.obj_rescale_factor,
                              apply_modifiers=self.obj_apply_modifiers,
                            #   export_eval_mode="DAG_EVAL_RENDER",
                              export_selected_objects=self.export_selected_objects,
                              export_normals=True,
                              export_materials=True,
                              export_triangulated_mesh=True)

        if path_obj.exists() is False:
            print(f"FCE export failed at OBJ export to '{path_obj}'")
            return {"CANCELLED"}

        print(f"Writing to {path}")
        ptn = time.process_time_ns()
        CONFIG = {
            "fce_version"        : self.fce_version,  # output format version; expects "keep" or "3"|"4"|"4M" for FCE3, FCE4, FCE4M, respectively
            "center_parts"       : True,  # localize part vertice positions to part centroid, setting part position (expects 0|1)
            "material2texpage"   : self.fce_material2texpage,  # maps OBJ face materials to FCE texpages (expects 0|1)
            "material2triagflag" : 1,  # maps OBJ face materials to FCE triangles flag (expects 0|1)
            "normals2vertices"   :  self.fce_normals2vertices,  #  (expects 0|1)
        }
        workload_Obj2Fce(path_obj, path, CONFIG)
        print(f"FCE export of '{path.name}' took {(float(time.process_time_ns() - ptn) / 1e6):.2f} ms")


        print(f"Apply options to {path}")
        ptn = time.process_time_ns()

        if self.fce_convertpartnames:
            if self.fce_version == "4M":
                workload_ConvertPartnames_Fce4_to_Fce4M_(path, self.fce_version)

        if self.fce_version == "3":
            workload_SortPartsToFce3Order(path, self.fce_version)

        if self.fce_convertdummies:
            if self.fce_version == "3":
                workload_ConvertDummies_to_Fce3_(path, self.fce_version)
            else:
                workload_ConvertDummies_Fce3_to_Fce4_(path, self.fce_version)


        if self.fce_reordertriags:
            mesh = fc.Mesh()
            mesh = LoadFce(mesh, path)
            WriteFce(self.fce_version, mesh, path, mesh_function=HiBody_ReorderTriagsTransparentToLast)


        # colors... just a stop-gap measure
        if self.fce_write_color:
            hls = colorsys.rgb_to_hls(
                self.fce_color_picker[0],  # red
                self.fce_color_picker[1],  # green
                self.fce_color_picker[2],  # blue
            )
            hsbt = np.array([
                int(hls[0] * 255),  # hue
                int(hls[2] * 255),  # saturation
                int(hls[1] * 255),  # brightness (lightness)
                int(self.fce_color_picker[3] * 255),  # transparency (also lightness)
            ], dtype=np.uint8)

            # hsbt2 = hsbt.copy()
            # hsbt2[3] = 127

            # hsbt3 = hsbt.copy()
            # hsbt3[3] = 255

            tcolor = np.array([
                [
                    hsbt,  # PriColor per MSetColors()
                    hsbt,  # IntColor per MSetColors()
                    hsbt,  # SecColor per MSetColors()
                    hsbt,  # DriColor per MSetColors()
                ],
                # [
                #     hsbt2,  # PriColor per MSetColors()
                #     hsbt2,  # IntColor per MSetColors()
                #     hsbt2,  # SecColor per MSetColors()
                #     hsbt2,  # DriColor per MSetColors()
                # ],
                # [
                #     hsbt3,  # PriColor per MSetColors()
                #     hsbt3,  # IntColor per MSetColors()
                #     hsbt3,  # SecColor per MSetColors()
                #     hsbt3,  # DriColor per MSetColors()
                # ],
            ], dtype=np.uint8)
            print(f"tcolor: {tcolor} ({tcolor.shape})")

            mesh = fc.Mesh()
            mesh = LoadFce(mesh, path)
            mesh.MSetColors(tcolor)
            WriteFce(self.fce_version, mesh, path)


        # For officer models and pursuit road objects, the NumArts value must equal the greatest used texture page minus 1.
        # In all other cases, NumArts = 1.
        if self.fce_modify_num_arts:
            print(f"Setting NumArts from existing texture page")
            val = 1
            mesh = fc.Mesh()
            mesh = LoadFce(mesh, path)
            for pidx in range(mesh.MNumParts):
                texps = mesh.PGetTriagsTexpages(pidx)
                val = max(val, np.amax(texps) - 1)
                print(f"part {pidx} texpage range: {np.amin(texps)}-{np.amax(texps)}")
            mesh.MNumArts = val
            print(f"mesh.MNumArts: {mesh.MNumArts}")
            WriteFce(self.fce_version, mesh, path)


        ptn = float(time.process_time_ns() - ptn) / 1e6
        PrintFceInfo(path)
        print(f"Applying options to '{path.name}' took {ptn:.2f} ms")


        # if VIV selected in export dialog, updated selected VIV with exported FCE
        if vivpath:
            def update_viv(vivpath: pathlib.Path, fce_path):
                # check that active files from UILists are valid in selected VIV archive
                sce = context.scene
                if len(sce.fce_files) < 1:
                    print("Warning: No FCE file selected from VIV archive")
                    return None, None, tempfiles

                files = np.array(vd.get("files", []))
                print(f"files: {files}")
                active_fce = sce.fce_files[sce.active_fce_files_index].name
                print(f"active_fce: {active_fce}")
                active_fce_idx = np.where(files == active_fce)
                if len(active_fce_idx) < 1 or len(active_fce_idx[0]) < 1:
                    print("Warning: Selected FCE file not found in selected VIV archive")
                    return None, None, tempfiles
                active_fce_idx = int(active_fce_idx[0][0])

                print(f"active_fce_idx: {active_fce_idx}")

                # update active FCE in selected VIV archive
                ptn = time.process_time_ns()
                uvt.update(vivpath, fce_path, active_fce_idx+1, replace_filename=False, verbose=True)
                ptn = float(time.process_time_ns() - ptn) / 1e6
                print(f"Updating '{vivpath.name}' took {ptn:.2f} ms")

                viv = None
                viv = dict(uvt.get_info(vivpath))
                print(viv)

            print(f"Updating archive {vivpath}")
            update_viv(vivpath, path)


        # move temp file to destination
        if vivpath is None:
            if path_actual.is_file(): path_actual.unlink()
            path = path.rename(path_actual)

        # cleanup
        if not self.addon_dev_mode:
            for f in tempfiles:
                f = pathlib.Path(f).unlink()

        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(FcecodecImport.bl_idname, text="Need For Speed (.fce)")

def menu_func_export(self, context):
    self.layout.operator(FcecodecExport.bl_idname, text="Need For Speed (.fce)")

classes = (
    FcecodecImport,
    FcecodecExport,
    FCEC_UL_stringlist,
    FCEC_OT_UpdateUIList,
    FCEC_OT_UpdateUIListExport,
    VivItem,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.Scene.fce_files = CollectionProperty(type=VivItem)
    bpy.types.Scene.active_fce_files_index = IntProperty(name="active_fce_files_index")
    bpy.types.Scene.tga_files = CollectionProperty(type=VivItem)
    bpy.types.Scene.active_tga_files_index = IntProperty(name="active_tga_files_index")

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

