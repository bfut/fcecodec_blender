# Copyright (C) 2024 and later Benjamin Futasz <https://github.com/bfut>
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
"""
    fcecodec_blender.py

    Features:
        import/export FCE to Blender 4.x

    Installation:
        Edit > Preferences > Add-ons > Install... > fcecodec_blender.py

    * File > Import > Need For Speed (.fce)
        - select FCE file and optional TGA texture file in the import dialog
        - check the checkboxes for additional import options

    * File > Export > Need For Speed (.fce)
        - self-explaning Blender export dialog

"""

bl_info = {
    "name": "fcecodec_blender",
    "author": "bfut",
    "version": (0, 6),
    "blender": (4, 0, 0),
    "location": "File > Import/Export > Need For Speed (.fce)",
    "description": "Imports & Exports Need For Speed (.fce) files",
    "category": "Import-Export",
    "url": "https://github.com/bfut/fcecodec",
}

# import contextlib
# import os
import pathlib
import re
import subprocess
import sys
import time


def pip_install(package, upgrade=False, pre=False):
    if upgrade and not pre:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])
    elif upgrade and pre:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package, "--pre"])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import fcecodec as fc
except ImportError:
    pip_install("pip", upgrade=True)
    pip_install("fcecodec", upgrade=True)
    import fcecodec as fc

try:
    import numpy as np
except ImportError:
    pip_install("numpy")
    import numpy as np

try:
    import tinyobjloader
except ImportError:
    pip_install("pip", upgrade=True)
    pip_install("tinyobjloader", upgrade=True, pre=True)
    import tinyobjloader


import bpy
from bpy.props import (StringProperty,
                       BoolProperty,
                       EnumProperty,
                       IntProperty,
                       FloatProperty,
                       CollectionProperty)
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper, ExportHelper #, poll_file_object_drop


# wrappers
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

def ExportObj(mesh, objpath, mtlpath, texname, print_damage, print_dummies,
              use_part_positions, print_part_positions):
    mesh.IoExportObj(str(objpath), str(mtlpath), str(texname), print_damage,
                     print_dummies, use_part_positions, print_part_positions)

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
    # print(arr)
    print(type(attrib.numpy_vertices()), arr.shape, arr.ndim)
    return arr

def GetNormals(reader):  # xyzxyzxyz
    attrib = reader.GetAttrib()
    arr = np.array(attrib.normals)
    print(type(arr), arr.shape, arr.ndim)
    return arr

def GetTexcoords(reader):  # uvuvuv
    attrib = reader.GetAttrib()
    arr = np.array(attrib.texcoords)
    print(type(arr), arr.shape, arr.ndim)
    return arr

def PrintShapes(reader):
    shapes = reader.GetShapes()
    print("Num shapes: ", len(shapes))
    for shape in shapes:
        print(shape.name,
              f"faces={int(shape.mesh.numpy_indices().shape[0] / 3)}")

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
    print(shape.name, f"normals={int(normals_idxs.shape[0])}")
    print(shape.name, f"texcoords={int(texcoord_idxs.shape[0])}")

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
    if np.amax(s_faces) <= int(normals.shape[0] / 3):
        print("norm_selection")
        s_norms = normals.reshape(-1, 3)[ norm_selection ].flatten()  # normals[normals_idxs]
    else:
        print("shape has no normals... use vert positions as normals")
        s_norms = np.copy(s_verts)

    print("GetShapeFaces: uvuvuv... -> uuuvvv...")
    s_texcs = np.empty(s_NumFaces * 6)
    for i in range(s_NumFaces):
        for j in range(3):
            s_texcs[i*6 + 0*3 + j] = texcoords[texcoord_idxs[i*3 + j] * 2 + 0]
            s_texcs[i*6 + 1*3 + j] = texcoords[texcoord_idxs[i*3 + j] * 2 + 1]

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
            if materials[s_matls[i]].name[:2] == "0x":
                # Blender may change the name of the material to "0x003.001" from "0x003"
                mat_ = materials[s_matls[i]].name[2:]
                mat_ = re.sub(r'\.(.*)', '', mat_)
                texps[i] = int(mat_, base=16)
            else:
                # Blender may change the name of the material to "<name>.001" from "<name>"
                mat_ = materials[s_matls[i]].name
                mat_ = re.sub(r'\.(.*)', '', mat_)
                tags = mat_.split("_")
                texps[i] = GetTexPageFromTags(tags)
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
            try:
                # print(materials[s_matls[i]].name, int(materials[s_matls[i]].name[2:], base=16))
                tflags[i] = int(materials[s_matls[i]].name[2:], base=16)
            except ValueError:
                tags = materials[s_matls[i]].name.split("_")
                tflags[i] = GetFlagFromTags(tags)
        mesh.PSetTriagsFlags(mesh.MNumParts - 1, tflags)

        for i in range(len(materials)):
            tmp = materials[i].name
            # print(tmp)
            if tmp[:2] == "0x":
                try:
                    # print(tmp, "->", int(tmp[2:], base=16), f"0x{hex(int(tmp[2:], base=16))}")
                    val = int(tmp[2:], base=16)
                    del val
                except ValueError:
                    print(f"Cannot map faces material name to triangles flags ('{materials[i].name}' is not hex value) 1")
                    break

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
            pid = np.argwhere(part_names == part_names[damgd_pid][7:])
            # print(damgd_pid, part_names[damgd_pid], pid, len(pid))
            if len(pid) < 1:
                continue
            pid = pid[0][0]
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
            damgd_pids += [damgd_pid]
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
    print(pos_parts)
    for pid in range(mesh.MNumParts):
        if part_names[pid][:9] != "POSITION_":
            if part_names[pid] not in pos_parts:  # POSITION_<partname> not available
                print(f"center {part_names[pid]} to local centroid")
                mesh.OpCenterPart(pid)
            else:                                 # POSITION_<partname> is available
                pos_pid = pos_parts[part_names[pid]]
                print(f"center {part_names[pid]} to centroid of {part_names[pos_pid]}")
                pos_pids += [pos_pid]
                mesh.OpCenterPart(pos_pid)
                mesh.OpSetPartCenter(pid, mesh.PGetPos(pos_pid))
    for i in sorted(pos_pids, reverse=True):
        mesh.OpDeletePart(i)
    return mesh


#
def workload_Obj2Fce(filepath_obj_input, filepath_fce_output, CONFIG):
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
    mesh = CopyDamagePartsVertsToPartsVerts(mesh)
    mesh = PartsToDummies(mesh)
    mesh = SetAnimatedVerts(mesh)
    if CONFIG["center_parts"] == 1:
        mesh = CenterParts(mesh)

    # Write FCE
    WriteFce(CONFIG["fce_version"], mesh, filepath_fce_output, center_parts=False)
    print(flush=True)
    # PrintFceInfo(filepath_fce_output)
    print(f"filepath_fce_output={filepath_fce_output}")

# -------------------------------------- bfut_Obj2Fce3
#########################################################

#
def workload_RescaleModel(filepath_fce_input, rescale_factor, fce_outversion, CONFIG):
    filepath_fce_output = filepath_fce_input

    # Load FCE
    mesh = fc.Mesh()
    mesh = LoadFce(mesh, filepath_fce_input)

    # Rescale
    print(f"rescale_factor={rescale_factor}")
    if rescale_factor > 0.0 and abs(rescale_factor) > 0.1:
        for pid in reversed(range(mesh.MNumParts)):
            mesh.OpCenterPart(pid)
            ppos = mesh.PGetPos(pid)  # xyz
            mesh.PSetPos(pid, ppos * rescale_factor)
        v = mesh.MVertsPos  # xyzxyzxyz...
        mesh.MVertsPos = v * rescale_factor
        dv = mesh.MGetDummyPos()  # xyzxyzxyz...
        mesh.MSetDummyPos(dv * rescale_factor)

    # Write FCE
    WriteFce(fce_outversion, mesh, filepath_fce_output, CONFIG["center_parts"])
    # PrintFceInfo(filepath_fce_output)
    # print("OUTPUT =", filepath_fce_output, flush=True)


#########################################################
# -------------------------------------- bfut_SortPartsToFce3Order

# -------------------------------------- script functions
def PrintMeshParts_order(mesh, part_names_sorted):
    print("pid  IS                          SHOULD")
    for pid in range(mesh.MNumParts):
        print(f"{pid:<2} {mesh.PGetName(pid):<12} {part_names_sorted[pid]:<12}")

def AssertPartsOrder(mesh, part_names_sorted):
    for pid in range(mesh.MNumParts):
        if mesh.PGetName(pid) != part_names_sorted[pid]:
            PrintMeshParts_order(mesh, part_names_sorted)
            raise AssertionError (f"pid={pid} {mesh.PGetName(pid)} != {part_names_sorted[pid]}")

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
        AssertPartsOrder(mesh, part_names_sorted)
        # PrintMeshParts(mesh, part_names_sorted)


    WriteFce(fce_outversion, mesh, filepath_fce_output)
    PrintFceInfo(filepath_fce_output)
    # print(f"OUTPUT = {filepath_fce_output}", flush=True)

# -------------------------------------- bfut_SortPartsToFce3Order
#########################################################


# workload
class FcecodecImport(Operator, ImportHelper):
    """Using fcecodec, import an FCE file"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "import_scene.fce"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Need For Speed (.fce)"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    filter_glob: StringProperty(default="*.fce;*.tga", options={'HIDDEN'})

    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )

    print_damage: BoolProperty(
        name='Load damage model',
        description='Import parts damage models as extra objects DAMAGE_*',
        default=True
    )

    print_dummies: BoolProperty(
        name='Load dummies',
        description='Import dummies as extra objects DUMMY_*',
        default=True
    )

    use_part_positions: BoolProperty(
        name='Use part positions',
        description='Load relative part positions from the file and apply them to the parts',
        default=True
    )

    print_part_positions: BoolProperty(
        name='Load part positions as objects',
        description='Import parts positions as extra objects POSITION_*',
        default=True
    )

    def __init__(self):
        pass

    # draws checkboxes in the import dialog
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        # layout.use_property_decorate = False  # No animation.
        layout.prop(self, 'print_damage')
        layout.prop(self, 'print_dummies')
        layout.prop(self, 'print_part_positions')
        layout.prop(self, 'use_part_positions')

    # execute() is called when running the operator.
    def execute(self, context):
        # path = pathlib.Path(self.filepath)
        pdir = pathlib.Path(self.filepath).parent
        path = None
        texname = ""
        if self.files:
            for f in self.files:
                fp = pathlib.Path(f.name)
                if fp.suffix.lower() == ".fce":
                    path = pdir / fp
                    break
            for f in self.files:
                fp = pathlib.Path(f.name)
                if fp.suffix.lower() == ".tga":
                    texname = pdir / fp
                    break

        print(f"path: {path}")
        print(f"texname: {texname}")

        if not path:
            return {'CANCELLED'}

        # paths to temporary files
        time_suffix = str(time.time())
        path_obj = path.with_name(path.stem + "_" + time_suffix).with_suffix(".obj")
        path_mtl = path.with_name(path.stem + "_" + time_suffix).with_suffix(".mtl")

        # PrintFceInfo(path)
        mesh = fc.Mesh()
        mesh = LoadFce(mesh, path)
        ExportObj(mesh, path_obj, path_mtl, texname, self.print_damage, self.print_dummies, self.use_part_positions, self.print_part_positions)
        bpy.ops.wm.obj_import(filepath=str(path_obj))

        clrs = mesh.MGetColors()

        # cleanup
        path_obj.unlink()
        path_mtl.unlink()

        return {'FINISHED'}


class FcecodecExport(Operator, ExportHelper):
    # https://github.com/blender/blender-addons/tree/main/io_scene_gltf2

    """Using fcecodec, import an FCE file"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "export_scene.fce"        # Unique identifier for buttons and menu items to reference.
    bl_label = "Need For Speed (.fce)"         # Display name in the interface.

    filename_ext = ''

    filter_glob: StringProperty(default='*.fce', options={'HIDDEN'})

    export_selected_objects: BoolProperty(
        name='Limit export to selected objects',
        description='Ignore parts that are not selected. ',
        default=False
    )

    fce_version: EnumProperty(
        name='Format',
        items=(('3', 'NFS3 (.fce)',
                'Exports a single file, compatible with NFS3: Hot Pursuit. '
                'This format does not feature damage models.'),
               ('4', 'NFS:HS (.fce)',
                'Exports a single file, compatible with NFS: High Stakes. '
                'Easiest to edit later, includes damage models.'),
               ('4M', 'MCO (.fce)',
                'Exports a single file, compatible with Motor City Online. '
                'Includes damage models.'),),
        description=(
            'Output format. NFS3 and NFS:HS '
            'are the the most common formats.'
        ),
        default=0,  # Warning => If you change the default, need to change the default filter too
    )

    fce_material2texpage: BoolProperty(
        name='Map materials to FCE texture pages',
        description='Maps face materials to FCE texpages. '
                     'Only relevant for NFS3 & HS officer models, NFS:HS road objects, and MCO',
        default=False
    )

    fce_num_arts: IntProperty(
        name='Number of arts',
        description='Must be set to 1 for NFS3 and NFS:HS unless you know what you are doing. ',
        default=1,
        min=0,
        max=30
    )

    fce_rescale_factor: FloatProperty(
        name='Scale',
        description='Warning: Values other than 1.0 will reset custom part positions.',
        default=1.0, min=0.1, max=10.0)

    # draws checkboxes in the import dialog
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        # layout.use_property_decorate = False  # No animation.
        layout.prop(self, 'export_selected_objects')
        layout.prop(self, 'fce_version')
        layout.prop(self, 'fce_material2texpage')
        layout.prop(self, 'fce_num_arts')
        layout.prop(self, 'fce_rescale_factor')

    def execute(self, context):
        path = pathlib.Path(self.filepath)
        if path.suffix.lower() != ".fce":
            path = path.with_suffix(".fce")

        # paths to temporary files
        time_suffix = str(time.time())
        path_obj = path.with_name(path.stem + "_" + time_suffix).with_suffix(".obj")
        path_mtl = path.with_name(path.stem + "_" + time_suffix).with_suffix(".mtl")

        bpy.ops.wm.obj_export(filepath=str(path_obj),
                              export_selected_objects=self.export_selected_objects,
                              export_materials=True,
                              export_triangulated_mesh=True)

        if path_obj.exists() is False:
            print(f"FCE export failed at OBJ export to '{path_obj}'")
            return {'CANCELLED'}

        print(f"Writing to {path}")
        ptn = time.process_time_ns()
        CONFIG = {
            "fce_version"        : self.fce_version,  # output format version; expects "keep" or "3"|"4"|"4M" for FCE3, FCE4, FCE4M, respectively
            "center_parts"       : True,  # localize part vertice positions to part centroid, setting part position (expects 0|1)
            "material2texpage"   : self.fce_material2texpage,  # maps OBJ face materials to FCE texpages (expects 0|1)
            "material2triagflag" : 1,  # maps OBJ face materials to FCE triangles flag (expects 0|1)
        }
        # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        #     workload_Obj2Fce(path_obj, path, CONFIG)
        workload_Obj2Fce(path_obj, path, CONFIG)
        ptn = float(time.process_time_ns() - ptn) / 1e6
        print(f"FCE export of '{path.name}' took {ptn:.2f} ms")


        print(f"Apply options to {path}")
        ptn = time.process_time_ns()

        if self.fce_rescale_factor < 0.1:
            self.fce_rescale_factor = 1.0
        if abs(1.0 - self.fce_rescale_factor) > 1e-3:
            CONFIG = {
                "fce_version"  : self.fce_version,  # output format version; expects "keep" or "3"|"4"|"4M" for FCE3, FCE4, FCE4M, respectively
                "center_parts" : True,  # localize part vertice positions to part centroid, setting part position (expects 0|1)
                "rescale_factor" : self.fce_rescale_factor,  # 1.1 and 1.2 give good results for vanilla FCE4/FCE4M models; unchanged if 1.0 or smaller than 0.1
            }
            workload_RescaleModel(path, self.fce_rescale_factor, self.fce_version, CONFIG)

        if self.fce_version == "3":
            workload_SortPartsToFce3Order(path, self.fce_version)

        # TODO: For officer models and pursuit road objects,
        # the NumArts value must equal the greatest used texture page minus 1. In all other cases, NumArts = 1.
        if self.fce_num_arts != 1:
            print(f"Setting NumArts to {self.fce_num_arts}")
            mesh = fc.Mesh()
            mesh = LoadFce(mesh, path)
            mesh.MNumArts = self.fce_num_arts
            WriteFce(self.fce_version, mesh, path)
            PrintFceInfo(path)

        ptn = float(time.process_time_ns() - ptn) / 1e6
        print(f"Applying options to '{path.name}' took {ptn:.2f} ms")

        # cleanup
        path_obj.unlink()
        path_mtl.unlink()

        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(FcecodecImport.bl_idname)

def menu_func_export(self, context):
    self.layout.operator(FcecodecExport.bl_idname)

def register():
    bpy.utils.register_class(FcecodecImport)
    bpy.utils.register_class(FcecodecExport)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(FcecodecImport)
    bpy.utils.unregister_class(FcecodecExport)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
