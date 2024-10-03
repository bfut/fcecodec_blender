# fcecodec_blender
Full-featured Blender (.fce) Import/Export Add-on for Need For Speed 3 & High Stakes car models.

Supports Blender 4.x, 4.2 LTS, and 3.6 LTS on Windows, Linux, and macOS.

## Installation
1. Download fcecodec_blender.py
1. Open Blender
1. Edit > Preferences > Add-ons > Install... > fcecodec_blender.py
   - requires an active internet connection while Blender installs Python modules "fcecodec" and "tinyobjloader" and "unvivtool"

## Usage
 * File > Import > Need For Speed (.fce)
    - load selected (.fce) file and optional (.tga) texture file
    - alternatively, select (.viv) archive
        1. Hit "Select from (.viv)" button
        2. Select (.fce) file and optional (.tga) from the lists
        3. Hit "Import FCE" button

* File > Export > Need For Speed (.fce)
    - export as (.fce) file
    - alternatively, select (.viv) archive
        1. Hit "Select in (.viv)" button
        2. Select (.fce) file from the list
        3. Hit "Export FCE" button

## Tutorial

The following tutorial describes how to use Blender to:
* create/modify damage models
* edit part centers
* set vertice animation flags
* edit dummies (light / fx objects)
* edit triangle flags
* edit texture pages

[Tutorial](https://github.com/bfut/fcecodec/tree/main/scripts/doc_Obj2Fce.md)

## Information
__License:__ GNU General Public License v3.0+<br/>
__Website:__ <https://github.com/bfut/fcecodec_blender>

powered by [fcecodec](https://github.com/bfut/fcecodec) and [unvivtool](https://github.com/bfut/unvivtool)
