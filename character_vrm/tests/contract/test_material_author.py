"""Contract test for ``blender/material_author.py`` against a tiny hand-written node graph (not a fake Blender).

It proves the wiring the glTF exporter reads (roughness image, glTF Material Output occlusion, specular, the
``1 - (alpha < cutoff)`` clip) and that linked sockets are respected. Real Blender: tests/integration/README.md.
"""

from types import SimpleNamespace

from character_vrm.blender import material_author
from character_vrm.core.authoring.materials import MAX_MAP_SIZE, MaterialEdit


class Socket:
    def __init__(self, name, value=0.0):
        self.name, self.default_value, self.links = name, value, []

    @property
    def is_linked(self):
        return bool(self.links)


class Sockets(dict):
    def __init__(self, names):
        super().__init__({n: Socket(n) for n in names})

    def __getitem__(self, key):
        return list(self.values())[key] if isinstance(key, int) else super().__getitem__(key)


class Node:
    def __init__(self, type_, inputs=(), outputs=("Color",)):
        self.type, self.inputs, self.outputs = type_, Sockets(inputs), Sockets(outputs)
        for s in self.outputs.values():
            s.node = self
        self.location = SimpleNamespace(x=0.0, y=0.0)
        self.operation = self.image = self.node_tree = self.label = None


class Link:
    def __init__(self, a, b):
        self.from_socket, self.to_socket, self.from_node = a, b, a.node


class Links(list):
    def new(self, a, b):
        link = Link(a, b)
        b.links = [link]
        self.append(link)
        return link

    def remove(self, link):
        link.to_socket.links = []
        list.remove(self, link)


class Nodes(list):
    KINDS = {"ShaderNodeTexImage": ("TEX_IMAGE", (), ("Color", "Alpha")),
             "ShaderNodeMath": ("MATH", ("A", "B"), ("Value",)),
             "ShaderNodeGroup": ("GROUP", ("Occlusion",), ())}

    def new(self, kind):
        t, ins, outs = self.KINDS[kind]
        node = Node(t, ins, outs)
        self.append(node)
        return node


class Image:
    def __init__(self, path):
        self.path, self.size, self.colorspace_settings = path, [4096, 4096], SimpleNamespace(name="sRGB")

    def scale(self, w, h):
        self.size = [w, h]


def _world():
    bsdf = Node("BSDF_PRINCIPLED", ("Roughness", "Specular IOR Level", "Specular Tint", "Alpha"), ("BSDF",))
    bsdf.inputs["Specular Tint"].default_value = (0.0, 0.0, 0.0, 1.0)
    tree = SimpleNamespace(nodes=Nodes([bsdf]), links=Links())
    opacity = tree.nodes.new("ShaderNodeTexImage")
    tree.links.new(opacity.outputs["Color"], bsdf.inputs["Alpha"])
    mat = SimpleNamespace(use_nodes=True, node_tree=tree)
    images: dict[str, Image] = {}
    groups: dict[str, object] = {}

    def load(path, check_existing=True):
        return images.setdefault(path, Image(path))

    def new_group(name, kind):
        g = SimpleNamespace(name=name, interface=SimpleNamespace(new_socket=lambda *a, **k: None), nodes=Nodes())
        g.nodes.new = lambda kind: None
        groups[name] = g
        return g

    data = SimpleNamespace(materials={"Hair": mat}, images=SimpleNamespace(load=load),
                           node_groups=SimpleNamespace(get=groups.get, new=new_group))
    return SimpleNamespace(data=data), bsdf, tree, opacity, images


def test_hair_gets_maps_specular_and_the_exporters_alpha_clip_pattern():
    bpy, bsdf, tree, opacity, images = _world()
    edit = MaterialEdit("Hair", "hair", 0.9, roughness=None, specular=0.5, alpha_cutoff=0.35,
                        textures={"roughness": "/t/Hair_roughness.png", "ao": "/t/Hair_ao.png"})
    done, failed = material_author.apply_materials([edit], bpy)
    assert failed == {} and set(done["Hair"]) == {"roughness map", "specular 0.5", "ao map", "alpha clip 0.35"}
    rough_src = bsdf.inputs["Roughness"].links[0].from_node
    assert rough_src.type == "TEX_IMAGE" and rough_src.image.colorspace_settings.name == "Non-Color"
    assert all(max(i.size) == MAX_MAP_SIZE for i in images.values())
    assert bsdf.inputs["Specular Tint"].default_value == (1.0, 1.0, 1.0, 1.0)
    sub = bsdf.inputs["Alpha"].links[0].from_node
    less = sub.inputs[1].links[0].from_node
    assert (sub.operation, sub.inputs[0].default_value) == ("SUBTRACT", 1.0)
    assert (less.operation, less.inputs[1].default_value) == ("LESS_THAN", 0.35)
    assert less.inputs[0].links[0].from_node is opacity
    group = next(n for n in tree.nodes if n.type == "GROUP")
    assert group.node_tree.name == material_author.GLTF_GROUP and group.inputs["Occlusion"].is_linked


def test_a_linked_roughness_is_the_sources_own_and_stays():
    bpy, bsdf, tree, _, _ = _world()
    own = tree.nodes.new("ShaderNodeTexImage")
    tree.links.new(own.outputs["Color"], bsdf.inputs["Roughness"])
    done, failed = material_author.apply_materials(
        [MaterialEdit("Hair", "hair", 0.9, roughness=0.55, textures={"roughness": "/t/r.png"})], bpy)
    assert bsdf.inputs["Roughness"].links[0].from_node is own and "Hair" not in done and failed == {}


def test_a_material_without_principled_fails_alone():
    bpy, *_ = _world()
    bpy.data.materials["Other"] = SimpleNamespace(use_nodes=False, node_tree=None)
    done, failed = material_author.apply_materials(
        [MaterialEdit("Other", "skin", 0.9, roughness=0.5), MaterialEdit("Hair", "hair", 0.9, specular=0.5)], bpy)
    assert "material Other" in failed and done == {"Hair": ["specular 0.5"]}
