"""Check URDF prerequisites for the B1/B2 asset index contract without Isaac Gym.

These tests compare the retained joint trees with collapse_fixed_joints=True.
XML traversal is not an implementation of Isaac Gym's importer: actual rigid
body, DOF and shape indices must also be checked by loading both assets in Gym.
"""

from collections import defaultdict
from pathlib import Path
import unittest
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
ASSET_PATHS = {
    "b1": ROOT / "resources/robots/b1z1/urdf/b1z1.urdf",
    "b2": ROOT / "resources/robots/b2_z1_lidar_mount_fast/urdf/b2_z1_lidar_mount_fast.urdf",
}


def retained_tree(document):
    """Return retained edges in XML child order and each link's merged owner."""
    links = [link.attrib["name"] for link in document.findall("link")]
    joints = document.findall("joint")
    joint_names = [joint.attrib["name"] for joint in joints]
    if len(set(links)) != len(links) or len(set(joint_names)) != len(joint_names):
        raise ValueError("Duplicate link or joint names")

    children = defaultdict(list)
    incoming = set()
    for joint in joints:
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if parent not in links or child not in links or child in incoming:
            raise ValueError("Invalid joint references or multiple parents")
        children[parent].append(joint)
        incoming.add(child)

    roots = set(links) - incoming
    if len(roots) != 1:
        raise ValueError("Expected exactly one root link")
    root = roots.pop()
    edges, owners = [], {}

    def visit(link, owner):
        if link in owners:
            raise ValueError("Cycle in joint tree")
        owners[link] = owner
        for joint in children[link]:
            child = joint.find("child").attrib["link"]
            collapsible = (joint.attrib["type"] == "fixed"
                           and joint.get("dont_collapse", "false") != "true")
            if collapsible:
                visit(child, owner)
            else:
                edges.append((owner, child, joint.attrib["name"], joint.attrib["type"]))
                visit(child, child)

    visit(root, root)
    if set(owners) != set(links):
        raise ValueError("Disconnected links in joint tree")
    return root, edges, owners


def normalized_edges(edges):
    # Configs deliberately retain their chassis names: B1 trunk, B2 base_link.
    def normalize(name):
        return "trunk" if name == "base_link" else name

    return [(normalize(parent), normalize(child), joint, kind)
            for parent, child, joint, kind in edges]


def joints_in_name_order(edges, root="base"):
    """Traverse the retained tree deterministically, not via the Gym importer."""
    children = defaultdict(list)
    for parent, child, joint, kind in edges:
        children[parent].append((joint, child, kind))

    def visit(parent):
        for joint, child, kind in sorted(children[parent]):
            yield child, joint, kind
            yield from visit(child)

    return list(visit(root))


class RobotAssetIndexCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {
            name: ElementTree.parse(path).getroot()
            for name, path in ASSET_PATHS.items()
        }
        cls.trees = {name: retained_tree(doc) for name, doc in cls.documents.items()}

    def test_retained_tree_and_xml_child_order_match_b1(self):
        b1_root, b1_edges, _ = self.trees["b1"]
        b2_root, b2_edges, _ = self.trees["b2"]
        self.assertEqual((b1_root, b2_root), ("base", "base"))
        self.assertEqual(normalized_edges(b2_edges), normalized_edges(b1_edges))
        # 2 base bodies, 4 * 4 leg bodies, 6 arm bodies, moving jaw and EE.
        self.assertEqual(len(b1_edges) + 1, 26)

    def test_fixed_joints_keep_the_same_body_boundaries(self):
        for name, document in self.documents.items():
            with self.subTest(asset=name):
                _, edges, owners = self.trees[name]
                chassis = "trunk" if name == "b1" else "base_link"
                base = document.find("link[@name='base']")
                self.assertIsNotNone(base)
                self.assertIsNone(base.find("inertial"))
                self.assertIsNone(base.find("collision"))
                joint = document.find("joint[@name='floating_base']")
                self.assertIsNotNone(joint)
                self.assertEqual(joint.attrib["type"], "fixed")
                self.assertEqual(joint.attrib.get("dont_collapse"), "true")
                self.assertEqual(joint.find("parent").attrib["link"], "base")
                self.assertEqual(joint.find("child").attrib["link"], chassis)
                for field in ("xyz", "rpy"):
                    self.assertEqual(
                        [float(value) for value in joint.find("origin").get(field).split()],
                        [0.0, 0.0, 0.0],
                    )

                self.assertEqual(owners[chassis], chassis)
                self.assertEqual(owners["link00"], "base")
                self.assertEqual(owners["gripperStator"], "link06")
                retained_fixed_children = {child for _, child, _, kind in edges if kind == "fixed"}
                self.assertEqual(retained_fixed_children, {
                    chassis, "FL_foot", "FR_foot", "RL_foot", "RR_foot", "ee_gripper_link",
                })

    def test_retained_body_groups_in_joint_name_order(self):
        expected = ["base", "trunk"]
        expected += [f"{leg}_{part}" for leg in ("FL", "FR", "RL", "RR")
                     for part in ("hip", "thigh", "calf", "foot")]
        expected += [f"link{number:02}" for number in range(1, 7)]
        expected += ["ee_gripper_link", "gripperMover"]
        for name, (_, edges, _) in self.trees.items():
            with self.subTest(asset=name):
                bodies = ["base"] + [child for child, _, _ in
                                     joints_in_name_order(normalized_edges(edges))]
                self.assertEqual(bodies, expected)

    def test_active_joint_groups_and_order_match(self):
        arm_joints = ["z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle",
                      "z1_forearm_roll", "z1_wrist_rotate", "z1_jointGripper"]
        for name, (_, edges, _) in self.trees.items():
            with self.subTest(asset=name):
                active = [(joint, kind) for _, _, joint, kind in edges if kind != "fixed"]
                expected_xml = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL")
                                for part in ("hip", "thigh", "calf")] + arm_joints
                self.assertEqual(active, [(joint, "revolute") for joint in expected_xml])

                # Check the same group contract under deterministic joint-name
                # traversal, independently of XML declaration order. This does
                # not assert that Gym uses this traversal internally.
                expected_groups = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
                                   for part in ("hip", "thigh", "calf")] + arm_joints
                self.assertEqual([joint for _, joint, kind in joints_in_name_order(edges)
                                  if kind != "fixed"], expected_groups)
                self.assertEqual(len(active), 19)

    def test_mesh_references_resolve(self):
        for name, document in self.documents.items():
            meshes = document.findall(".//mesh")
            self.assertTrue(meshes)
            for mesh in meshes:
                filename = mesh.attrib["filename"]
                with self.subTest(asset=name, mesh=filename):
                    self.assertTrue((ASSET_PATHS[name].parent / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
