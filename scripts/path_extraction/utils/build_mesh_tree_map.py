#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from lxml import etree


def build_tree_map(xml_path: Path) -> dict:
    if not xml_path.exists():
        raise FileNotFoundError(f"MeSH XML file not found: {xml_path}")
    tree_map = {}
    context = etree.iterparse(str(xml_path), events=("end",), tag="DescriptorRecord")
    for _, elem in context:
        ui = elem.findtext(".//DescriptorUI")
        name = elem.findtext(".//DescriptorName/String")
        if ui and name:
            for tn_elem in elem.findall(".//TreeNumber"):
                tree_num = tn_elem.text
                if tree_num:
                    tree_map[tree_num] = {"ui": ui, "name": name}
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    return tree_map


def main():
    parser = argparse.ArgumentParser(description="Build MeSH tree map (tree number -> {ui,name}) from desc XML")
    parser.add_argument("--xml-file", required=True, help="Path to desc{year}.xml (e.g., desc2025.xml)")
    parser.add_argument("--output-pkl", required=True, help="Path to write mesh_tree_map_{year}.pkl")
    args = parser.parse_args()

    xml_path = Path(args.xml_file)
    out_path = Path(args.output_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tree_map = build_tree_map(xml_path)
    with open(out_path, "wb") as f:
        pickle.dump(tree_map, f)
    print(f"Saved tree map with {len(tree_map)} entries to {out_path}")


if __name__ == "__main__":
    main()
