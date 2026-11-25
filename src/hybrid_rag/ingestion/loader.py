"""Filesystem discovery only; Frontmatter extraction remains in the Cleaner."""
from pathlib import Path


def discover(root: Path, products=("langchain", "langgraph", "deepagents"), include_md=False):
    for product in sorted(set(products)):
        directory = root / "oss" / product
        if not directory.is_dir():
            raise ValueError(f"Missing product directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in ({".mdx", ".md"} if include_md else {".mdx"}):
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"Source escapes trusted root: {path}")
                yield product, path
