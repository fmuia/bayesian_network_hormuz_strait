"""Command-line export/import between the elicitation store and JSON specs.

Usage:
    python -m src.elicitation.export.cli export --db <url> --network-id 1 --out spec.json
    python -m src.elicitation.export.cli import --db <url> --in spec.json --name hormuz
"""
# TODO(pack-separation): usage example uses the Hormuz pack id as a sample --name;
# harmless (it's an example) but worth genericising when packs are documented.

from __future__ import annotations

import argparse
import json
import sys

from ..db.session import create_all, make_engine, make_session_factory
from .network_spec import (
    cpts_to_network_spec,
    network_spec_to_cpts,
    spec_from_dict,
    spec_to_dict,
)


def _export(args: argparse.Namespace) -> int:
    engine = make_engine(args.db)
    Session = make_session_factory(engine)
    with Session() as session:
        spec = cpts_to_network_spec(session, args.network_id)
    with open(args.out, "w") as fh:
        json.dump(spec_to_dict(spec), fh, indent=2)
    print(f"exported network {args.network_id} -> {args.out} ({len(spec.nodes)} nodes)")
    return 0


def _import(args: argparse.Namespace) -> int:
    engine = make_engine(args.db)
    create_all(engine)
    Session = make_session_factory(engine)
    with open(args.infile) as fh:
        spec = spec_from_dict(json.load(fh))
    with Session() as session:
        network_id = network_spec_to_cpts(session, spec, args.name, topology=args.topology)
    print(f"imported {args.infile} -> network id {network_id} ({len(spec.nodes)} nodes)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="elicitation-export")
    sub = parser.add_subparsers(dest="command", required=True)

    p_exp = sub.add_parser("export", help="export stored CPTs to a JSON spec")
    p_exp.add_argument("--db", required=True)
    p_exp.add_argument("--network-id", type=int, required=True)
    p_exp.add_argument("--out", required=True)
    p_exp.set_defaults(func=_export)

    p_imp = sub.add_parser("import", help="import a JSON spec into the store")
    p_imp.add_argument("--db", required=True)
    p_imp.add_argument("--in", dest="infile", required=True)
    p_imp.add_argument("--name", required=True)
    p_imp.add_argument("--topology", default="latent_regime")
    p_imp.set_defaults(func=_import)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
