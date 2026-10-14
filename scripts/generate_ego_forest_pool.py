"""Freeze the v1 map pool before training; never select by policy performance."""
import argparse
import json
from pathlib import Path
from envs.map_atlas.forest_pool import build_pool, DEFAULT_MANIFEST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace frozen manifest: {output}')
    payload = build_pool()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
    print(json.dumps(dict(path=str(output), fingerprint=payload['fingerprint'],
                         sizes={k: len(v) for k, v in payload['maps'].items()}, rejected=payload['rejected']), indent=2))


if __name__ == '__main__':
    main()
