"""Resolve audio paths in a ParaBridge training jsonl.

The shipped sample data uses paths relative to the repository root (e.g.
``data/audio/foo.wav``) so the dataset is portable. ms-swift, however, passes
audio paths through to the audio loader as-is, so it is safest to feed it a
jsonl whose ``audios`` entries are absolute paths.

Usage:
    python parabridge/prepare_data.py \
        --input data/sample_train.jsonl \
        --output data/sample_train.resolved.jsonl \
        --root  $(pwd)

Any path that already starts with ``/`` is left unchanged, so jsonl files that
mix absolute and relative entries are also handled correctly.
"""
import argparse
import json
import os


def resolve(path: str, root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='input jsonl with possibly relative audio paths')
    p.add_argument('--output', required=True, help='output jsonl with absolute audio paths')
    p.add_argument('--root', default=os.getcwd(),
                   help='root used to resolve relative paths (defaults to cwd)')
    args = p.parse_args()

    n = 0
    with open(args.input, 'r', encoding='utf-8') as fin, \
            open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audios = rec.get('audios') or []
            rec['audios'] = [resolve(a, args.root) for a in audios]
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
    print(f'Resolved {n} records: {args.input} -> {args.output}')


if __name__ == '__main__':
    main()
