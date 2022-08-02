import argparse
import sys
from typing import Callable, Dict
from hict_utils.cool_to_hict import flatten_conv

commands_entrypoints: Dict[str, Callable] = {
    'convert': flatten_conv.main,
}


def main():
    parser = argparse.ArgumentParser(
        description="HiCT utilities package", prefix_chars="-+",
        epilog="Visit https://github.com/ctlab/HiCT for more info."
    )
    parser.add_argument(
        "tool",
        help="Tool to run",
        choices=commands_entrypoints.keys(),
    )
    args = parser.parse_args(sys.argv[1:2])
    subroutine: Callable
    try:
        subroutine: Callable = commands_entrypoints[args.tool]
    except KeyError:
        print(f"Unrecognized tool: {args.tool}")
        parser.print_help()
        exit(1)
    subroutine(sys.argv[2:])


if __name__ == '__main__':
    main()
