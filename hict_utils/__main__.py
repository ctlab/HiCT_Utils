#  MIT License
#
#  Copyright (c) 2021-2024. Aleksandr Serdiukov, Anton Zamyatin, Aleksandr Sinitsyn, Vitalii Dravgelis and Computer Technologies Laboratory ITMO University team.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import argparse
import sys
from typing import Callable, Dict
from hict_utils.hict_to_cool import export_to_cooler
from hict_utils.cool_to_hict import flatten_conv
from hict_utils.export_assembly import export_assembly

commands_entrypoints: Dict[str, Callable] = {
    'convert': flatten_conv.main,
    'export': export_to_cooler.main,
    'assemble': export_assembly.main
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
