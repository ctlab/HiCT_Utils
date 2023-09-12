import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import h5py
import numpy as np
from h5py import Dataset
from scipy.sparse import coo_matrix

from readerwriterlock import rwlock
from hict.core.common import ContigDirection

from hict.core.AGPProcessor import *

import tqdm

import gzip

import multiprocessing
import multiprocessing.managers


def is_sorted(a: np.ndarray) -> bool:
    return np.all(a[:-1] <= a[1:])

compression_to_extension = {"none": "fasta", "gzip": "fa.gz"}


class AssemblyExporter(object):

    def __init__(
        self,
        fasta_file_path: Path,        
        mp_manager: Optional[multiprocessing.managers.SyncManager] = None
    ) -> None:
        self.fasta_file_path: Path = fasta_file_path
        self.mp_manager = mp_manager if mp_manager is not None else multiprocessing.Manager()
        if (str(fasta_file_path).endswith(".gz") or str(fasta_file_path).endswith(".gzip")):
            with gzip.open(fasta_file_path, mode="rt") as uncompressed_fasta:
                self.record_dict = SeqIO.to_dict(SeqIO.parse(uncompressed_fasta, "fasta"))
        else:
            self.record_dict = SeqIO.index(str(fasta_file_path.absolute()), "fasta")

    def export_assembly(
        self,
        agp_file_path: Path,
        output_file_path: Path,
        compression: str,
        compression_level: int
    ):
        agpParser: AGPparser = AGPparser(agp_file_path.absolute())
        print(f"Will read AGP from {str(agp_file_path.absolute())}")
        contig_records = agpParser.getAGPContigRecords()
        scaffold_records = agpParser.getAGPScaffoldRecords()
        
        ctg_position: int = 0
        
        def get_contigs_in_scaffold(scaffold_record):
            nonlocal ctg_position
            if ctg_position < len(contig_records):
                ctg_record = contig_records[ctg_position]
                s = self.record_dict[ctg_record.name][ctg_record.start_position-1 : ctg_record.end_position-1]
                if ctg_record.direction == ContigDirection.REVERSED:
                    s = s.reverse_complement()
                yield str(s.seq)
                if contig_records[ctg_position].name == scaffold_record.end_ctg:
                    return
                else:
                    ctg_position += 1
            
        def get_scaffold_records():
            for scaffold in tqdm.tqdm(scaffold_records, desc="Scaffolds", leave=False, dynamic_ncols=True):
                sequence = "".join(get_contigs_in_scaffold(scaffold))
                record = SeqRecord(
                    seq=Seq(sequence),
                    id=scaffold.name,
                    description="",
                    name=""
                )
                yield record
            
        try:
            if compression == "none":
                f = open(output_file_path, mode="wt", encoding="utf-8")
            elif compression == "gzip":
                f = gzip.open(output_file_path, mode="wt", compresslevel=compression_level, encoding="utf-8")
            else:
                raise RuntimeError(f"Unknown compression type: {compression}")
            
            SeqIO.write(get_scaffold_records(), f, format="fasta-2line")
                            
        finally:
            f.close()


def main(cmdline: Optional[List[Any]]):
    parser = argparse.ArgumentParser(
        description="Export an assembly defined by AGP file with given FASTA file", prefix_chars="-+"
    )
    parser.add_argument("-c", "--compression", default="gzip", choices=[
                        "gzip", "none"], dest="compression", help="Output FASTA compression algorithm")
    parser.add_argument("-a", "--agp", required=True, dest="agp_file", type=Path, help="AGP file with assembly")
    parser.add_argument("-l", "--compression-level", dest="level", type=int, default=9, help="Compression level")
    parser.add_argument("input", help="Input FASTA file path (uncompressed or gzip-compressed with file name ending in .gz)",
                        type=Path)
    parser.add_argument(
        "-o", "--output", help="Output assembly file path (defaults to input file + agp name)", dest="output")

    args = (
        parser.parse_args(cmdline)
        if cmdline is not None
        else parser.parse_args()
    )

    print(f"args object: {args}")

    exporter = AssemblyExporter(
        args.input              
    )
    
    compression = args.compression.lower()
    extension = compression_to_extension[compression]
    
    output_file_path = args.output
    if output_file_path is None:
        output_file_path = Path(Path(args.input).parent, f"{Path(args.input).name}.agp_{Path(args.agp_file).name}.{extension}")
    
    exporter.export_assembly(
        args.agp_file,
        output_file_path,
        compression,
        args.level
    )