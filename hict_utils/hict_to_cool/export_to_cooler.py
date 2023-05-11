import argparse
# import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import h5py
import numpy as np
from scipy.sparse import coo_array

# from readerwriterlock import rwlock

# import multiprocessing
# import multiprocessing.managers

# import ext_sort
# import bz2
# import tempfile
# import numpy.lib.recfunctions as npr
# import os
import pandas as pd
from hict.api.ContactMatrixFacet import ContactMatrixFacet

from hict.core.chunked_file import ChunkedFile
from hict.core.common import QueryLengthUnit, ScaffoldDescriptor

import cooler
# import io
# import csv

# class CSVSerializer(ext_sort.Serializer):

#     def __init__(self, writer):
#         super().__init__(csv.writer(io.TextIOWrapper(writer, write_through=True)))

#     def write(self, item):
#         return self._writer.writerow(item)


# class CSVDeserializer(ext_sort.Deserializer):

#     def __init__(self, reader):
#         super().__init__(csv.reader(io.TextIOWrapper(reader)))

#     def read(self):
#         return next(self._reader)

class HiCTToCoolerConverter(object):

    def __init__(
        self,
        chunked_file: ChunkedFile,
        output_file_path: Path
    ) -> None:
        self.chunked_file = chunked_file
        self.output_file_path = output_file_path

    def convert(
        self,
        resolutions: Optional[List[int]] = None,
        maximum_fetch_size_bytes: int = 256*1024*1024*8
    ):
        if resolutions is None:
            resolutions = [min(self.chunked_file.resolutions)]

        ds_args = {
            "compression": "gzip",
            "shuffle": True,
            "chunks": True,
        }

        # chroms:
        _, scaffolds = ContactMatrixFacet.get_assembly_info(self.chunked_file)
        scaffold_count: np.int64 = np.int64(len(scaffolds))
        scaffold_length_bp = np.array(
            tuple(map(lambda t: t[1], scaffolds)), dtype=np.int64)

        unnamed_count: int = 0

        def get_name(sd: Optional[ScaffoldDescriptor]) -> str:
            nonlocal unnamed_count
            if sd is not None:
                return sd.scaffold_name
            else:
                unnamed_count += 1
                return f"unscaffolded_contig_region_{unnamed_count}"

        scaffold_names_map = map(get_name, map(lambda t: t[0], scaffolds))
        scaffold_names_nd = np.array(tuple(scaffold_names_map))
        # maximum_length = max(map(len, scaffold_names_map))

        for resolution in resolutions:
            # bins:
            # bins/chrom:
            scaffold_length_bp_prefix_sum = np.zeros(
                shape=(1+scaffold_count,), dtype=np.int64)
            scaffold_length_bins = np.zeros(
                shape=(scaffold_count,), dtype=np.int64)
            np.cumsum(scaffold_length_bp,
                      out=scaffold_length_bp_prefix_sum[1:], dtype=np.int64)
            for i in range(scaffold_count):
                es = self.chunked_file.contig_tree.expose_segment(
                    0,
                    scaffold_length_bp_prefix_sum[i],
                    scaffold_length_bp_prefix_sum[1+i],
                    units=QueryLengthUnit.BASE_PAIRS
                )
                scaffold_length_bins[i] = es.segment.get_sizes(
                )[0][resolution] if es.segment is not None else np.int64(0)
            scaffold_length_bins_prefix_sum = np.zeros(
                shape=(1+scaffold_count,), dtype=np.int64)
            np.cumsum(scaffold_length_bins,
                      out=scaffold_length_bins_prefix_sum[1:], dtype=np.int64)
            total_bin_length = scaffold_length_bins_prefix_sum[-1]
            bin_to_scaffold = np.zeros(
                shape=(total_bin_length), dtype=np.int32)
            for scaffold_index in range(len(scaffold_length_bins)):
                bin_to_scaffold[scaffold_length_bins_prefix_sum[i]:scaffold_length_bins_prefix_sum[1+i]] = scaffold_index
            # /resolutions/1000/bins/start:
            # /resolutions/1000/bins/end:
            bins_start = np.zeros(shape=(total_bin_length,), dtype=np.int32)
            bins_end = np.zeros(shape=(total_bin_length,), dtype=np.int32)
            for scaffold_index in range(scaffold_count):
                bin_ords = np.arange(
                    scaffold_length_bins[scaffold_index], dtype=np.int64)
                bins_start[scaffold_length_bins_prefix_sum[scaffold_index]:scaffold_length_bins_prefix_sum[1+scaffold_index]] = np.int32(resolution*bin_ords)
                bins_end[scaffold_length_bins_prefix_sum[scaffold_index]:scaffold_length_bins_prefix_sum[1+scaffold_index]] = np.int32(resolution*(1+bin_ords))
                bins_end[scaffold_length_bins_prefix_sum[1+scaffold_index]-1
                         ] = np.int32(scaffold_length_bp[scaffold_index])

            bins = pd.DataFrame.from_dict(
                {'chrom': scaffold_names_nd[bin_to_scaffold], 'start': bins_start, 'end': bins_end})

            val_dtype = self.chunked_file.dtype
            val_dtype_width_bytes: int = int(
                np.dtype(val_dtype).itemsize) if val_dtype is not None else 8
            number_of_blocks_to_fetch: int = int(max(1, maximum_fetch_size_bytes // (
                val_dtype_width_bytes * self.chunked_file.dense_submatrix_size[resolution] * self.chunked_file.dense_submatrix_size[resolution])))

            def fetch_chunk():
                for start_row_incl in range(0, total_bin_length, self.chunked_file.dense_submatrix_size[resolution]):
                    end_row_excl = min(
                        start_row_incl + self.chunked_file.dense_submatrix_size[resolution], total_bin_length)
                    for start_col_incl in range(start_row_incl, total_bin_length, self.chunked_file.dense_submatrix_size[resolution] * number_of_blocks_to_fetch):
                        end_col_excl = min(
                            start_col_incl + self.chunked_file.dense_submatrix_size[resolution] * number_of_blocks_to_fetch, total_bin_length)
                        dense, _, _ = self.chunked_file.get_submatrix(
                            resolution,
                            start_row_incl,
                            start_col_incl,
                            end_row_excl,
                            end_col_excl,
                            exclude_hidden_contigs=False
                        )
                        if start_row_incl == start_col_incl:
                            dense = np.triu(dense)
                        sparse = coo_array(dense, dtype=(
                            self.chunked_file.dtype if self.chunked_file.dtype is not None else np.int32))
                        rows = sparse.row + start_row_incl
                        cols = sparse.col + start_col_incl
                        # coo_record_row = np.rec.array(sparse.row + start_row_incl, dtype=('row', row_dtype))
                        # coo_record_rcv = npr.append_fields(coo_record_row, data=(sparse.col, sparse.data), names=('col', 'val'), dtypes=(col_dtype, val_dtype), usemask=False, asrecarray=True)
                        ind = np.lexsort((cols, rows))
                        coo_chunk = {
                            'bin1_id': rows[ind],
                            'bin2_id': cols[ind],
                            'count': sparse.data[ind]
                        }
                        # print("Yielding chunk", flush=True)
                        yield coo_chunk
                        print(f"Exported: {float(start_row_incl*total_bin_length + start_col_incl) / float(total_bin_length*total_bin_length)} Time: {datetime.now().strftime('%H:%M:%S')}", flush=True)

            #pixels_generator=(fetch_chunk(start_row_incl, start_col_incl) for )

            cooler.create_cooler(
                cool_uri=f"{str(self.output_file_path.absolute())}::/resolutions/{resolution}",
                bins=bins,
                pixels=fetch_chunk(),#(fetch_chunk(start_row_incl, start_col_incl) for),
                symmetric_upper=True,
                ordered=False,
                h5opts=ds_args
            )
            
    # def convert_extsort(
    #     self,
    #     resolutions: Optional[List[int]] = None,
    #     maximum_fetch_size_bytes: int = 256*1024*1024*8
    # ):
    #     global Ser, DeSer
    #     if resolutions is None:
    #         resolutions = [min(self.chunked_file.resolutions)]

    #     ds_args = {
    #         "compression": "gzip",
    #         "shuffle": True,
    #         "chunks": True,
    #     }

    #     # chroms:
    #     _, scaffolds = ContactMatrixFacet.get_assembly_info(self.chunked_file)
    #     scaffold_count: np.int64 = np.int64(len(scaffolds))
    #     scaffold_length_bp = np.array(
    #         tuple(map(lambda t: t[1], scaffolds)), dtype=np.int64)

    #     unnamed_count: int = 0

    #     def get_name(sd: Optional[ScaffoldDescriptor]) -> str:
    #         nonlocal unnamed_count
    #         if sd is not None:
    #             return sd.scaffold_name
    #         else:
    #             unnamed_count += 1
    #             return f"unscaffolded_contig_region_{unnamed_count}"

    #     scaffold_names_map = map(get_name, map(lambda t: t[0], scaffolds))
    #     scaffold_names_nd = np.array(tuple(scaffold_names_map))
    #     # maximum_length = max(map(len, scaffold_names_map))

    #     for resolution in resolutions:
    #         # bins:
    #         # bins/chrom:
    #         scaffold_length_bp_prefix_sum = np.zeros(
    #             shape=(1+scaffold_count,), dtype=np.int64)
    #         scaffold_length_bins = np.zeros(
    #             shape=(scaffold_count,), dtype=np.int64)
    #         np.cumsum(scaffold_length_bp,
    #                   out=scaffold_length_bp_prefix_sum[1:], dtype=np.int64)
    #         for i in range(scaffold_count):
    #             es = self.chunked_file.contig_tree.expose_segment(
    #                 0,
    #                 scaffold_length_bp_prefix_sum[i],
    #                 scaffold_length_bp_prefix_sum[1+i],
    #                 units=QueryLengthUnit.BASE_PAIRS
    #             )
    #             scaffold_length_bins[i] = es.segment.get_sizes(
    #             )[0][resolution] if es.segment is not None else np.int64(0)
    #         scaffold_length_bins_prefix_sum = np.zeros(
    #             shape=(1+scaffold_count,), dtype=np.int64)
    #         np.cumsum(scaffold_length_bins,
    #                   out=scaffold_length_bins_prefix_sum[1:], dtype=np.int64)
    #         total_bin_length = scaffold_length_bins_prefix_sum[-1]
    #         bin_to_scaffold = np.zeros(
    #             shape=(total_bin_length), dtype=np.int32)
    #         for scaffold_index in range(len(scaffold_length_bins)):
    #             bin_to_scaffold[scaffold_length_bins_prefix_sum[i]:scaffold_length_bins_prefix_sum[1+i]] = scaffold_index
    #         # /resolutions/1000/bins/start:
    #         # /resolutions/1000/bins/end:
    #         bins_start = np.zeros(shape=(total_bin_length,), dtype=np.int32)
    #         bins_end = np.zeros(shape=(total_bin_length,), dtype=np.int32)
    #         for scaffold_index in range(scaffold_count):
    #             bin_ords = np.arange(
    #                 scaffold_length_bins[scaffold_index], dtype=np.int64)
    #             bins_start[scaffold_length_bins_prefix_sum[scaffold_index]:scaffold_length_bins_prefix_sum[1+scaffold_index]] = np.int32(resolution*bin_ords)
    #             bins_end[scaffold_length_bins_prefix_sum[scaffold_index]:scaffold_length_bins_prefix_sum[1+scaffold_index]] = np.int32(resolution*(1+bin_ords))
    #             bins_end[scaffold_length_bins_prefix_sum[1+scaffold_index]-1
    #                      ] = np.int32(scaffold_length_bp[scaffold_index])

    #         bins = pd.DataFrame.from_dict(
    #             {'chrom': scaffold_names_nd[bin_to_scaffold], 'start': bins_start, 'end': bins_end})

    #         val_dtype = self.chunked_file.dtype
    #         val_dtype_width_bytes: int = int(
    #             np.dtype(val_dtype).itemsize) if val_dtype is not None else 8
    #         number_of_blocks_to_fetch: int = int(max(1, maximum_fetch_size_bytes // (
    #             val_dtype_width_bytes * self.chunked_file.dense_submatrix_size[resolution] * self.chunked_file.dense_submatrix_size[resolution])))

    #         row_dtype = np.int32
    #         row_dtype_width_bytes: int = int(np.dtype(row_dtype).itemsize) if row_dtype is not None else 4
    #         col_dtype = np.int32
    #         col_dtype_width_bytes: int = int(np.dtype(col_dtype).itemsize) if col_dtype is not None else 4
            

    #         with tempfile.NamedTemporaryFile("w+b") as tmpf:
    #             with bz2.open(tmpf, mode="wt", compresslevel=5) as cstream:            
    #                 for start_row_incl in range(0, total_bin_length, self.chunked_file.dense_submatrix_size[resolution]):
    #                     end_row_excl = min(
    #                         start_row_incl + self.chunked_file.dense_submatrix_size[resolution], total_bin_length)
    #                     for start_col_incl in range(start_row_incl, total_bin_length, self.chunked_file.dense_submatrix_size[resolution] * number_of_blocks_to_fetch):
    #                         end_col_excl = min(
    #                             start_col_incl + self.chunked_file.dense_submatrix_size[resolution] * number_of_blocks_to_fetch, total_bin_length)
    #                         dense, _, _ = self.chunked_file.get_submatrix(
    #                             resolution,
    #                             start_row_incl,
    #                             start_col_incl,
    #                             end_row_excl,
    #                             end_col_excl,
    #                             exclude_hidden_contigs=False
    #                         )
    #                         if start_row_incl == start_col_incl:
    #                             dense = np.triu(dense)
    #                         sparse = coo_array(dense, dtype=(
    #                             self.chunked_file.dtype if self.chunked_file.dtype is not None else np.int32))
    #                         rows = sparse.row + start_row_incl
    #                         cols = sparse.col + start_col_incl
                            
    #                         coo_record_row = np.rec.array(rows, dtype=[('bin1_id', row_dtype)])
    #                         coo_record_rcv = npr.append_fields(coo_record_row, data=[cols, sparse.data], names=['bin2_id', 'count'], dtypes=[col_dtype, val_dtype], usemask=False, asrecarray=True)
    #                         df = pd.DataFrame(coo_record_rcv)
    #                         df.to_csv(cstream, sep=',', header=None, index=None)
    #                         del coo_record_rcv
    #                         del coo_record_row
    #                         del sparse                            
                            
    #                         print(f"Exporting raw pixeltable: {float(start_row_incl*total_bin_length + start_col_incl) / float(total_bin_length*total_bin_length)} Time: {datetime.now().strftime('%H:%M:%S')}", flush=True)
                            
    #                 print(f"Exported raw pixeltable, preparing to sort. Time: {datetime.now().strftime('%H:%M:%S')}", flush=True)
                    
    #             tmpf.seek(0)
    #             with bz2.open(tmpf, mode="rt") as cstream:                       
                    
                    
    #                 with tempfile.NamedTemporaryFile("w+b") as tmpres:
    #                     with bz2.open(tmpres, mode="wb", compresslevel=5) as cresstream:
    #                         with bz2.open(tmpf, mode="rb") as cstream:
    #                             print(f"Launching external sorting. Time: {datetime.now().strftime('%H:%M:%S')}", flush=True)
    #                             ext_sort.sort(
    #                                 cstream,
    #                                 cresstream,
    #                                 Serializer=CSVSerializer,
    #                                 Deserializer=CSVDeserializer,
    #                                 chunk_size=(maximum_fetch_size_bytes // val_dtype_width_bytes),
    #                                 workers_cnt=os.cpu_count()
    #                             )
    #                             print(f"External sorting finished. Time: {datetime.now().strftime('%H:%M:%S')}", flush=True)
    #                     tmpres.seek(0)
    #                     with bz2.open(tmpres, mode="rt") as cresstream:
    #                         # Copy from compressed temporary file to Cooler datasets                            
    #                         def fetch_chunk():
    #                             #csv.reader(io.TextIOWrapper(cresstream))
    #                             buf = cresstream.readlines((maximum_fetch_size_bytes // val_dtype_width_bytes))
    #                             if buf:
    #                                 df = pd.read_csv(io.StringIO("".join(buf)), sep=',', header=None, index_col=False)
    #                                 rows = df.iloc[:, 0].values.astype(row_dtype)
    #                                 cols = df.iloc[:, 1].values.astype(col_dtype)
    #                                 vals = df.iloc[:, 2].values.astype(val_dtype)
    #                                 yield {
    #                                     'bin1_id': rows,
    #                                     'bin2_id': cols,
    #                                     'count': vals
    #                                 }
                                    
    #                         print(f"Starting cooler creation. Time: {datetime.now().strftime('%H:%M:%S')}", flush=True)

    #                         cooler.create_cooler(
    #                             cool_uri=f"{str(self.output_file_path.absolute())}::/resolutions/{resolution}",
    #                             bins=bins,
    #                             pixels=fetch_chunk(),
    #                             symmetric_upper=True,
    #                             ensure_sorted=True,
    #                             ordered=True,
    #                             h5opts=ds_args
    #                         )


def main(cmdline: Optional[List[Any]]):
    def cool_file_checker(parser: argparse.ArgumentParser, filename: str):
        try:
            with h5py.File(name=filename, mode='r') as f:
                return filename
        except IOError:
            parser.error(f"{filename} does not point to correct HDF5 file")
            
    def agp_file_checker(parser: argparse.ArgumentParser, filename: str):
        try:
            path = Path(filename).absolute()
            if path.is_file():
                return path
            else:
                parser.error(f"{filename} does not point to file")
        except IOError:
            parser.error(f"{filename} does not point to file")

    parser = argparse.ArgumentParser(
        description="Convert .hict.hdf5 into .mcool", prefix_chars="-+"
    )
    parser.add_argument("-r", "--resolutions", nargs='*',
                        type=int, help="Select resolutions that should be converted (if not specified, converts only the most high-resolution one)", dest="resolutions")
    parser.add_argument("-a", "--agp",
                        type=lambda f: agp_file_checker(parser, f), help="AGP file that should be applied to hict.hdf5 before export", dest="agp")
    # parser.add_argument("-e", "--ext-sort", action="store_true",
    #                     help="Use multithreaded ext-sort module instead of single-threaded Cooler implementation (might require significantly more disk space)", dest="ext_sort")
    parser.add_argument("input", help="Input file path",
                        type=lambda f: cool_file_checker(parser, f))
    parser.add_argument(
        "-o", "--output", help="Output file path", dest="output")

    args = (
        parser.parse_args(cmdline)
        if cmdline is not None
        else parser.parse_args()
    )

    print(f"args object: {args}")

    chunked_file: ChunkedFile

    try:
        chunked_file = ContactMatrixFacet.get_file_descriptor(
            Path(args.input).absolute())
        ContactMatrixFacet.open_file(chunked_file)
        
        if args.agp is not None:
            agp_path = Path(args.agp).absolute()
            print(f"Applying AGP assembly: {str(agp_path)}", flush=True)
            ContactMatrixFacet.load_assembly_from_agp(chunked_file, agp_path)
            print("AGP applied", flush=True)
        
        converter: HiCTToCoolerConverter = HiCTToCoolerConverter(
            chunked_file, output_file_path=Path(
                args.output if args.output is not None else f"{args.input}.mcool"
            ).absolute()
        )

        # if not args.ext_sort:
        converter.convert(
            resolutions=[
                int(r)
                for r in args.resolutions
            ] if args.resolutions is not None else None,
        )
        # else:
        #     converter.convert_extsort(
        #         resolutions=[
        #             int(r)
        #             for r in args.resolutions
        #         ] if args.resolutions is not None else None,
        #     )
    finally:
        del chunked_file
