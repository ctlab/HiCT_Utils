import argparse
import copy
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import h5py
import numpy as np
from h5py import Dataset
from scipy.sparse import coo_matrix

from hict.core.common import ContigDirection, ContigHideType, ContigDescriptor, ATUDescriptor, ATUDirection, StripeDescriptor

def save_indirect_block(
    row_stripe_id: np.int64,
    pixel_row_stripes: np.ndarray,
    pixel_col_stripes: np.ndarray,
    pixel_intra_stripe_row: np.ndarray,
    pixel_intra_stripe_col: np.ndarray,
    ordered_values: np.ndarray,
    block_datasets: Tuple[Dataset, Dataset, Dataset, Dataset, Dataset, Dataset],
    current_sparse_offset: np.int64,
    current_dense_offset: np.int64,
    stripes: List[StripeDescriptor],
    submatrix_size: np.int64,
) -> Tuple[np.int64, np.int64]:
    (block_rows_ds, block_cols_ds, block_vals_ds, block_offset_ds,
     block_length_ds, dense_blocks_ds) = block_datasets

    # np.where(np.roll(pixel_col_stripes, 1) != pixel_col_stripes)[0]
    block_start_indices = np.where(
        pixel_col_stripes[:-1] != pixel_col_stripes[1:]
    )[0] + 1
    # Last element should store length of the pixel table
    block_start_indices = np.hstack(((0,), block_start_indices, len(pixel_col_stripes)))
    block_count: np.int64 = len(block_start_indices)-1
    block_lengths = block_start_indices[1:] - block_start_indices[:-1]
    stripe_count: np.int64 = np.int64(len(stripes))

    for block_index in range(block_count):
        block_start_index: np.int64 = block_start_indices[block_index]
        block_col_stripe_id = pixel_col_stripes[block_start_index]
        block_index_in_datasets = (
            row_stripe_id*stripe_count + block_col_stripe_id
        )
        block_nonzero_element_count = block_lengths[block_index]
        if block_nonzero_element_count <= 0:
            continue
        block_rows = pixel_intra_stripe_row[block_start_index:
                                            block_start_index+block_nonzero_element_count]
        block_cols = pixel_intra_stripe_col[block_start_index:
                                            block_start_index+block_nonzero_element_count]
        block_vals = ordered_values[block_start_index:
                                    block_start_index+block_nonzero_element_count]

        if (
            block_nonzero_element_count >= (
                (submatrix_size * submatrix_size) // 2)
        ):
            block_offset_ds[block_index_in_datasets] = - \
                current_dense_offset - 1
            block_length_ds[block_index_in_datasets] = block_nonzero_element_count
            dense_blocks_ds.resize(1 + current_dense_offset, axis=0)
            mx_coo = coo_matrix(
                (
                    block_vals,
                    (
                        block_rows,
                        block_cols
                    )
                ),
                shape=(submatrix_size, submatrix_size)
            )
            dense_blocks_ds[current_dense_offset, 0, :, :] = mx_coo.toarray()
            current_dense_offset += 1
        else:
            block_offset_ds[block_index_in_datasets] = current_sparse_offset
            block_length_ds[block_index_in_datasets] = block_nonzero_element_count
            block_rows_ds[current_sparse_offset:current_sparse_offset +
                          block_nonzero_element_count] = block_rows
            block_cols_ds[current_sparse_offset:current_sparse_offset +
                          block_nonzero_element_count] = block_cols
            block_vals_ds[current_sparse_offset:current_sparse_offset +
                          block_nonzero_element_count] = block_vals
            current_sparse_offset += block_nonzero_element_count

    return current_sparse_offset, current_dense_offset


def dump_stripe_data(
        src_file: h5py.File,
        dst_file: h5py.File,
        submatrix_size: np.int64,
        resolution: np.int64,
        path_to_name_and_length: str = '/chroms/',
        additional_dataset_creation_args: Optional[dict] = None
) -> List[StripeDescriptor]:
    assert len(src_file[f'{path_to_name_and_length}/length']
               ) == len(src_file[f'{path_to_name_and_length}/name'])
    stripes_group: h5py.Group = dst_file.create_group(
        f'/resolutions/{resolution}/stripes')
    bins_group: h5py.Group = src_file[f'/resolutions/{resolution}/bins']
    bin_id_to_initial_contig_id: h5py.Dataset = bins_group['chrom']
    bin_weights: Optional[h5py.Dataset] = bins_group['weight'] if 'weight' in bins_group.keys(
    ) else None

    bin_count = len(bin_id_to_initial_contig_id)
    stripe_count: np.int64 = np.int64(
        (bin_count // submatrix_size) + min(bin_count % submatrix_size, 1)
    )

    stripes: List[StripeDescriptor] = [
        StripeDescriptor.make_stripe_descriptor(
            stripe_index,
            min(submatrix_size, bin_count - stripe_index*submatrix_size),
            np.array(
                bin_weights[
                    (stripe_index*submatrix_size):
                        min((1+stripe_index)*submatrix_size, bin_count)
                ]
            ) if bin_weights is not None else np.ones(submatrix_size, dtype=np.float64)
        )
        for stripe_index in range(stripe_count)
    ]

    if additional_dataset_creation_args is None:
        additional_dataset_creation_args = {}
    stripes_group.create_dataset('stripe_length_bins',
                                 data=[stripe.stripe_length_bins for stripe in stripes], dtype=np.int64,
                                 **additional_dataset_creation_args)
    stripes_group.create_dataset('stripes_bin_weights',
                                 shape=(len(stripes),
                                        submatrix_size),
                                 maxshape=(None, submatrix_size),
                                 data=[
                                     np.pad(
                                         stripe.bin_weights,
                                         [(0, submatrix_size -
                                           len(stripe.bin_weights))],
                                         mode='constant',
                                         constant_values=1.0
                                     ) for stripe in stripes],
                                 dtype=np.float64,
                                 **additional_dataset_creation_args)
    return stripes


def dump_contig_data(
        src_file: h5py.File,
        dst_file: h5py.File,
        path_to_name_and_length: str,
        resolutions: List[np.int64],
        stripes: Dict[np.int64, List[StripeDescriptor]],
        submatrix_size: np.int64,
        additional_dataset_creation_args: Optional[dict] = None
) -> List[ContigDescriptor]:  # np.ndarray:
    # TODO: Maybe in .mcool different contigs may be present/not present at different resolutions
    anyresolution: np.int64 = resolutions[0]
    contig_info_group: h5py.Group = dst_file.create_group('/contig_info/')
    contig_info_group.copy(
        src_file[f'{path_to_name_and_length}/name'], 'contig_name')
    contig_names: h5py.Dataset = src_file[f'{path_to_name_and_length}/name']

    contig_count: np.int64 = len(contig_info_group['contig_name'])

    if additional_dataset_creation_args is None:
        additional_dataset_creation_args = {}

    contig_info_group.create_dataset('contig_direction',
                                     data=[ContigDirection.FORWARD.value for _ in range(
                                         0, contig_count)],
                                     **additional_dataset_creation_args)
    contig_info_group.create_dataset('ordered_contig_ids', data=list(range(0, contig_count)),
                                     dtype=np.int64, **additional_dataset_creation_args)
    contig_info_group.create_dataset('contig_scaffold_id', data=[-1 for _ in range(0, contig_count)],
                                     dtype=np.int64, **additional_dataset_creation_args)

    src_contig_chrom_offset: h5py.Dataset = src_file[
        f'/resolutions/{anyresolution}/indexes/chrom_offset']
    src_bin_ends: h5py.Dataset = src_file[f'/resolutions/{anyresolution}/bins/end']

    if additional_dataset_creation_args is None:
        additional_dataset_creation_args = {}

    contig_id_to_contig_length_bp: np.ndarray

    if 'length' in src_file[path_to_name_and_length].keys():
        contig_info_group.copy(
            src_file[f'{path_to_name_and_length}/length'], 'contig_length_bp')
        contig_id_to_contig_length_bp = src_file[f'{path_to_name_and_length}/length'][:]
    else:
        contig_length: np.ndarray = np.zeros(contig_count, dtype=np.int64)

        contig_length[:-1] = src_bin_ends[src_contig_chrom_offset[1:-1] - 1]
        contig_length[-1] = src_bin_ends[-1]

        contig_info_group.create_dataset('contig_length_bp', data=contig_length, dtype=np.int64,
                                         **additional_dataset_creation_args)
        contig_id_to_contig_length_bp = contig_length[:]

    contig_start_bins_at_resolution: Dict[np.int64, np.ndarray] = dict()
    resolution_to_contig_length_bins: Dict[np.int64, np.ndarray] = dict()

    for resolution in resolutions:
        contig_start_bins: h5py.Dataset = src_file[f'/resolutions/{resolution}/indexes/chrom_offset'].astype(
            np.int64)
        contig_start_bins_at_resolution[resolution] = contig_start_bins
        contig_length_bins_ds: np.ndarray = contig_start_bins[1:] - \
            contig_start_bins[:-1]
        resolution_to_contig_length_bins[resolution] = contig_length_bins_ds

    for resolution in resolutions:
        assert np.all(
            resolution_to_contig_length_bins[resolution] > 0
        ),        "Zero-length contigs are present??"

    def generate_atus_for_contig(contig_id: np.int64, resolution: np.int64) -> List[ATUDescriptor]:
        start_bin: np.int64 = contig_start_bins_at_resolution[resolution][contig_id]
        #ctg_length: np.int64 = resolution_to_contig_length_bins[resolution][contig_id]
        end_bin: np.int64 = start_bin + \
            resolution_to_contig_length_bins[resolution][contig_id]
            
        start_stripe_id = start_bin // submatrix_size

        atus: List[ATUDescriptor] = [
            ATUDescriptor.make_atu_descriptor(
                stripes[resolution][start_stripe_id],
                start_bin % submatrix_size,
                submatrix_size if ((start_bin // submatrix_size) < (end_bin // submatrix_size)) else
                1+((end_bin-1) % submatrix_size),
                ATUDirection.FORWARD
            )
        ]
        
        start_bin = ((start_bin // submatrix_size)+1)*submatrix_size

        equal_parts_count: np.int64 = (end_bin - start_bin) // 256

        atus.extend((
            ATUDescriptor.make_atu_descriptor(
                stripes[resolution][start_stripe_id + part + 1],
                0,
                submatrix_size,
                ATUDirection.FORWARD
            ) for part in range(equal_parts_count)
        ))

        start_bin += (len(atus)-1)*submatrix_size

        # while start_bin < end_bin:
        #     atus.append(ATUDescriptor.make_atu_descriptor(
        #         stripes[resolution][start_bin // submatrix_size],
        #         start_bin % submatrix_size,
        #         ((min(submatrix_size, end_bin - start_bin)-1)%submatrix_size)+1,
        #         ATUDirection.FORWARD
        #     ))
        #     start_bin += submatrix_size
        if start_bin < end_bin:
            atus.append(ATUDescriptor.make_atu_descriptor(
                    stripes[resolution][start_stripe_id + 1 + equal_parts_count],
                    0,
                    1+((end_bin-1)%submatrix_size),
                    ATUDirection.FORWARD
                ))
            
        return atus

    contigs: List[ContigDescriptor] = [
        ContigDescriptor.make_contig_descriptor(
            contig_id=contig_id,
            contig_name=contig_names[contig_id],
            contig_length_bp=contig_id_to_contig_length_bp[contig_id],
            contig_length_at_resolution={
                resolution: resolution_to_contig_length_bins[resolution][contig_id] for resolution in resolutions},
            contig_presence_in_resolution={resolution: ContigHideType.AUTO_SHOWN if resolution_to_contig_length_bins[
                resolution][contig_id] > 1 else ContigHideType.AUTO_HIDDEN for resolution in resolutions},
            atus={resolution: generate_atus_for_contig(contig_id, resolution) for resolution in resolutions},
            # scaffold_id=None
        ) for contig_id in range(0, contig_count)
    ]

    for resolution in resolutions:
        contigs_group: h5py.Group = dst_file.create_group(
            f'/resolutions/{resolution}/contigs')
        atl_group: h5py.Group=dst_file.create_group(
            f'/resolutions/{resolution}/atl')

        contigs_group.create_dataset(
            'contig_length_bins',
            data = [contig.contig_length_at_resolution[resolution]
                  for contig in contigs],
            dtype = np.int64,
            **additional_dataset_creation_args
        )
        contigs_group.create_dataset(
            'contig_hide_type',
            data=[contig.presence_in_resolution[resolution].value for contig in contigs],
            dtype=np.int8,
            **additional_dataset_creation_args
        )

        contig_id_to_atus_id: List[Tuple[np.int64, np.int64]] = []
        all_atus_at_resolution: List[ATUDescriptor] = []
        current_atu_id: np.int64 = np.int64(0)
        for contig in contigs:
            contig_id_to_atus_id.extend((
                (contig.contig_id, current_atu_id + i) for i in range(len(contig.atus[resolution]))
            ))
            current_atu_id += np.int64(len(contig.atus[resolution]))
            all_atus_at_resolution.extend(contig.atus[resolution])

        atu_array: np.ndarray = np.array(
            [
                (
                    atu.stripe_descriptor.stripe_id,
                    atu.start_index_in_stripe_incl,
                    atu.end_index_in_stripe_excl,
                    atu.direction.value
                ) for atu in all_atus_at_resolution
            ], dtype=np.int64
        )

        contigs_atu_array: np.ndarray = np.array(
            contig_id_to_atus_id,
            dtype=np.int64
        )

        contigs_group.create_dataset(
            "atl",
            data=contigs_atu_array,
            **additional_dataset_creation_args
        )

        atl_group.create_dataset(
            "basis_atu",
            data=atu_array,
            **additional_dataset_creation_args
        )

    return contigs, contig_id_to_contig_length_bp


def is_sorted(a: np.ndarray) -> bool:
    return np.all(a[:-1] <= a[1:])


def cool_flatten_convert(
        src_file_path: str,
        dst_file_path: str,
        get_name_and_length_path: Callable[[np.int64], str],
        resolutions: Optional[List[np.int64]] = None,
        additional_dataset_creation_args: Optional[dict] = None
):
    submatrix_size: np.int64 = 256
    hdf5_max_chunk_size: np.int64 = 32 * 1024 * 1024 * 8
    if additional_dataset_creation_args is None:
        additional_dataset_creation_args = {}

    with h5py.File(name=src_file_path, mode='r') as src_file:
        if resolutions is None:
            resolutions = [np.int64(sdn) for sdn in filter(
                lambda s: s.isnumeric(), src_file['resolutions'].keys())]
        with h5py.File(name=dst_file_path, mode='w') as dst_file:
            resolution_to_stripes: Dict[np.int64,
                                        List[StripeDescriptor]] = dict()
            for resolution in sorted(resolutions, reverse=True):
                print(f"Resolution {resolution} out of {resolutions}")
                stripes = dump_stripe_data(
                    src_file,
                    dst_file,
                    submatrix_size,
                    resolution,
                    get_name_and_length_path(resolution),
                    additional_dataset_creation_args
                )
                resolution_to_stripes[resolution] = stripes
                dst_file['resolutions'].attrs.create("hict_version", "0.1.3.1")
                res_group: h5py.Group = dst_file.create_group(
                    f'resolutions/{resolution}/treap_coo')
                res_group.attrs.create(
                    name='dense_submatrix_size', data=submatrix_size)
                res_group.attrs.create(
                    name='hdf5_max_chunk_size', data=hdf5_max_chunk_size)
                src_bins_count = len(
                    src_file[f'resolutions/{resolution}/bins/end'])
                src_pixels = src_file[f'resolutions/{resolution}/pixels']
                src_pixel_row: h5py.Dataset = src_pixels['bin1_id']
                src_pixel_col: h5py.Dataset = src_pixels['bin2_id']
                src_pixel_val: h5py.Dataset = src_pixels['count']
                nonzero_pixel_count: np.int64 = len(src_pixel_row)
                res_group.attrs.create(name='bins_count', data=src_bins_count)
                # Maximum count of stripes horizontally [covering rows 0..submatrix_size)
                stripes_count = len(stripes)             
                all_rows_start_indices: h5py.Dataset = src_file[
                    f'resolutions/{resolution}/indexes/bin1_offset']

                print(
                    f"bins_count: {src_bins_count}, "
                    f"stripes_count: {stripes_count}, "
                )

                ds_creation_args: dict = copy.deepcopy(
                    additional_dataset_creation_args)
                ds_creation_args['chunks'] = True

                block_rows: h5py.Dataset = res_group.create_dataset(
                    'block_rows', shape=(nonzero_pixel_count,),
                    dtype=np.int64,
                    **ds_creation_args
                )
                block_cols: h5py.Dataset = res_group.create_dataset(
                    'block_cols', shape=(nonzero_pixel_count,),
                    dtype=np.int64,
                    **ds_creation_args
                )
                block_vals: h5py.Dataset = res_group.create_dataset(
                    'block_vals', shape=(nonzero_pixel_count,),
                    dtype=src_pixel_val.dtype,
                    **ds_creation_args
                )
                total_block_count: np.int64 = stripes_count * stripes_count
                block_offset: h5py.Dataset = res_group.create_dataset(
                    'block_offset', shape=(total_block_count,),
                    dtype=np.int64,
                    **ds_creation_args
                )
                block_length: h5py.Dataset = res_group.create_dataset(
                    'block_length', shape=(total_block_count,),
                    dtype=np.int64,
                    **ds_creation_args
                )

                dense_blocks: h5py.Dataset = res_group.create_dataset(
                    'dense_blocks',
                    shape=(1, 1, submatrix_size, submatrix_size),
                    maxshape=(None, 1, submatrix_size, submatrix_size),
                    dtype=src_pixel_val.dtype,
                    **ds_creation_args,
                )

                block_datasets: Tuple[Dataset, Dataset, Dataset, Dataset, Dataset, Dataset] = (
                    block_rows, block_cols, block_vals, block_offset, block_length, dense_blocks)

                res_group.attrs.create(
                    name='stripes_count', data=stripes_count)

                current_sparse_offset: np.int64 = 0
                current_dense_offset: np.int64 = 0

                for vstripe_l in range(0, stripes_count):
                    singlerowstripe_pixel_row, singlerowstripe_pixel_col, singlerowstripe_pixel_val = (
                        src_pixel_row[all_rows_start_indices[vstripe_l*submatrix_size]:all_rows_start_indices[min(
                            (vstripe_l+1)*submatrix_size, len(all_rows_start_indices)-1)]],
                        src_pixel_col[all_rows_start_indices[vstripe_l*submatrix_size]:all_rows_start_indices[min(
                            (vstripe_l+1)*submatrix_size, len(all_rows_start_indices)-1)]],
                        src_pixel_val[all_rows_start_indices[vstripe_l*submatrix_size]:all_rows_start_indices[min(
                            (vstripe_l+1)*submatrix_size, len(all_rows_start_indices)-1)]],
                    )
                    if len(singlerowstripe_pixel_row) <= 0:
                        print("Zero-length stripe subarray")
                        continue
                    pixel_row_stripes = singlerowstripe_pixel_row // submatrix_size
                    pixel_col_stripes = singlerowstripe_pixel_col // submatrix_size
                    pixel_intra_stripe_row = singlerowstripe_pixel_row % submatrix_size
                    pixel_intra_stripe_col = singlerowstripe_pixel_col % submatrix_size

                    assert (
                        np.all(pixel_row_stripes == vstripe_l)
                    ), "Single row stripe contains pixels for different stripe??"

                    assert (
                        np.all(pixel_row_stripes == pixel_row_stripes[0])
                    ), "Single row stripe contains pixels for multiple stripes??"

                    chunked_order = np.lexsort(
                        (pixel_intra_stripe_col, pixel_intra_stripe_row, pixel_row_stripes, pixel_col_stripes))

                    current_sparse_offset, current_dense_offset = save_indirect_block(
                        vstripe_l,
                        pixel_row_stripes[chunked_order],
                        pixel_col_stripes[chunked_order],
                        pixel_intra_stripe_row[chunked_order],
                        pixel_intra_stripe_col[chunked_order],
                        singlerowstripe_pixel_val[chunked_order],
                        block_datasets,
                        current_sparse_offset,
                        current_dense_offset,
                        stripes,
                        submatrix_size
                    )
                dst_file.flush()

            contigs, contig_id_to_contig_length_bp = dump_contig_data(
                src_file,
                dst_file,
                get_name_and_length_path(resolutions[0]),
                resolutions,
                resolution_to_stripes,
                submatrix_size,
                additional_dataset_creation_args
            )
            dst_file.flush()


def main(cmdline: Optional[List[Any]]):
    def cool_file_checker(parser: argparse.ArgumentParser, filename: str):
        try:
            with h5py.File(name=filename, mode='r') as f:
                return filename
        except IOError:
            parser.error(f"{filename} does not point to correct HDF5 file")

    parser = argparse.ArgumentParser(
        description="Convert .mcool file into HiCT format", prefix_chars="-+"
    )
    parser.add_argument("-c", "--compression", default="lzf", choices=[
                        "lzf", "gzip", "none"], help="Select HDF5 dataset compression algorithm")
    parser.add_argument("-t", "--layout", default="default", choices=[
                        "default", "globalchroms"], help="Non-standardized layouts support")
    parser.add_argument("-n", "--no-shuffle", action="store_false",
                        help="Disable HDF5 shuffle filter", dest="shuffle")
    parser.add_argument("-r", "--resolutions", nargs='*',
                        type=int, help="Select resolutions that should be converted (if not specified, converts all that are found)", dest="resolutions")
    parser.add_argument("input", help="Input file path",
                        type=lambda f: cool_file_checker(parser, f))
    parser.add_argument(
        "-o", "--output", help="Output file path", dest="output")

    args = (
        parser.parse_args(cmdline)
        if cmdline is not None
        else parser.parse_args()
    )

    additional_dataset_creation_args: Dict = {
        'shuffle': args.shuffle,
    }
    if (args.compression.lower() != "none"):
        additional_dataset_creation_args['compression'] = (
            args.compression.lower()
        )
    
    path_to_name_and_length = {
        "default": (lambda r: f'/resolutions/{str(r)}/chroms'),
        "globalchroms": (lambda _: '/chroms'),
    }[args.layout]

    print(f"args object: {args}")

    cool_flatten_convert(
        args.input,
        args.output if args.output is not None else f"{args.input}.hict.hdf5",
        path_to_name_and_length,
        resolutions=[
            np.int64(r)
            for r in args.resolutions
        ] if args.resolutions is not None else None,
        additional_dataset_creation_args=additional_dataset_creation_args
    )


# cool_flatten_convert(
#     "..\\HiCT_Server\\data\\mat18_100k.mcool",
#     "..\\HiCT_Server\\data\\mat18_100k.hict.hdf5",
#     lambda r: f'/resolutions/{str(r)}/chroms',
#     resolutions=None,
#     additional_dataset_creation_args=None
# )
