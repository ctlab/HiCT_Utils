import argparse
import copy
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from h5py import Dataset
from scipy.sparse import coo_matrix

from hict.core.common import ContigDirection, ContigHideType, ContigDescriptor, ATUDescriptor, ATUDirection
from hict.core.stripe_tree import *


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
    block_count: np.int64 = len(block_start_indices)
    # Last element should store length of the pixel table
    block_start_indices = np.append(
        block_start_indices, len(pixel_col_stripes))
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
            block_rows_ds[current_sparse_offset:current_sparse_offset +
                          block_nonzero_element_count] = block_rows
            block_cols_ds[current_sparse_offset:current_sparse_offset +
                          block_nonzero_element_count] = block_cols
            block_vals_ds[current_sparse_offset:current_sparse_offset +
                          block_nonzero_element_count] = block_vals
            current_sparse_offset += block_nonzero_element_count

    return current_sparse_offset, current_dense_offset


# def save_values_to_group(
#         blocks: List[Tuple[List[np.int64], List[np.int64], List]],
#         leftmost_block_index: np.int64,
#         block_datasets: Tuple[Dataset, Dataset, Dataset, Dataset, Dataset, Dataset],
#         current_sparse_offset: np.int64,
#         current_dense_offset: np.int64,
#         stripes: List[StripeDescriptor],
#         dense_submatrix_size: np.int64,
# ) -> Tuple[np.int64, np.int64]:
#     (block_rows, block_cols, block_vals, block_offset,
#      block_length, dense_blocks) = block_datasets
#     block_count_in_row: np.int64 = len(blocks)
#     row_stripe: StripeDescriptor = stripes[leftmost_block_index]
#     block_row_count: np.int64 = row_stripe.stripe_length_bins
#     for block_index in range(leftmost_block_index, block_count_in_row):
#         col_stripe: StripeDescriptor = stripes[block_index]
#         block = blocks[block_index]
#         block_index_in_datasets: np.int64 = leftmost_block_index * \
#             block_count_in_row + block_index
#         # Do not store empty blocks:
#         block_nonzero_element_count = len(block[0])
#         if block_nonzero_element_count == 0:
#             # print(f'Not dumping empty block {block_index}')
#             block_length[block_index_in_datasets] = 0
#             continue

#         block_col_count: np.int64 = col_stripe.stripe_length_bins

#         block_length[block_index_in_datasets] = block_nonzero_element_count

#         if (
#                 (
#                     block_row_count == dense_submatrix_size
#                 )
#                 and
#                 (
#                     block_col_count == dense_submatrix_size
#                 )
#                 and
#                 (
#                     block_nonzero_element_count >= (
#                         (block_row_count * block_col_count) // 2)
#                 )
#         ):
#             # Save as a dense matrix
#             block_offset[block_index_in_datasets] = -current_dense_offset - 1
#             dense_blocks.resize(1 + current_dense_offset, axis=0)
#             mx_coo = coo_matrix(
#                 (
#                     block[2],
#                     (
#                         block[0],
#                         block[1]
#                     )
#                 ),
#                 shape=(block_row_count, block_col_count)
#             )
#             dense_blocks[current_dense_offset, 0, :, :] = mx_coo.toarray()
#             current_dense_offset += 1
#         else:
#             # Save as sparse matrix
#             block_offset[block_index_in_datasets] = current_sparse_offset
#             block_rows[current_sparse_offset:current_sparse_offset +
#                        block_nonzero_element_count] = block[0]
#             block_cols[current_sparse_offset:current_sparse_offset +
#                        block_nonzero_element_count] = block[1]
#             block_vals[current_sparse_offset:current_sparse_offset +
#                        block_nonzero_element_count] = block[2]
#             current_sparse_offset += block_nonzero_element_count
#     return current_sparse_offset, current_dense_offset


# def clear_blocks(blocks: List[Tuple[List, List, List]], current_row_stripe_id: np.int64):
#     for block_index in range(current_row_stripe_id, len(blocks)):
#         blocks[block_index][0].clear()
#         blocks[block_index][1].clear()
#         blocks[block_index][2].clear()


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
    # contig_start_bins: h5py.Dataset = src_file[f'/resolutions/{resolution}/indexes/chrom_offset'].astype(
    #     np.int64)
    # contig_length_bins_ds: np.ndarray = contig_start_bins[1:] - \
    #     contig_start_bins[:-1]

    # bins_count: np.int64 = len(bins_group['chrom'])
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
    # print(
    #     f"src file keys: {list(src_file.keys())}", flush=True)
    # print(f"pathtonameandlength: {path_to_name_and_length}", flush=True)
    # print(
    #     f"pathtonameandlength keys: {list(src_file[f'{path_to_name_and_length}'].keys())}", flush=True)
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

    if 'length' in src_file[f'/resolutions/{anyresolution}/chroms'].keys():
        contig_info_group.copy(
            src_file[f'/resolutions/{anyresolution}/chroms/length'], 'contig_length_bp')
        contig_id_to_contig_length_bp = src_file[f'/resolutions/{anyresolution}/chroms/length'][:]
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

    contigs: List[ContigDescriptor] = [
        ContigDescriptor.make_contig_descriptor(
            contig_id=contig_id,
            contig_name=contig_names[contig_id],
            direction=ContigDirection.FORWARD,
            contig_length_bp=contig_id_to_contig_length_bp[contig_id],
            contig_length_at_resolution={
                resolution: resolution_to_contig_length_bins[resolution][contig_id] for resolution in resolutions},
            contig_presence_in_resolution={resolution: ContigHideType.AUTO_SHOWN if resolution_to_contig_length_bins[
                resolution][contig_id] > 1 else ContigHideType.AUTO_HIDDEN for resolution in resolutions},
            atus={resolution: [
                ATUDescriptor.make_atu_descriptor(
                    stripe_descriptor=stripes[resolution][contig_part_id + (contig_start_bins_at_resolution[resolution][contig_id] // resolution)],
                    start_index_in_stripe_incl=0,
                    end_index_in_stripe_excl=min(submatrix_size, resolution_to_contig_length_bins[resolution]
                     [contig_id] - contig_part_id*submatrix_size),
                    direction=ATUDirection.FORWARD
                ) for contig_part_id in range(
                    (resolution_to_contig_length_bins[resolution]
                     [contig_id] // resolution)
                    +
                    min(1,
                        resolution_to_contig_length_bins[resolution][contig_id] % resolution)
                )
            ] for resolution in resolutions},
            scaffold_id=None
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
    max_fetch_size: np.int64 = 8 * 1024 * 1024
    if additional_dataset_creation_args is None:
        additional_dataset_creation_args = {}

    with h5py.File(name=src_file_path, mode='r') as src_file:
        if resolutions is None:
            resolutions = [np.int64(sdn) for sdn in filter(
                lambda s: s.isnumeric(), src_file['resolutions'].keys())]
        with h5py.File(name=dst_file_path, mode='w') as dst_file:
            resolution_to_stripes: Dict[np.int64, List[StripeDescriptor]] = dict()
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
                # stripe_data: List[
                #     Tuple[
                #         List[np.int64],
                #         List[np.int64],
                #         List]
                # ] = [([], [], []) for _ in range(0, stripes_count)]
                # previous_pixel_row = 0
                # Note: since stripe !== contig, these are in fact different:
                # stripe_start_indices: np.ndarray = np.cumsum([0] + [s.stripe_length_bins for s in stripes],
                #                                              dtype=np.int64)
                # stripe_start_indices: h5py.Dataset = src_file[f'resolutions/{resolution}/indexes/chrom_offset']
                all_rows_start_indices: h5py.Dataset = src_file[
                    f'resolutions/{resolution}/indexes/bin1_offset']

                fetch_size: np.int64 = len(src_pixel_row)
                fetchblock_count: np.int64 = fetch_size // max_fetch_size + \
                    min(1, fetch_size % max_fetch_size)

                print(
                    f"bins_count: {src_bins_count}, "
                    f"stripes_count: {stripes_count}, "
                    f"fetch_size: {fetch_size}, "
                    f"fetchblock_count: {fetchblock_count}"
                )

                current_row_stripe_id: np.int64 = 0
                # current_col_stripe_id: np.int64 = 0
                current_row_stripe_start_row: np.int64 = 0
                # current_col_stripe_start_col: np.int64 = 0
                # # Note: these values are exclusive:
                current_row_stripe_max_row: np.int64 = stripes[0].stripe_length_bins
                # current_col_stripe_max_col: np.int64 = stripes[0].stripe_length
                # # In current stripe all rows are:
                # current_row_stripe_start_row <= row < current_row_stripe_max_row
                pixeltable_start_position: np.int64 = 0

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
                        src_pixel_row[all_rows_start_indices[vstripe_l*submatrix_size]:all_rows_start_indices[min((vstripe_l+1)*submatrix_size, len(all_rows_start_indices)-1)]],
                        src_pixel_col[all_rows_start_indices[vstripe_l*submatrix_size]:all_rows_start_indices[min((vstripe_l+1)*submatrix_size, len(all_rows_start_indices)-1)]],
                        src_pixel_val[all_rows_start_indices[vstripe_l*submatrix_size]:all_rows_start_indices[min((vstripe_l+1)*submatrix_size, len(all_rows_start_indices)-1)]],
                    )
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

                ################################
                # for pixel_row_fetchblock, pixel_col_fetchblock, pixel_cnt_fetchblock in (
                #         (
                #             src_pixel_row[i * max_fetch_size: min(
                #                 (1 + i) * max_fetch_size, fetch_size)],
                #             src_pixel_col[i * max_fetch_size: min(
                #                 (1 + i) * max_fetch_size, fetch_size)],
                #             src_pixel_val[i * max_fetch_size: min(
                #                 (1 + i) * max_fetch_size, fetch_size)]
                #         ) for i in range(0, fetchblock_count)
                # ):
                #     minimum_row_in_fetchblock: np.int64 = pixel_row_fetchblock[0]
                #     maximum_row_in_fetchblock: np.int64 = pixel_row_fetchblock[-1]
                #     assert (
                #         minimum_row_in_fetchblock <= maximum_row_in_fetchblock
                #     ), "Row numbers are not monotonely non-decreasing in source data??"
                #     # print(
                #     #     f"Searching start positions for rows between {minimum_row_in_fetchblock} and {maximum_row_in_fetchblock}")
                #     # TODO: Maybe I search for the same thing as bin1_id_offset index from cooler? (Yes, it is, use it!)
                #     # row_start_indices: np.ndarray = np.searchsorted(
                #     #     pixel_row_fetchblock,
                #     #     range(minimum_row_in_fetchblock, 2 + maximum_row_in_fetchblock),
                #     #     side='left'
                #     # )
                #     pixel_row_stripes = pixel_row_fetchblock // submatrix_size
                #     pixel_col_stripes = pixel_col_fetchblock // submatrix_size
                #     pixel_intra_stripe_row = pixel_row_fetchblock % submatrix_size
                #     pixel_intra_stripe_col = pixel_col_fetchblock % submatrix_size

                #     row_start_indices: np.ndarray = all_rows_start_indices[
                #         minimum_row_in_fetchblock:2 + maximum_row_in_fetchblock
                #     ] - pixeltable_start_position
                #     if row_start_indices[0] < 0:
                #         first_zero_index: np.int64 = np.searchsorted(
                #             row_start_indices, 0, side='left')
                #         row_start_indices[:first_zero_index] = 0
                #     pixeltable_start_position += len(pixel_row_fetchblock)

                #     for i in range(0, len(row_start_indices) - 1):
                #         row_start_index: np.int64 = row_start_indices[i]
                #         row_number: np.int64 = pixel_row_fetchblock[row_start_index]
                #         next_row_start_index: np.int64 = row_start_indices[1 + i]
                #         # print(f"Parsing row {row_number}")
                #         while row_number >= current_row_stripe_max_row:
                #             current_sparse_offset, current_dense_offset = save_values_to_group(
                #                 stripe_data,
                #                 current_row_stripe_id,
                #                 block_datasets,
                #                 current_sparse_offset,
                #                 current_dense_offset,
                #                 stripes,
                #                 submatrix_size
                #             )
                #             clear_blocks(stripe_data, 0)
                #             dst_file.flush()
                #             current_row_stripe_id += 1
                #             current_row_stripe_start_row = current_row_stripe_max_row
                #             current_row_stripe_max_row += stripes[current_row_stripe_id].stripe_length_bins
                #             print(
                #                 f"{print(datetime.datetime.now())}\nStripe {current_row_stripe_id} out of {stripes_count}")
                #         if row_start_index == next_row_start_index:
                #             continue
                #         assert row_start_index >= 0
                #         assert next_row_start_index >= 0
                #         single_row_row: np.ndarray = pixel_row_fetchblock[
                #             row_start_index:next_row_start_index] - current_row_stripe_start_row
                #         single_row_col: np.ndarray = pixel_col_fetchblock[
                #             row_start_index:next_row_start_index]
                #         single_row_cnt: np.ndarray = pixel_cnt_fetchblock[
                #             row_start_index:next_row_start_index]
                #         assert max(single_row_row) == min(single_row_row)
                #         if not is_sorted(single_row_col):
                #             # print("AAAA")
                #             pass
                #         assert is_sorted(single_row_col)
                #         block_start_indices: np.ndarray = np.searchsorted(
                #             single_row_col,
                #             stripe_start_indices[current_row_stripe_id:],
                #             side='left'
                #         )

                #         # print(f"Dumping column stripes: {block_start_indices}")

                #         for j in range(0, len(block_start_indices) - 1):
                #             block_in_row_start_index: np.int64 = block_start_indices[j]
                #             next_block_in_row_start_index: np.int64 = block_start_indices[1 + j]
                #             # If there are some points from current block in current row:
                #             if next_block_in_row_start_index > block_in_row_start_index:
                #                 row_ids = single_row_row[block_in_row_start_index:next_block_in_row_start_index]
                #                 assert 0 <= max(row_ids) < submatrix_size
                #                 assert 0 <= min(row_ids) < submatrix_size
                #                 stripe_data[current_row_stripe_id +
                #                             j][0].extend(row_ids)
                #                 column_ids_original = single_row_col[
                #                     block_in_row_start_index:next_block_in_row_start_index
                #                 ]
                #                 column_ids_in_block = column_ids_original - stripe_start_indices[
                #                     current_row_stripe_id + j]
                #                 assert 0 <= max(
                #                     column_ids_in_block) < submatrix_size
                #                 assert 0 <= min(
                #                     column_ids_in_block) < submatrix_size
                #                 stripe_data[current_row_stripe_id +
                #                             j][1].extend(column_ids_in_block)
                #                 stripe_data[current_row_stripe_id + j][2].extend(
                #                     single_row_cnt[block_in_row_start_index:next_block_in_row_start_index])

                # current_sparse_offset, current_dense_offset = save_values_to_group(
                #     stripe_data,
                #     current_row_stripe_id,
                #     block_datasets,
                #     current_sparse_offset,
                #     current_dense_offset,
                #     stripes,
                #     submatrix_size
                # )
                # clear_blocks(stripe_data, current_row_stripe_id)
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
        additional_dataset_creation_args['compression'] = args.compression.lower(
        )

    print(f"args object: {args}")

    cool_flatten_convert(
        args.input,
        args.output if args.output is not None else f"{args.input}.hict.hdf5",
        lambda r: f'/resolutions/{str(r)}/chroms',
        resolutions=[
            np.int64(r)
            for r in args.resolutions
        ] if args.resolutions is not None else None,
        additional_dataset_creation_args=additional_dataset_creation_args
    )


cool_flatten_convert(
    "..\\HiCT_Server\\data\\mat18_100k.cool",
    "..\\HiCT_Server\\data\\mat18_100k.cool.hict.hdf5",
    lambda r: f'/resolutions/{str(r)}/chroms',
    resolutions=None,
    additional_dataset_creation_args=None
)
