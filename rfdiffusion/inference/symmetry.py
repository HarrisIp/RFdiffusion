"""Helper class for handle symmetric assemblies."""
from pyrsistent import v
from scipy.spatial.transform import Rotation
import functools as fn
import torch
import string
import logging
import numpy as np
import pathlib
from rfdiffusion.contigs import ContigMap
from omegaconf import ListConfig

format_rots = lambda r: torch.tensor(r).float()

T3_ROTATIONS = [
    torch.Tensor([
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]]).float(),
    torch.Tensor([
        [-1., -0.,  0.],
        [-0.,  1.,  0.],
        [-0.,  0., -1.]]).float(),
    torch.Tensor([
        [-1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 0.,  0.,  1.]]).float(),
    torch.Tensor([
        [ 1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 0.,  0., -1.]]).float(),
]

saved_symmetries = ['tetrahedral', 'octahedral', 'icosahedral']

class SymGen:

    def __init__(self, global_sym, recenter, radius, model_only_neighbors=False, contig=None):
        self._log = logging.getLogger(__name__)
        self._recenter = recenter
        self._radius = radius
        self._contig = contig

        if global_sym.lower().startswith('c'):
            # Cyclic symmetry
            if not global_sym[1:].isdigit():
                raise ValueError(f'Invalid cyclic symmetry {global_sym}')
            self._log.info(
                f'Initializing cyclic symmetry order {global_sym[1:]}.')
            self._init_cyclic(int(global_sym[1:]))
            self.apply_symmetry = self._apply_cyclic

        elif global_sym.lower().startswith('d'):
            # Dihedral symmetry
            if not global_sym[1:].isdigit():
                raise ValueError(f'Invalid dihedral symmetry {global_sym}')
            self._log.info(
                f'Initializing dihedral symmetry order {global_sym[1:]}.')
            self._init_dihedral(int(global_sym[1:]))
            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic

        elif global_sym.lower() == 't3':
            # Tetrahedral (T3) symmetry
            self._log.info('Initializing T3 symmetry order.')
            self.sym_rots = T3_ROTATIONS
            self.order = 4
            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic

        elif global_sym == 'octahedral':
            # Octahedral symmetry
            self._log.info(
                'Initializing octahedral symmetry.')
            self._init_octahedral()
            self.apply_symmetry = self._apply_octahedral

        elif global_sym.lower() in saved_symmetries:
            # Using a saved symmetry 
            self._log.info('Initializing %s symmetry order.'%global_sym)
            self._init_from_symrots_file(global_sym)

            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic
        else:
            raise ValueError(f'Unrecognized symmetry {global_sym}')

        self.res_idx_procesing = fn.partial(
            self._lin_chainbreaks, num_breaks=self.order)

    #####################
    ## Cyclic symmetry ##
    #####################

    
    def _init_cyclic(self, order):
        self.order = order
        
        sym_rots = []
        for i in range(order):
            deg = i * 360.0 / order
            r = Rotation.from_euler('z', deg, degrees=True)
            sym_rots.append(format_rots(r.as_matrix()))
        self.sym_rots = sym_rots


        
    ###TEST
    def _get_nchain():
        # Parse the contig to determine subunit lengths
        if isinstance(self._contig, ListConfig):
            sampled_mask = list(self._contig)[0].split()
        elif isinstance(self._contig, list):
            sampled_mask = self._contig
        else:
            raise ValueError(f"Unexpected type for self._contig: {type(self._contig)}")
        subunit_lengths = [int(part.split("/")[0]) for part in sampled_mask]
        nchains = len(sampled_mask)
        chain_per_subunit = nchains / self.order
        return nchains
    
    def _get_chain_per_subunit(self):
        # Parse the contig to determine subunit lengths
        if isinstance(self._contig, ListConfig):
            sampled_mask = list(self._contig)[0].split()
        elif isinstance(self._contig, list):
            sampled_mask = self._contig
        else:
            raise ValueError(f"Unexpected type for self._contig: {type(self._contig)}")
        nchains = len(sampled_mask)
        chain_per_subunit = nchains / self.order
        return chain_per_subunit
    ###
    ###ORIGINAL
    def _apply_cyclic(self, coords_in, seq_in):
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisble by {self.order}')
        subunit_len = seq_out.shape[0] // self.order
        for i in range(self.order):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            coords_out[start_i:end_i] = torch.einsum(
                'bnj,kj->bnk', coords_out[:subunit_len], self.sym_rots[i])
            seq_out[start_i:end_i]  = seq_out[:subunit_len]
        return coords_out, seq_out

    

    
    ###test###
    '''def _apply_cyclic(self, coords_in, seq_in):
        """
        Applies cyclic symmetry to the input coordinates and sequence.

        Args:
            coords_in (torch.Tensor): Input coordinates, shape [L, N, 3].
            seq_in (torch.Tensor): Input sequence, shape [L, num_classes].

        Returns:
            tuple: Transformed coordinates and sequence after applying cyclic symmetry.
        """
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        # Parse the contig to determine subunit lengths
        if isinstance(self._contig, ListConfig):
            sampled_mask = list(self._contig)[0].split()
        elif isinstance(self._contig, list):
            sampled_mask = self._contig
        else:
            raise ValueError(f"Unexpected type for self._contig: {type(self._contig)}")
        # Extract subunit lengths from the parsed mask
        subunit_lengths = [int(part.split("/")[0]) for part in sampled_mask]
        nchains = len(sampled_mask)
        chain_per_subunit = nchains / self.order
        # Ensure total sequence length matches the sum of subunit lengths
        total_length = sum(subunit_lengths)
        if seq_out.shape[0] != total_length:
            raise ValueError(
                f"Mismatch between sequence length ({seq_out.shape[0]}) and "
                f"total subunit length ({total_length})."
            )
        current_index = 0
        nchains = len(subunit_lengths)  # Number of chains provided
        chain_per_subunit = nchains // self.order  # Chains per symmetry unit
        if nchains % self.order != 0:
            raise ValueError(
                f"The number of chains ({nchains}) must be divisible by the symmetry order ({self.order})."
            )
        for subunit_idx in range(self.order):  # Iterate over symmetry order

            for chain_idx in range(chain_per_subunit):  # Iterate over chains within one symmetry unit
                # Get the current subunit length
                subunit_length = subunit_lengths[subunit_idx * chain_per_subunit + chain_idx]
                start_idx = current_index
                end_idx = current_index + subunit_length

                # Ensure rotation order matches symmetry requirements
                if subunit_idx >= self.order:
                    raise ValueError(
                        f"Subunit index ({subunit_idx}) exceeds symmetry order ({self.order}). Subunit lengths: {subunit_lengths}"
                    )
                # Apply rotation for the current subunit
                coords_out[start_idx:end_idx] = torch.einsum(
                    'bnj,kj->bnk', coords_out[:subunit_length], self.sym_rots[subunit_idx]
                )
                seq_out[start_idx:end_idx] = seq_out[:subunit_length].clone()
                # Update the current index for the next chain
                current_index = end_idx

        # Ensure all residues were processed
        if current_index != total_length:
            raise ValueError(
                f"Mismatch during symmetry application. Processed residues ({current_index}) "
                f"do not match total residues ({total_length})."
            )

        return coords_out, seq_out'''
    
        ###break chain in same symmetry unit###
    def _lin_chainbreaks(self, num_breaks, res_idx, offset=None):
        # Handle ListConfig and ensure it's treated as a list
        if  isinstance(self._contig, str ):
            sampled_mask = self._contig.split(' ')
        elif isinstance(self._contig, ListConfig):
            sampled_mask = list(self._contig)
            sampled_mask = sampled_mask[0].split()
        elif isinstance(self._contig, list): 
            sampled_mask = self._contig
        else:
            raise ValueError(f"Unexpected type for self._contig: {type(self._contig)}")

        assert res_idx.ndim == 2, "res_idx must be a 2D tensor"

        res_idx = torch.clone(res_idx)
        chain_delimiters = []

        # Generate chain labels
        chain_labels = list(string.ascii_uppercase) + [
            f"{i}{j}" for i in string.ascii_uppercase for j in string.ascii_uppercase
        ]

        # Set default offset if not provided
        if offset is None:
            offset = res_idx.shape[-1]

        current_chain_label_idx = 0
        current_residue_index = 0  # Tracks the residue position in res_idx

        for mask_part in sampled_mask:
            # Parse mask_part to get mask_count
            try:
                mask_count = int(mask_part.split("/")[0])  # Get the number of residues
            except ValueError:
                raise ValueError(f"Invalid format in sampled_mask: {mask_part}")

            # Ensure enough chain labels
            if current_chain_label_idx >= len(chain_labels):
                raise ValueError("Not enough chain labels for the given configuration.")

            # Extend chain_delimiters for the current mask part
            chain_delimiters.extend([chain_labels[current_chain_label_idx]] * mask_count)

            # Update res_idx with the correct offsets for the current segment
            end_residue_index = current_residue_index + mask_count
            res_idx[:, current_residue_index:end_residue_index] += offset * (current_chain_label_idx + 1)

            # Update indices and chain label index
            current_residue_index = end_residue_index
            current_chain_label_idx += 1

        # Verify that the chain_delimiters list matches the total residues in res_idx
        if len(chain_delimiters) != res_idx.shape[-1]:
            raise ValueError(
                f"Mismatch between chain_delimiters ({len(chain_delimiters)}) and residues in res_idx ({res_idx.shape[-1]}), self._contig: {self._contig[0]}, sampled mask{sampled_mask}{type(sampled_mask[0])}"
            )

        return res_idx, chain_delimiters

    
    #original#
    '''def _lin_chainbreaks(self, num_breaks, res_idx, offset=None):
        assert res_idx.ndim == 2
        res_idx = torch.clone(res_idx)
        subunit_len = res_idx.shape[-1] // num_breaks
        chain_delimiters = []
        if offset is None:
            offset = res_idx.shape[-1]
        for i in range(num_breaks):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            chain_labels = list(string.ascii_uppercase) + [str(i+j) for i in
                    string.ascii_uppercase for j in string.ascii_uppercase]
            chain_delimiters.extend(
                [chain_labels[i] for _ in range(subunit_len)]
            )
            res_idx[:, start_i:end_i] = res_idx[:, start_i:end_i] + offset * (i+1)
        return res_idx, chain_delimiters'''
    ##

    #######################
    ## Dihedral symmetry ##
    #######################
    def _init_dihedral(self, order):
        sym_rots = []
        flip = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        for i in range(order):
            deg = i * 360.0 / order
            rot = Rotation.from_euler('z', deg, degrees=True).as_matrix()
            sym_rots.append(format_rots(rot))
            rot2 = flip @ rot
            sym_rots.append(format_rots(rot2))
        self.sym_rots = sym_rots
        self.order = order * 2

    #########################
    ## Octahedral symmetry ##
    #########################
    def _init_octahedral(self):
        sym_rots = np.load(f"{pathlib.Path(__file__).parent.resolve()}/sym_rots.npz")
        self.sym_rots = [
            torch.tensor(v_i, dtype=torch.float32)
            for v_i in sym_rots['octahedral']
        ]
        self.order = len(self.sym_rots)

    def _apply_octahedral(self, coords_in, seq_in):
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisble by {self.order}')
        subunit_len = seq_out.shape[0] // self.order
        base_axis = torch.tensor([self._radius, 0., 0.])[None]
        for i in range(self.order):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            subunit_chain = torch.einsum(
                'bnj,kj->bnk', coords_in[:subunit_len], self.sym_rots[i])

            if self._recenter:
                center = torch.mean(subunit_chain[:, 1, :], axis=0)
                subunit_chain -= center[None, None, :]
                rotated_axis = torch.einsum(
                    'nj,kj->nk', base_axis, self.sym_rots[i]) 
                subunit_chain += rotated_axis[:, None, :]

            coords_out[start_i:end_i] = subunit_chain
            seq_out[start_i:end_i]  = seq_out[:subunit_len]
        return coords_out, seq_out

    #######################
    ## symmetry from file #
    #######################
    def _init_from_symrots_file(self, name):
        """ _init_from_symrots_file initializes using 
        ./inference/sym_rots.npz

        Args:
            name: name of symmetry (of tetrahedral, octahedral, icosahedral)

        sets self.sym_rots to be a list of torch.tensor of shape [3, 3]
        """
        assert name in saved_symmetries, name + " not in " + str(saved_symmetries)

        # Load in list of rotation matrices for `name`
        fn = f"{pathlib.Path(__file__).parent.resolve()}/sym_rots.npz"
        obj = np.load(fn)
        symms = None
        for k, v in obj.items():
            if str(k) == name: symms = v
        assert symms is not None, "%s not found in %s"%(name, fn)

        
        self.sym_rots =  [torch.tensor(v_i, dtype=torch.float32) for v_i in symms]
        self.order = len(self.sym_rots)

        # Return if identity is the first rotation  
        if not np.isclose(((self.sym_rots[0]-np.eye(3))**2).sum(), 0):

            # Move identity to be the first rotation
            for i, rot in enumerate(self.sym_rots):
                if np.isclose(((rot-np.eye(3))**2).sum(), 0):
                    self.sym_rots = [self.sym_rots.pop(i)]  + self.sym_rots

            assert len(self.sym_rots) == self.order
            assert np.isclose(((self.sym_rots[0]-np.eye(3))**2).sum(), 0)

    def close_neighbors(self):
        """close_neighbors finds the rotations within self.sym_rots that
        correspond to close neighbors.

        Returns:
            list of rotation matrices corresponding to the identity and close neighbors
        """
        # set of small rotation angle rotations
        rel_rot = lambda M: np.linalg.norm(Rotation.from_matrix(M).as_rotvec())
        rel_rots = [(i+1, rel_rot(M)) for i, M in enumerate(self.sym_rots[1:])]
        min_rot = min(rel_rot_val[1] for rel_rot_val in rel_rots)
        close_rots = [np.eye(3)] + [
                self.sym_rots[i] for i, rel_rot_val in rel_rots if
                np.isclose(rel_rot_val, min_rot)
                ]
        return close_rots
