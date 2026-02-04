#TODO: TEST AT RC IF LINJE ≈446 fungerer

from math import perm
from pathlib import Path
from typing import Union, List, Optional, Tuple, Any
from ase.io import read as ase_read
from ase import Atoms
from dataclasses import dataclass, field
import time
#from __future__ import annotations

import os
import numpy as np
import pandas as pd

#from builder import Paths, WhamSettings, PredictivePowerSettings, CVMatrixSettings
#from wham_weights import TrajWeights
#from frame_selector import extract_frames
from label_manager.type_manager import TypeManager
from label_manager.index_invariant import IndexInvariantMapper  #index_invariant is used to keep labels constant for each frame by mapping to coordinates. Might not be needed, but had som inital problems that was cause by PBC 
from storage.h5_writer import H5Writer
from storage.mat_writer import write_mat_data
from CVmanager.reaction_center import select_reaction_center, ReactionCenter
from CVmanager.load_cp2k_aux import load_cp2k_aux_data_for_frame

from CVmanager import (
    WireCompressionCV,
    WireLinearityCVCos,
    WireSigmaOOCV,
    WireFirstOOCV,
    ZundelCoordinateCV,
    MultiProtonCoordsCV,
    AcceptorsCountCV,
    DonorsCountCV,
    DonorAcceptorImbalanceCV,
    DualPresolvationIndicatorCV,
    TetrahedralityQ_CV,
    EParallelCV,
    HBStrengthCV,
    HBAngleMeanCV,
    HBTotalCountCV,
    HBDonAcceptRatioCV,
    SecondShellAcceptorsCV,
    DeltaN1N2_CV,
    SteinhardtQ4_CV,
    LocalDensityCV,
    LocalSofR_CV,
    ROODotCV,
    DeltaDotCV,
    HBBondLifetimeCV,
    HBSwitchFrequencyCV,
    DipoleAlongAxisCV,
    ForcesAlongAxisCV,
    HOMOFieldProxyCV,
    LpOrientationAtAcceptorCV,
    BondCenterPositionCV,
    ElectronicZundelSymmetryCV,
    WannierRCFeaturesCV,
    build_neighborhood,
    ReactionNeighborhood,
    DANeighborRatiosCV,
    EParallelMultiAxisCV,
    ElectronicPresolvationCV,
    ElectronicDualPresolvationCV
)








@dataclass
class CVInputs:
    # Core
    coords: np.ndarray
    pairs_all: np.ndarray
    pairs_labels: np.ndarray
    # Water topology (invariant)
    water_triplets: np.ndarray
    water_triplet_labels: np.ndarray
    ho_pairs: np.ndarray | None = None
    hh_pairs: np.ndarray | None = None
    # Run metadata / diagnostics
    flags: dict[str, Any] = field(default_factory=dict)
    key: object | None = None
    box: float | None = None
    # Mapping helpers (for file-based CVs like Mulliken/Wannier)
    perm: Optional[np.ndarray] = None            # REF -> SORTED
    sort_idx: Optional[np.ndarray] = None
    traj_path: Optional[Path] = None
    # Optional QA labels
    water_labels_physical: Optional[list[str]] = None
    water_labels_invariant: Optional[list[str]] = None
    reaction: ReactionCenter | None = None #precomputed reaction center for this frame
    neighborhood: ReactionNeighborhood | None = None
    ion_index: Optional[int] = None      # index in *invariant* coords (same space as coords)
    ion_label: Optional[str] = None      # "Na" / "Cl" (or general symbol)


class SimulationRunner: #TODO: define K neighborhood in toml/builder. Optional build in anchor point for Cl and Na systems
    def __init__(
            self, 
            paths, #: Paths,
            config, #: dict,
            interfaces: List[float], 
            wham, #: WhamSettings, 
            grid, #: GridDefinition,
            pred_power, #: PredictivePowerSettings,
            CVs, #: CVMatrixSettings,
            all_types: set, #: set[str],
            weights, #: TrajWeights
        ):
        #same indices → numbers from coords_invariant and names from labels.
        self.paths = paths
        self.config = config
        self.interfaces = interfaces
        self.wham = wham
        self.grid = grid
        self.pred_power = pred_power
        self.CVs = CVs
        self.all_types = all_types
        self.weights = weights
        self.water_tracker = WaterLabelTracker() #Water-labels are not static, and needs to be constructed inside the retis_steps loop
        self.label_maps = TypeManager(self.CVs, self.all_types, self.paths.load_dir)  #manages types and labels - TODO: The code is a bit redundant, but findes reference atoms for later analysis
        self.sort_idx = self.label_maps.sort_idx

        self.index_mapper = IndexInvariantMapper(    #Currently this reads the first frame from first trajectory, could enhance by reading the phasepoint closes to target OP, but seems to be working fine!
            cutoff=1.25,          # OH cutoff in Å   #TODO: This is a bit arbitary since I we are not considering equilibrium
            max_iter=7,           # a few more ICP rounds
            err_thresh=4.0,       # Å^2 (RMSD ≈ 2 Å) – tighten later if you want
            box=self.CVs.cell_size,   # e.g., 12.4138
            prefer_hungarian=True     # uses SciPy if available; falls back to greedy
        )

        self.index_mapper.set_reference(self.label_maps.df_sorted_reference)

        # 1) All-atom PAIRS (numeric + readable) — invariant
        self.index_invariant_pairs = self.label_maps.pairs["all_atoms"]          # (M,2) int
        self.index_invariant_pair_labels = self.label_maps.labels["all_atoms"]   # (M,) str

        # 2) Water MOLECULES as TRIPLETS (H1, O, H2) — invariant
        idx_O_ref  = self.index_mapper.idx_O_ref              # absolute row idx (ref) for all O
        idx_H_ref  = self.index_mapper.idx_H_ref              # absolute row idx (ref) for all H
        pairs_Href = self.index_mapper.ref_H_pairs_per_O      # (nO,2) H indices in H_ref-list order

        if idx_H_ref is None:
            raise ValueError("idx_H_ref must be initialized before use")
        if idx_O_ref is None:
            raise ValueError("idx_O_ref must be initialized before use")
        if pairs_Href is None:
            raise ValueError("pairs_Href must be initialized before use")
        
        # numeric triplets for angles etc.
        self.index_invariant_water_triplets = np.array(
            [[idx_H_ref[h1], idx_O_ref[o], idx_H_ref[h2]] for o, (h1, h2) in enumerate(pairs_Href)],
            dtype=int
        )

        # readable molecule IDs (stable across frames)
        names = self.label_maps.names_ref                     # canonical Hxx/Oxx names in ref order
        if names is None:
            raise ValueError("names must be initialized before use")
        self.index_invariant_water_labels = np.array(
            [f"{names[idx_H_ref[h1]]}-{names[idx_O_ref[o]]}-{names[idx_H_ref[h2]]}"
            for o, (h1, h2) in enumerate(pairs_Href)]
        )

        # HO bond pairs (flattened across waters)
        self.index_invariant_ho_pairs = np.array(
            [(idx_O_ref[o], idx_H_ref[h1]) for o,(h1, h2) in enumerate(pairs_Href)] +
            [(idx_O_ref[o], idx_H_ref[h2]) for o,(h1, h2) in enumerate(pairs_Href)],
            dtype=int
        )
        # HH pairs inside each water (for H–H distance)
        self.index_invariant_hh_pairs = np.array(
            [(idx_H_ref[h1], idx_H_ref[h2]) for o,(h1, h2) in enumerate(pairs_Href)],
            dtype=int
        )
        if self.label_maps.names_ref is None:
            raise ValueError("self.label_maps.names_ref not initialized")

        print(f"invariant water triplets:\n {self.index_invariant_water_triplets}")
        print(f"invariant HO-pairs:\n {self.index_invariant_ho_pairs}")
        print(f"invariant HH-pairs:\n {self.index_invariant_hh_pairs}")
        print(f"invariant pairs:\n {self.index_invariant_pairs}")

        """
        self.cv_modules = [
            NeighborOODistCV(K=32),
            NeighborOHDistCV(
                K=32,
                skip_self=False,                    # start at O2 as you requested
                h_distance_to="O_d",               # or "H_star"
                names_ref=self.label_maps.names_ref,  # to get "Oxx-Hyy" physical labels
            ),
            NeighborHOHAngleCV(
                K=32,
                skip_self=False,
                names_ref=self.label_maps.names_ref,  # for "Hxx-Oyy-Hzz" labels
            ),
            NeighborOHOrientationCV_ed(
                K=32,
                skip_self=False,
                names_ref=self.label_maps.names_ref,  # for "Oxx-Hyy" labels
            ),
            NeighborWannierRankedCV(
                K=32,
                skip_self=False,                      # include O1 so you can compare with /wannier_geom
                names_ref=self.label_maps.names_ref,  # for "Oxx-Oyy" labels
                use_global_for_self=True,             # O1 uses the global RC for exact match
                oxygen_indices=self.index_mapper.idx_O_ref,
                ho_pairs=self.index_invariant_ho_pairs,
                require_exact_four=False,                # None to disable
            ),
            NeighborForcesRankedViaBaseCV(
                K=32,
                skip_self=False,                       # include O1 to compare with /forces_projection
                names_ref=self.label_maps.names_ref,   # for "Oxx-Oyy" labels
                use_global_for_self=True,              # O1 uses the global RC for exact match
                sort_idx=self.sort_idx,                # ORIG -> SORTED (required by base forces CV)
            ),
            NeighborMullikenRankedViaBaseCV(
                K=32,
                skip_self=False,                       # include O1 to compare with /mulliken_charge
                names_ref=self.label_maps.names_ref,   # "Oxx-Oyy" labels
                use_global_for_self=True,              # O1 uses global RC for exact match
                oxygen_indices=self.index_mapper.idx_O_ref,
                hydrogen_indices=self.index_mapper.idx_H_ref,
                ho_pairs=self.index_invariant_ho_pairs,
                sort_idx=self.sort_idx,                # ORIG -> SORTED (required by base Mulliken CV)
            ),
            NeighborPTGeometryRankedCV(
                K=32,
                skip_self=False,                       # include O1 so you can compare to /pt_geometry
                names_ref=self.label_maps.names_ref,   # for "Oxx-Oyy" labels
                use_global_for_self=True,              # O1 uses global RC ⇒ exact match expected
                oxygen_indices=self.index_mapper.idx_O_ref,
                ho_pairs=self.index_invariant_ho_pairs,  # (2*nO,2) (O,H) in invariant order
            ),
            HBWireLengthCV(
                rcut=3.5,                # Å
                angle_cut_deg=30.0,      # donor-H-bond angle cutoff
                center="donor",          # or "acceptor" if you want the wire to include O_a
                path_node_lengths=(3,4,5),
                names_ref=self.label_maps.names_ref,
            ),
        ]
        """

        self.cv_modules = [
            # NB: bruk samme rcut/angle/center/path_node_lengths som i HBWireLengthCV for sammenlignbarhet
            WireCompressionCV(rcut=3.5, angle_cut_deg=30.0, center="donor", path_node_lengths=(3,4,5,6)),
            WireFirstOOCV(rcut=3.5, angle_cut_deg=30.0, center="donor", path_node_lengths=(3,4,5,6)),
            WireSigmaOOCV(rcut=3.5, angle_cut_deg=30.0, center="donor", path_node_lengths=(3,4,5,6)),
            WireLinearityCVCos(rcut=3.5, angle_cut_deg=30.0, center="donor", path_node_lengths=(3,4,5,6), reduce="mean"),
            DonorAcceptorImbalanceCV(R_OO_max=3.5, angle_min_deg=150.0, use_donor_O=True),
            DualPresolvationIndicatorCV(R1=3.5, R2=5.0, angle_min_deg=150.0, use_donor_O=True),
            TetrahedralityQ_CV(use_donor_O=True),
            HBStrengthCV(R_OO_max=3.5, R0=2.8, sigma=0.3, use_donor_O=True),
            ZundelCoordinateCV(),  # labels: ("delta","R_OdH","R_OaH","R_OdOa")
            MultiProtonCoordsCV(
                rcut=3.5, angle_cut_deg=30.0, center="donor",
                path_node_lengths=(3,4,5,6),            # velg N-noder du vil (gir 2 kolonner per N)
            ),

            HBAngleMeanCV(R_OO_max=3.5, angle_min_deg=150.0, use_donor_O=True),
            HBTotalCountCV(R_OO_max=3.5, angle_min_deg=150.0, use_donor_O=True),
            HBDonAcceptRatioCV(R_OO_max=3.5, angle_min_deg=150.0, use_donor_O=True),

            # Andre skall / presolvasjon
            SecondShellAcceptorsCV(R1=3.5, R2=5.0, use_donor_O=True),
            DeltaN1N2_CV(R1_max=3.5, R2_max=5.0, use_donor_O=True),

            # Orden/struktur
            SteinhardtQ4_CV(k_neigh=12, use_donor_O=True),
            LocalDensityCV(R=4.0, use_donor_O=True),
            LocalSofR_CV(R_max=5.0, dr=0.2, use_donor_O=True),

            # Dynamikk (krever at runner kaller compute flere ganger innen samme bucket)
            ROODotCV(dt_fs=1.0),
            DeltaDotCV(dt_fs=1.0),
            HBBondLifetimeCV(dt_fs=1.0, R_OO_max=3.5, angle_min_deg=150.0),
            HBSwitchFrequencyCV(dt_fs=1.0, R_OO_max=3.5, angle_min_deg=150.0),

            HOMOFieldProxyCV(p=2.0),
            ForcesAlongAxisCV(),
            DipoleAlongAxisCV(),

            AcceptorsCountCV(R_OO_max=3.5, angle_min_deg=150.0, use_donor_O=True),
            DonorsCountCV(R_OO_max=3.5, angle_min_deg=150.0, use_donor_O=True),

            # Eksplisitte elektroniske observabler (med HOMO-fallback)
            LpOrientationAtAcceptorCV(),
            BondCenterPositionCV(),
            ElectronicZundelSymmetryCV(),

            # Kompakt RC-basert Wannier-feature-set (erstatter gammel neighbor-rank-variant)
            WannierRCFeaturesCV(require_exact_four=False),

            DANeighborRatiosCV(), 

            EParallelCV(use_midpoint_if_no_H=True),
            EParallelMultiAxisCV(use_midpoint_if_no_H=True),

            ElectronicPresolvationCV(R1_elec=3.0, R2_elec=5.0, use_donor_O=False),  # rundt O_a
            ElectronicDualPresolvationCV(use_donor_O=False),
        ]


    def run(self) -> None:
        """
        Iterate grid/steps, build CVInputs, compute selected CVs, append to HDF5.
        """
        base = self.paths.load_dir.parent
        cv_root = base / "CVs" / str(self.wham.n_grid)

        # --- single-writer guard to avoid HDF5 lock contention ---
        rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
        world = int(os.environ.get("SLURM_NTASKS",  os.environ.get("PMI_SIZE", "1")))

        # Optional: a small sentinel file to signal completion
        sentinel = (cv_root / ".matrix_meta.ready")

        if rank == 0:
            # Only rank 0 creates/updates the shared meta H5 once
            # If multiple jobs rerun, it's fine to overwrite/append as your function intends.
            for attempt in range(10):
                try:
                    write_mat_data(
                        cv_root=cv_root,
                        grid_ids=self.grid.grid_idx,
                        pair_labels=self.index_invariant_pair_labels,
                        n_grid=len(self.grid.grid_idx),
                    )
                    # touch sentinel so others can proceed without polling the H5
                    try:
                        sentinel.write_text("ok")
                    except Exception:
                        pass
                    break
                except BlockingIOError:
                    # another process may still have it open; back off briefly
                    time.sleep(0.5)
            else:
                print("[WARN] rank0 could not write matrix meta after retries; continuing anyway.")
        else:
            # Non-zero ranks: wait briefly for rank0 to finish first-time creation.
            # This avoids opening the file concurrently during creation.
            for _ in range(40):  # up to ~20s
                if sentinel.exists():
                    break
                time.sleep(0.5)
        """
        write_mat_data(                                             #without sharding
            cv_root=cv_root,
            grid_ids=self.grid.grid_idx,                          # e.g. [100, 230, 300, ...]
            pair_labels=self.index_invariant_pair_labels,         # (M,) strings
            n_grid=len(self.grid.grid_idx),
        )
        """
        for step in self.weights.retis_steps:
            writer = None
            rc = None
            rc_op = None
            try:
                for (op, grid_id) in zip(self.grid.grid_pts, self.grid.grid_idx):
                    load_step = self.paths.load_dir / str(step)
                    frames, lambda_max = extract_frames(load_step, grid_idx=grid_id, grid_points=op)
                    if not frames:
                        continue

                    ase_idx, traj_path = self.load_phasepoint_frame(load_step, frames)

                    atom_obj  = ase_read(traj_path, index=ase_idx)
                    assert isinstance(atom_obj, Atoms)
                    positions = atom_obj.get_positions()
                    symbols   = atom_obj.get_chemical_symbols()

                    df_sorted = make_sorted_df_from_frame(positions, symbols, self.sort_idx)

                    # mapper + invariant coords
                    perm, err, flags = self.index_mapper.map_to_reference(df_sorted)
                    coords_sorted    = df_sorted[['x','y','z']].to_numpy()
                    coords_invariant = coords_sorted[perm]


                    mull, homo, dip, frc = load_cp2k_aux_data_for_frame(
                        traj_path=traj_path,
                        ase_idx=ase_idx,
                        n_atoms=coords_invariant.shape[0],
                    )

                    # 1) detect ion on the *sorted* frame (anything not H or O)
                    if 'symbol' in df_sorted.columns:
                        is_ion_row = ~df_sorted['symbol'].isin(['H', 'O'])
                        ion_rows = np.flatnonzero(is_ion_row.to_numpy())
                    else:
                        # Fallback: use ASE symbols if df_sorted has no 'symbol' column
                        sym = np.array(symbols, dtype=object)
                        # Map original -> sorted
                        sym_sorted = sym[self.sort_idx]
                        is_ion_row = (sym_sorted != 'H') & (sym_sorted != 'O')
                        ion_rows = np.flatnonzero(is_ion_row)

                    ion_sorted_idx = None
                    ion_symbol = None
                    if ion_rows.size == 1:
                        ion_sorted_idx = int(ion_rows[0])
                        # 2) map SORTED → INVARIANT (invert perm: REF→SORTED)
                        inv_perm = np.empty_like(perm)
                        inv_perm[perm] = np.arange(perm.size)
                        ion_invariant_idx = int(inv_perm[ion_sorted_idx])

                        # 3) ion label
                        if 'symbol' in df_sorted.columns:
                            ion_symbol = str(df_sorted['symbol'].iloc[ion_sorted_idx])
                        else:
                            ion_symbol = str(sym_sorted[ion_sorted_idx])
                    else:
                        ion_invariant_idx = None
                        ion_symbol = None
                    
                    # invariant & physical water labels for QA (canonical naming)
                    df_invariant = df_sorted.iloc[perm].reset_index(drop=True).copy()
                    if self.label_maps.df_sorted_reference is None:
                        raise RuntimeError("label_maps must be initialized before use")
                    df_invariant["orig_idx"] = self.label_maps.df_sorted_reference["orig_idx"].to_numpy()
                    water_labels_invariant = self.index_mapper.label_waters_O2H(df_invariant)

                    # (optional) physical labels on current df (current naming)
                    water_labels = self.index_mapper.label_waters_O2H(df_sorted)

                    # tracker (canonical)
                    key = FrameKey(step=step, op=float(op), ase_idx=ase_idx)
                    self.water_tracker.set(key, water_labels_invariant)
                    report = self.water_tracker.check_consistency(key)
                    jacc = (report or {}).get("jaccard", float("nan"))

                    if op <= 1.25:     
                    #Safeguard such that RC will always be defined on on the same atoms. 
                    #Does not have the same geometric meaning after dissociation, and will affect CVs that depend on actual RC geometries

                        if rc is None:
                            # første pre-diss frame: foreløpig RC
                            rc = select_reaction_center(
                                coords_invariant,
                                self.index_invariant_water_triplets,
                                self.CVs.cell_size
                            )
                            rc_op = float(op)
                        elif rc_op is not None and rc_op < 1.15 and op >= 1.15:
                            rc = select_reaction_center(
                                coords_invariant,
                                self.index_invariant_water_triplets,
                                self.CVs.cell_size
                            )
                            rc_op = float(op)

                    K = getattr(self.pred_power, "k_neighbors", 32) #TODO: make part of config

                    neigh = build_neighborhood(
                        coords=coords_invariant,
                        water_triplets=self.index_invariant_water_triplets,
                        rc=rc, K=K, box=self.CVs.cell_size,
                        names_ref=self.label_maps.names_ref,
                        include_self=True, metric="OO",
                    )

                    has_ion = system_has_ion_from_symbols(symbols)
                    if has_ion:
                        from CVmanager.neighbor_ion_o_dist_cv import NeighborIonODistCV
                        # Only append once per runner construction; or guard with a flag
                        # If you need per-frame logic, keep it here; otherwise move to __init__
                        if not any(getattr(cv, 'name', None) == 'neighbor_IonO_ranked' for cv in self.cv_modules):
                            self.cv_modules.append(NeighborIonODistCV(name="neighbor_IonO_ranked", K=K))  # or 12—match your scheme


                    # build inputs for CVs
                    cv_inputs = CVInputs(
                        coords=coords_invariant,                 # (N,3) in reference order
                        pairs_all=self.index_invariant_pairs,         # global pairs
                        pairs_labels=self.index_invariant_pair_labels,
                        water_triplets=self.index_invariant_water_triplets,      # (H1,O,H2)
                        water_triplet_labels=self.index_invariant_water_labels,
                        ho_pairs=self.index_invariant_ho_pairs,                       # optional
                        hh_pairs=self.index_invariant_hh_pairs,                       # optional
                        flags=flags,                             # contains bad_err / oh_exceeds_cutoff etc.
                        key=key,                                 # FrameKey(step, op, ase_idx)
                        box=self.CVs.cell_size,                  # add if your CVs need PBC corrections
                        water_labels_physical=water_labels,      # optional QA
                        water_labels_invariant=water_labels_invariant,  # optional pretty labels
                        perm=perm,                          # REF -> SORTED
                        sort_idx=self.sort_idx,             # original -> sorted
                        traj_path=traj_path,
                        reaction=rc,                     # Needed for locating additional data_files
                        neighborhood=neigh,               # ReactionNeighborhood
                        ion_index=ion_invariant_idx,      # None if no ion
                        ion_label=ion_symbol,             # "Na"/"Cl" or None
                    )

                    # Attach optional CP2K-derived fields used by some CVs
                    if mull is not None:
                        setattr(cv_inputs, "mulliken_charges", mull)         # (N,)
                    if frc is not None:
                        setattr(cv_inputs, "forces_invariant", frc)          # (N,3)
                    if dip is not None:
                        setattr(cv_inputs, "dipole_total", dip)              # (3,)
                    if homo is not None:
                        centers, spreads = homo
                        setattr(cv_inputs, "homo_centers", centers)          # (M,3)
                        setattr(cv_inputs, "homo_spreads", spreads)          # (M,)

                    if writer is None:
                        writer = H5Writer(cv_root, step=step)
                        # (optional) file-level attrs:
                        writer.init_step_file(
                            names_ref=self.label_maps.names_ref,            # required once per file
                            wham_ngrid=int(self.wham.n_grid),
                            cell_size=float(self.CVs.cell_size),
                            n_atoms=int(len(self.sort_idx)),
                            weight=self.weights.weights.get(step, np.nan),  # <- your dict {step: weight}
                            extra_attrs={"lambda_max": lambda_max}, 
                        )

                    # compute + write
                    for cv in self.cv_modules:
                        # ensure datasets exist (once per grid/cv)
                        if writer.has_frame(grid_id, cv, step=step, ase_idx=ase_idx):
                            continue 
                        writer.ensure_cv_group(grid_id, cv)
                        vals = cv.compute(cv_inputs)

                        # --- DEBUG BLOCK ---
                        try:
                            shape = vals.shape
                        except Exception:
                            shape = "unknown"

                        print(f"[CV DEBUG] {cv.name:30s}  shape={shape}")

                        # Empty CV output
                        if vals.size == 0:
                            print("    -> EMPTY CV (size 0)")
                        else:
                            finite_vals = vals[np.isfinite(vals)]
                            if finite_vals.size > 0:
                                print(f"    range: {finite_vals.min():8.4f} to {finite_vals.max():8.4f}   mean={finite_vals.mean():8.4f}")
                            else:
                                print("    range:  all values are inf/NaN")

                        print(f"    NaNs={np.isnan(vals).sum():3d}    infs={np.isinf(vals).sum():3d}")
                        print("-" * 60)

                        # --- END DEBUG BLOCK ---

                        writer._ensure_dsets(grid_id, cv, K=vals.shape[0])
                        writer.append(
                            grid_id, cv, vals,
                            meta={
                                "step": step, "op": float(op), "ase_idx": ase_idx,
                                "err": err, "flags": flags, "jaccard": jacc
                            }
                        )
            finally:
                if writer is not None:
                    writer.close()


    def load_phasepoint_frame(
        self,
        step_dir: Path,
        frame_info: List[int]  # Just one [grid_idx, frame_idx] pair
        ) -> Tuple[int, Path]:
        """
        Load a single frame from a trajectory file given a RETIS step directory
        and a (grid_idx, frame_idx) pair.
        """
        traj_txt_path = step_dir / "traj.txt"
        if not traj_txt_path.exists():
            raise FileNotFoundError(f"Missing traj.txt in {step_dir}")

        # Read traj.txt
        traj_df = pd.read_csv(
            traj_txt_path,
            skiprows=2,
            sep=r'\s+',
            names=["Step", "Filename", "index", "vel"]
        )
        print(f"Frame info: {frame_info}")
        grid_idx, frame_idx = frame_info

        # Find the row corresponding to frame_idx
        row = traj_df[traj_df["Step"] == frame_idx]
        if row.empty:
            raise ValueError(f"Frame {frame_idx} not found in traj.txt under {step_dir}")

        filename = row["Filename"].iloc[0]
        ase_index = int(row["index"].iloc[0])  # ASE frame index
        traj_path = step_dir / "accepted" / filename

        return ase_index, traj_path
    

def extract_frames(
        directory: str,
        grid_idx: Union[int, np.integer],
        grid_points: float,
    ) -> Tuple[Optional[List[int]], float]:
        """
        Extract frame indices where 'orderp' first crosses each grid point,
        and return a list of [grid_idx, frame] pairs along with the maximum orderp.

        Args:
            directory: Path to the directory containing 'order.txt'.
            grid_idx: Array of grid indices to check.
            grid_points: Array of grid values (physical OP) corresponding to grid_idx.

        Returns:
            frames: List of [grid_idx, frame] for each crossing found.
            lambda_max: The maximum 'orderp' value in the data.
        """
        # Read the order file
        #NOTES: - The order.txt file is expected to be in the format with columns: time, orderp, n_dissociation, atom1, atom2, atom3
        #retis_step = 1000  # Assuming a fixed RETIS step for this example
        #str(retis_step),
        #Remove this later!!!!

        order_file = os.path.join(directory, 'order.txt')
        order_df = pd.read_csv(
            order_file,
            skiprows=2,
            sep=r'\s+',
            names=['Time', 'Orderp', 'n_dissociation', 'atom1', 'atom2', 'atom3']
        )

        lambda_max = float(order_df['Orderp'].max())
        #frames: List[List[int]] = []

        # Loop through each grid point and find first positive crossing
        print(f"Processing grid point {grid_points} (index {grid_idx}) in directory {directory}")
        
        crossing = order_df[
            (order_df["Orderp"] > grid_points) &
            (order_df["Time"] != order_df["Time"].iloc[0]) &
            (order_df["Time"] != order_df["Time"].iloc[-1])
        ]

        if crossing.empty:
            print(
                f"Path: {directory}\n\tNo positive crossing for OP={grid_points} "
                f"(grid_idx {grid_idx})."
            )
            return None, lambda_max           

        # first row of the crossing slice
        frame_idx = int(crossing["Time"].iloc[0])
        print(
            f"Path: {directory}\n\tExtracted frame={frame_idx}  "
            f"lambda_max={lambda_max}"
        )


        return [int(grid_idx), int(frame_idx)], float(lambda_max)

def make_sorted_df_from_frame(
    positions: np.ndarray,
    symbols: list[str],
    sort_idx: np.ndarray        # ← from TypeManager
) -> pd.DataFrame:
    df = pd.DataFrame({
        "element": symbols,
        "orig_idx": np.arange(len(symbols)),
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
    })
    # Re-index to global order
    df = df.set_index("orig_idx").loc[sort_idx].reset_index()
    return df

def system_has_ion_from_symbols(symbols: list[str]) -> bool:
    # Any non-H/O atom counts as an ion for your systems
    return any(s not in ('H', 'O') for s in symbols)



@dataclass(frozen=True)
class FrameKey:
    step: int
    op: float        # or grid_id if you prefer integer indexing
    ase_idx: int

class WaterLabelTracker:
    def __init__(self, jaccard_thresh: float = 0.95):
        self.refs: dict[tuple, set[str]] = {}   # key_bucket -> ref set
        self.last: dict[tuple, set[str]] = {}   # key -> current set (optional)
        self.jaccard_thresh = jaccard_thresh

    @staticmethod
    def _bucket(key) -> tuple:
        # bucket by OP rounded + step (tweak as you like)
        return (int(key.step), round(float(key.op), 3))

    def set(self, key, labels: list[str]):
        S = set(labels)
        b = self._bucket(key)
        if b not in self.refs:
            self.refs[b] = S  # first seen becomes reference for this bucket
        self.last[(b, key.ase_idx)] = S

    def check_consistency(self, key):
        b = self._bucket(key)
        S = self.last.get((b, key.ase_idx))
        if S is None:
            return None  # not set
        R = self.refs[b]
        inter = len(R & S)
        union = len(R | S)
        jacc = inter / union if union else 1.0
        missing = sorted(R - S)
        extra   = sorted(S - R)
        return {"missing": missing, "extra": extra, "jaccard": jacc}









"""
OLD SINGLE-FRAME CVs (keep for reference)
            Just use the neighbour CVs to greatly reduce memory requirements. DistancePairCV generates a very large dataset!
            DistancePairsCV(
                name="pair_dist_all_atoms",
                pairs=self.index_invariant_pairs,
                labels=self.index_invariant_pair_labels,
            ),
            IntraWaterHO_CV(
                name="intra_water_OH",
                pairs=self.index_invariant_ho_pairs,
                labels=np.array([
                    # Build readable O–H labels from your reference names:
            #        f"{self.label_maps.names_ref[iO]}-{self.label_maps.names_ref[iH]}"
                    for (iO, iH) in self.index_invariant_ho_pairs
                ], dtype=object),
            ),
            IntraWaterHOHAngle_CV(
                name="intra_water_HOH_angle",
                triplets=self.index_invariant_water_triplets,
                labels=self.index_invariant_water_labels,   # already like "Hxx-Oyy-Hzz"
                degrees=True,
            ),
            WannierGeomCV(
                name="wannier_geom",
                oxygen_indices=self.index_mapper.idx_O_ref,   # (nO,)
                ho_pairs=self.index_invariant_ho_pairs,       # (2*nO,2), (O,H)
                require_exact_four=False,                     # set True if you prefer strictness → NaNs
            ),
            MullikenChargeCV(
                name="mulliken_charge",
                oxygen_indices=self.index_mapper.idx_O_ref,     # (nO,)
                hydrogen_indices=self.index_mapper.idx_H_ref,   # (nH,)
                ho_pairs=self.index_invariant_ho_pairs,         # (2*nO,2) (O,H)
                sort_idx=self.sort_idx,                         # ORIG -> SORTED
            ),
            ForcesProjectionCV(
            name="forces_projection",
            oxygen_indices=self.index_mapper.idx_O_ref,   # (nO,)
            ho_pairs=self.index_invariant_ho_pairs,       # (2*nO,2) (O,H)
            sort_idx=self.sort_idx,                      # ORIG -> SORTED
            ),
            PTGeometryCV(
            name="pt_geometry",
            oxygen_indices=self.index_mapper.idx_O_ref,   # (nO,)
            ho_pairs=self.index_invariant_ho_pairs,       # (2*nO,2) (O,H) in invariant order
            ),
"""
