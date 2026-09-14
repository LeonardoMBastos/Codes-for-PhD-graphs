#!/usr/bin/env python3

# ======================================================================
# HOMO / LUMO - WSL UBUNTU / LINUX
#
# COMPOSTOS:
#   LIN
#   NER
#   DZP
#   PHB
#   GABA
#
# WORKFLOW:
#
# SDF
#  ↓
# RDKit sanitization
#  ↓
# explicit H addition
#  ↓
# formula / charge / electron validation
#  ↓
# 3D generation if necessary
#  ↓
# MMFF94
#  ↓
# B3LYP/cc-pVDZ SINGLE-POINT
#  ↓
# HOMO / LUMO / HOMO-LUMO GAP
#  ↓
# HOMO.cube / LUMO.cube
#
# NÃO REALIZA OTIMIZAÇÃO GEOMÉTRICA DFT.
# ======================================================================


# ======================================================================
# 0. CONFIGURAÇÕES
# ======================================================================

import os

# ----------------------------------------------------------------------
# CPUs
# ----------------------------------------------------------------------
#
# Deixa 2 CPUs lógicas livres para o sistema.
# Se quiser usar absolutamente todas:
#
# N_THREADS = os.cpu_count() or 2
#

TOTAL_CPUS = os.cpu_count() or 2

N_THREADS = max(
    1,
    TOTAL_CPUS - 2
)

# Evita oversubscription de BLAS/OpenMP
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)


# ----------------------------------------------------------------------
# Método quântico
# ----------------------------------------------------------------------

METHOD = "B3LYP"

BASIS = "cc-pVDZ"


# ----------------------------------------------------------------------
# Estado eletrônico
# ----------------------------------------------------------------------

CHARGE = 0

# PySCF:
# spin = N(alpha) - N(beta)
#
# closed-shell singlet:
# spin = 0

SPIN = 0


# ----------------------------------------------------------------------
# MMFF94
# ----------------------------------------------------------------------

MMFF_MAX_ITERS = 5000


# ----------------------------------------------------------------------
# SCF
# ----------------------------------------------------------------------

DFT_GRID_LEVEL = 3

SCF_MAX_CYCLES = 200

SCF_CONV_TOL = 1e-8


# ----------------------------------------------------------------------
# Arquivos CUBE
# ----------------------------------------------------------------------

# 80 x 80 x 80:
# boa qualidade para visualização final.

CUBE_GRID = 80


# ======================================================================
# 1. IMPORTAÇÕES
# ======================================================================

import sys
import re
import gc
import time
import shutil
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

from pyscf import gto
from pyscf import dft
from pyscf import lib
from pyscf.tools import cubegen


# ======================================================================
# 2. CONFIGURAR PySCF
# ======================================================================

lib.num_threads(
    N_THREADS
)


# ======================================================================
# 3. DIRETÓRIOS
# ======================================================================

# Pasta de onde o script está sendo executado.
INPUT_DIR = Path.cwd()

OUTPUT_DIR = (
    INPUT_DIR /
    "HOMO_LUMO_results"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# 4. CONSTANTES
# ======================================================================

HARTREE_TO_EV = 27.211386245988


# ======================================================================
# 5. COMPOSTOS RECONHECIDOS
# ======================================================================

EXPECTED_FORMULAS = {

    # Linalool
    "lin": "C10H18O",
    "linalool": "C10H18O",

    # Nerolidol
    "ner": "C15H26O",
    "tner": "C15H26O",
    "nerolidol": "C15H26O",

    # Diazepam
    "dzp": "C16H13ClN2O",
    "diazepam": "C16H13ClN2O",

    # Phenobarbital
    "phb": "C12H12N2O3",
    "phenobarbital": "C12H12N2O3",
    "phenobarbitone": "C12H12N2O3",

    # GABA
    "gaba": "C4H9NO2",
    "gammaaminobutyricacid": "C4H9NO2"
}


EXPECTED_ELECTRONS = {

    "C10H18O": 86,       # LIN

    "C15H26O": 124,      # NER

    "C16H13ClN2O": 148,  # DZP

    "C12H12N2O3": 122,   # PHB

    "C4H9NO2": 56        # GABA
}


# ======================================================================
# 6. FUNÇÕES
# ======================================================================


def timestamp():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ----------------------------------------------------------------------


def format_seconds(seconds):

    seconds = int(round(seconds))

    hours, remainder = divmod(
        seconds,
        3600
    )

    minutes, seconds = divmod(
        remainder,
        60
    )

    if hours > 0:

        return (
            f"{hours} h "
            f"{minutes} min "
            f"{seconds} s"
        )

    if minutes > 0:

        return (
            f"{minutes} min "
            f"{seconds} s"
        )

    return f"{seconds} s"


# ----------------------------------------------------------------------


def sanitize_name(name):

    return re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        name
    )


# ----------------------------------------------------------------------


def normalized_name(name):

    return re.sub(
        r"[^a-z0-9]",
        "",
        name.lower()
    )


# ----------------------------------------------------------------------


def identify_expected_formula(name):

    norm = normalized_name(
        name
    )

    # nomes mais longos primeiro
    aliases = sorted(
        EXPECTED_FORMULAS.keys(),
        key=len,
        reverse=True
    )

    for alias in aliases:

        if normalized_name(alias) in norm:

            return EXPECTED_FORMULAS[
                alias
            ]

    return None


# ----------------------------------------------------------------------


def compound_identity(name):

    norm = normalized_name(
        name
    )

    if (
        "linalool" in norm
        or norm == "lin"
        or norm.startswith("lin")
    ):

        return "LIN"


    if (
        "nerolidol" in norm
        or norm == "ner"
        or norm == "tner"
        or norm.startswith("ner")
        or norm.startswith("tner")
    ):

        return "NER"


    if (
        "diazepam" in norm
        or "dzp" in norm
    ):

        return "DZP"


    if (
        "phenobarbital" in norm
        or "phenobarbitone" in norm
        or "phb" in norm
    ):

        return "PHB"


    if (
        "gaba" in norm
        or "gammaaminobutyric" in norm
    ):

        return "GABA"


    return name.upper()


# ----------------------------------------------------------------------


def load_sdf(file_path):

    supplier = Chem.SDMolSupplier(

        str(file_path),

        removeHs=False,

        sanitize=True
    )

    valid = [

        mol

        for mol in supplier

        if mol is not None
    ]


    if len(valid) == 0:

        raise RuntimeError(
            "Nenhuma molécula válida "
            f"encontrada em {file_path.name}."
        )


    if len(valid) > 1:

        print(
            f"AVISO: {file_path.name} contém "
            f"{len(valid)} moléculas."
        )

        print(
            "Somente a primeira será utilizada."
        )


    return valid[0]


# ----------------------------------------------------------------------


def has_real_3d_coordinates(mol):

    if mol.GetNumConformers() == 0:

        return False


    conf = mol.GetConformer()


    coords = []


    for i in range(
        mol.GetNumAtoms()
    ):

        p = conf.GetAtomPosition(i)

        coords.append(
            [
                p.x,
                p.y,
                p.z
            ]
        )


    coords = np.asarray(
        coords
    )


    if not np.all(
        np.isfinite(coords)
    ):

        return False


    z_range = np.ptp(
        coords[:, 2]
    )


    return z_range >= 1e-3


# ----------------------------------------------------------------------


def prepare_rdkit_molecule(mol):

    """
    Sanitização -> H -> 3D (se necessário) -> MMFF94
    """

    Chem.SanitizeMol(
        mol
    )


    had_3d = has_real_3d_coordinates(
        mol
    )


    # ----------------------------------------------------------
    # Adicionar H explícitos
    # ----------------------------------------------------------

    molH = Chem.AddHs(

        mol,

        addCoords=True
    )


    # ----------------------------------------------------------
    # Gerar geometria 3D se SDF for 2D
    # ----------------------------------------------------------

    if not had_3d:

        print(
            "SDF não contém geometria 3D válida."
        )

        print(
            "Gerando conformação com ETKDGv3..."
        )


        molH.RemoveAllConformers()


        params = AllChem.ETKDGv3()

        params.randomSeed = 2026

        params.useRandomCoords = False


        status = AllChem.EmbedMolecule(

            molH,

            params
        )


        if status != 0:

            print(
                "ETKDG padrão falhou."
            )

            print(
                "Tentando random coordinates..."
            )


            params.useRandomCoords = True


            status = AllChem.EmbedMolecule(

                molH,

                params
            )


        if status != 0:

            raise RuntimeError(
                "Não foi possível gerar "
                "uma estrutura 3D."
            )


        print(
            "✓ Geometria 3D gerada."
        )


    else:

        print(
            "✓ Coordenadas 3D encontradas "
            "no SDF."
        )


    # ----------------------------------------------------------
    # MMFF94
    # ----------------------------------------------------------

    if AllChem.MMFFHasAllMoleculeParams(
        molH
    ):

        print(
            "Executando MMFF94..."
        )


        status = AllChem.MMFFOptimizeMolecule(

            molH,

            mmffVariant="MMFF94",

            maxIters=MMFF_MAX_ITERS
        )


        if status == 0:

            print(
                "✓ MMFF94 convergiu."
            )


        elif status == 1:

            print(
                "AVISO: MMFF94 atingiu "
                "o limite de iterações."
            )


        else:

            print(
                f"AVISO: status MMFF94 = {status}"
            )


        method = "MMFF94"


    else:

        print(
            "MMFF94 não possui todos "
            "os parâmetros necessários."
        )

        print(
            "Utilizando UFF como fallback..."
        )


        AllChem.UFFOptimizeMolecule(

            molH,

            maxIters=MMFF_MAX_ITERS
        )


        method = "UFF"


        print(
            "✓ UFF concluído."
        )


    return (
        molH,
        had_3d,
        method
    )


# ----------------------------------------------------------------------


def rdkit_to_pyscf_atoms(mol):

    conf = mol.GetConformer()

    atoms = []


    for atom in mol.GetAtoms():

        i = atom.GetIdx()

        p = conf.GetAtomPosition(i)


        atoms.append(

            (

                atom.GetSymbol(),

                (
                    float(p.x),
                    float(p.y),
                    float(p.z)
                )

            )

        )


    return atoms


# ----------------------------------------------------------------------


def save_rdkit_sdf(
    mol,
    path
):

    writer = Chem.SDWriter(
        str(path)
    )

    writer.write(
        mol
    )

    writer.close()


# ----------------------------------------------------------------------


def write_xyz_from_rdkit(
    mol,
    filename,
    comment
):

    conf = mol.GetConformer()


    with open(
        filename,
        "w"
    ) as f:


        f.write(
            f"{mol.GetNumAtoms()}\n"
        )


        f.write(
            f"{comment}\n"
        )


        for atom in mol.GetAtoms():

            i = atom.GetIdx()

            p = conf.GetAtomPosition(i)


            f.write(

                f"{atom.GetSymbol():3s} "

                f"{p.x:16.8f} "

                f"{p.y:16.8f} "

                f"{p.z:16.8f}\n"

            )


# ----------------------------------------------------------------------


def build_rks(
    mol,
    chkfile=None
):

    mf = dft.RKS(
        mol
    )


    mf.xc = METHOD.lower()


    mf.grids.level = (
        DFT_GRID_LEVEL
    )


    mf.conv_tol = (
        SCF_CONV_TOL
    )


    mf.max_cycle = (
        SCF_MAX_CYCLES
    )


    if chkfile is not None:

        mf.chkfile = str(
            chkfile
        )


    return mf


# ----------------------------------------------------------------------


def run_scf_robust(
    mol,
    chkfile=None
):

    """
    Single-point SCF.

    Se o solver normal não convergir,
    tenta Newton-Raphson.
    """

    mf = build_rks(

        mol,

        chkfile
    )


    print(
        "\nIniciando SCF..."
    )

    print(
        f"Threads: {N_THREADS}"
    )


    energy = mf.kernel()


    if not mf.converged:

        print(
            "\nSCF convencional não convergiu."
        )

        print(
            "Tentando Newton solver..."
        )


        mf = mf.newton()


        mf.max_cycle = (
            SCF_MAX_CYCLES
        )


        energy = mf.kernel()


    if not mf.converged:

        raise RuntimeError(
            "SCF não convergiu."
        )


    return (
        mf,
        energy
    )


# ======================================================================
# 7. INFORMAÇÕES DA EXECUÇÃO
# ======================================================================

print("\n")

print("=" * 76)

print(
    "HOMO / LUMO - PySCF"
)

print("=" * 76)


print(
    f"Data: {timestamp()}"
)


print(
    f"Diretório: {INPUT_DIR}"
)


print(
    f"CPUs lógicas detectadas: {TOTAL_CPUS}"
)


print(
    f"Threads utilizadas: {N_THREADS}"
)


print(
    f"Método: {METHOD}/{BASIS}"
)


print(
    "DFT geometry optimization: NÃO"
)


print(
    "DFT: single-point"
)


# ======================================================================
# 8. LOCALIZAR SDF
# ======================================================================

sdf_files = sorted(

    [

        path

        for path in INPUT_DIR.glob(
            "*.sdf"
        )

        if path.is_file()

    ]

)


if len(sdf_files) == 0:

    print(
        "\nERRO: nenhum arquivo .sdf "
        "foi encontrado."
    )

    print(
        "Coloque os SDF na mesma pasta "
        "e execute novamente."
    )

    sys.exit(1)


print("\n")

print("=" * 76)

print(
    "ARQUIVOS SDF ENCONTRADOS"
)

print("=" * 76)


for file in sdf_files:

    print(
        f" • {file.name}"
    )


print(
    f"\nTotal: {len(sdf_files)}"
)


# ======================================================================
# 9. PROCESSAMENTO
# ======================================================================

results = []

overall_start = time.perf_counter()


for number, sdf_file in enumerate(
    sdf_files,
    start=1
):


    compound_start = (
        time.perf_counter()
    )


    original_name = (
        sdf_file.stem
    )


    compound_label = (
        compound_identity(
            original_name
        )
    )


    safe_name = sanitize_name(
        original_name
    )


    compound_dir = (

        OUTPUT_DIR /

        safe_name
    )


    compound_dir.mkdir(

        parents=True,

        exist_ok=True
    )


    print("\n\n")

    print("=" * 76)

    print(
        f"[{number}/{len(sdf_files)}] "
        f"PROCESSANDO: {sdf_file.name}"
    )

    print(
        f"COMPOSTO: {compound_label}"
    )

    print(
        f"INÍCIO: {timestamp()}"
    )

    print("=" * 76)


    try:


        # ==============================================================
        # 9.1 LEITURA
        # ==============================================================

        print(
            "\n[1/7] Leitura e sanitização"
        )


        mol_original = load_sdf(
            sdf_file
        )


        formula_original = (
            rdMolDescriptors.CalcMolFormula(
                mol_original
            )
        )


        original_charge = (
            Chem.GetFormalCharge(
                mol_original
            )
        )


        smiles_original = (
            Chem.MolToSmiles(

                mol_original,

                isomericSmiles=True
            )
        )


        print(
            f"Fórmula original: "
            f"{formula_original}"
        )


        print(
            f"Carga formal: "
            f"{original_charge}"
        )


        print(
            f"Isomeric SMILES: "
            f"{smiles_original}"
        )


        # ==============================================================
        # 9.2 PREPARAÇÃO
        # ==============================================================

        print(
            "\n[2/7] Preparação estrutural"
        )


        (
            molH,
            had_3d,
            geometry_method

        ) = prepare_rdkit_molecule(
            mol_original
        )


        prepared_formula = (
            rdMolDescriptors.CalcMolFormula(
                molH
            )
        )


        prepared_charge = (
            Chem.GetFormalCharge(
                molH
            )
        )


        prepared_smiles = (
            Chem.MolToSmiles(

                Chem.RemoveHs(
                    molH
                ),

                isomericSmiles=True
            )
        )


        print(
            f"Fórmula preparada: "
            f"{prepared_formula}"
        )


        print(
            f"Carga líquida: "
            f"{prepared_charge}"
        )


        print(
            f"SMILES preparado: "
            f"{prepared_smiles}"
        )


        # ==============================================================
        # 9.3 VALIDAÇÃO
        # ==============================================================

        print(
            "\n[3/7] Validação química"
        )


        expected_formula = (
            identify_expected_formula(
                original_name
            )
        )


        if expected_formula is not None:


            print(
                f"Fórmula esperada: "
                f"{expected_formula}"
            )


            if (
                prepared_formula
                !=
                expected_formula
            ):

                raise RuntimeError(

                    "\nFÓRMULA INCORRETA\n"

                    f"Esperado: {expected_formula}\n"

                    f"Detectado: {prepared_formula}"

                )


            print(
                "✓ Fórmula validada."
            )


        else:

            print(
                "AVISO: nome não reconhecido."
            )

            print(
                "Prosseguindo sem fórmula "
                "de referência."
            )


        if prepared_charge != CHARGE:

            raise RuntimeError(

                "\nCARGA INCORRETA\n"

                f"Carga esperada: {CHARGE}\n"

                f"Carga no SDF: {prepared_charge}"

            )


        print(
            "✓ Carga líquida validada."
        )


        # ==============================================================
        # SALVAR GEOMETRIA MMFF
        # ==============================================================

        prepared_sdf = (

            compound_dir /

            f"{safe_name}_MMFF94.sdf"
        )


        prepared_xyz = (

            compound_dir /

            f"{safe_name}_MMFF94.xyz"
        )


        save_rdkit_sdf(

            molH,

            prepared_sdf
        )


        write_xyz_from_rdkit(

            molH,

            prepared_xyz,

            (
                f"{geometry_method} "
                "optimized geometry"
            )
        )


        # ==============================================================
        # 9.4 MOLÉCULA PySCF
        # ==============================================================

        print(
            "\n[4/7] Preparando PySCF"
        )


        atoms = rdkit_to_pyscf_atoms(
            molH
        )


        mol = gto.M(

            atom=atoms,

            basis=BASIS,

            charge=CHARGE,

            spin=SPIN,

            unit="Angstrom",

            verbose=4
        )


        print(
            f"Átomos: {mol.natm}"
        )


        print(
            f"Elétrons: {mol.nelectron}"
        )


        print(
            f"Carga: {CHARGE}"
        )


        print(
            f"Multiplicidade: {SPIN + 1}"
        )


        if (
            mol.nelectron % 2 != 0
            and
            SPIN == 0
        ):

            raise RuntimeError(

                f"{mol.nelectron} elétrons "
                "para um sistema singlet."

            )


        if expected_formula is not None:


            expected_electrons = (
                EXPECTED_ELECTRONS[
                    expected_formula
                ]
            )


            print(
                f"Elétrons esperados: "
                f"{expected_electrons}"
            )


            if (
                mol.nelectron
                !=
                expected_electrons
            ):

                raise RuntimeError(

                    "\nNÚMERO DE ELÉTRONS INCORRETO\n"

                    f"Esperado: {expected_electrons}\n"

                    f"Detectado: {mol.nelectron}"

                )


            print(
                "✓ Número de elétrons validado."
            )


        # ==============================================================
        # 9.5 SINGLE-POINT
        # ==============================================================

        print(
            "\n[5/7] B3LYP/cc-pVDZ single-point"
        )


        print(
            "Nenhuma otimização geométrica DFT "
            "será realizada."
        )


        dft_start = (
            time.perf_counter()
        )


        chkfile = (

            compound_dir /

            f"{safe_name}.chk"
        )


        (
            mf_final,
            total_energy

        ) = run_scf_robust(

            mol,

            chkfile
        )


        dft_time = (
            time.perf_counter()
            -
            dft_start
        )


        print(
            "\n✓ SCF convergiu."
        )


        print(
            f"Tempo do single-point: "
            f"{format_seconds(dft_time)}"
        )


        # ==============================================================
        # HOMO / LUMO
        # ==============================================================

        occupations = np.asarray(
            mf_final.mo_occ
        )


        energies = np.asarray(
            mf_final.mo_energy
        )


        occupied = np.where(
            occupations > 0
        )[0]


        virtual = np.where(
            occupations == 0
        )[0]


        if len(occupied) == 0:

            raise RuntimeError(
                "Nenhum orbital ocupado."
            )


        if len(virtual) == 0:

            raise RuntimeError(
                "Nenhum orbital virtual."
            )


        homo_idx = (
            occupied[-1]
        )


        virtual_after_homo = (

            virtual[
                virtual > homo_idx
            ]

        )


        if len(
            virtual_after_homo
        ) == 0:

            raise RuntimeError(
                "LUMO não encontrado."
            )


        lumo_idx = (
            virtual_after_homo[0]
        )


        homo_hartree = float(
            energies[homo_idx]
        )


        lumo_hartree = float(
            energies[lumo_idx]
        )


        homo_ev = (
            homo_hartree
            *
            HARTREE_TO_EV
        )


        lumo_ev = (
            lumo_hartree
            *
            HARTREE_TO_EV
        )


        gap_hartree = (
            lumo_hartree
            -
            homo_hartree
        )


        gap_ev = (
            gap_hartree
            *
            HARTREE_TO_EV
        )


        print("\n")

        print("-" * 64)


        print(
            f"{compound_label}"
        )


        print(
            f"Total energy = "
            f"{total_energy:.10f} Hartree"
        )


        print(
            f"HOMO orbital = "
            f"{homo_idx + 1}"
        )


        print(
            f"LUMO orbital = "
            f"{lumo_idx + 1}"
        )


        print(
            f"HOMO = "
            f"{homo_hartree:.8f} Hartree"
        )


        print(
            f"HOMO = "
            f"{homo_ev:.4f} eV"
        )


        print(
            f"LUMO = "
            f"{lumo_hartree:.8f} Hartree"
        )


        print(
            f"LUMO = "
            f"{lumo_ev:.4f} eV"
        )


        print(
            f"HOMO-LUMO gap = "
            f"{gap_ev:.4f} eV"
        )


        print("-" * 64)


        # ==============================================================
        # 9.6 CUBE
        # ==============================================================

        print(
            "\n[6/7] Gerando HOMO/LUMO CUBE"
        )


        cube_start = (
            time.perf_counter()
        )


        homo_cube = (

            compound_dir /

            f"{safe_name}_HOMO.cube"
        )


        lumo_cube = (

            compound_dir /

            f"{safe_name}_LUMO.cube"
        )


        cubegen.orbital(

            mol,

            str(homo_cube),

            mf_final.mo_coeff[
                :,
                homo_idx
            ],

            nx=CUBE_GRID,

            ny=CUBE_GRID,

            nz=CUBE_GRID
        )


        cubegen.orbital(

            mol,

            str(lumo_cube),

            mf_final.mo_coeff[
                :,
                lumo_idx
            ],

            nx=CUBE_GRID,

            ny=CUBE_GRID,

            nz=CUBE_GRID
        )


        cube_time = (
            time.perf_counter()
            -
            cube_start
        )


        print(
            f"✓ {homo_cube.name}"
        )


        print(
            f"✓ {lumo_cube.name}"
        )


        print(
            f"Tempo CUBE: "
            f"{format_seconds(cube_time)}"
        )


        # ==============================================================
        # 9.7 RESULTADOS
        # ==============================================================

        compound_time = (
            time.perf_counter()
            -
            compound_start
        )


        results.append(

            {

                "compound":
                    compound_label,

                "input_file":
                    sdf_file.name,

                "formula":
                    prepared_formula,

                "formal_charge":
                    prepared_charge,

                "isomeric_SMILES":
                    prepared_smiles,

                "input_had_3D":
                    had_3d,

                "geometry_method":
                    geometry_method,

                "DFT_calculation":
                    "single-point",

                "DFT_method":
                    METHOD,

                "basis_set":
                    BASIS,

                "threads":
                    N_THREADS,

                "number_atoms":
                    mol.natm,

                "number_electrons":
                    mol.nelectron,

                "total_energy_Hartree":
                    total_energy,

                "HOMO_orbital":
                    homo_idx + 1,

                "LUMO_orbital":
                    lumo_idx + 1,

                "HOMO_Hartree":
                    homo_hartree,

                "LUMO_Hartree":
                    lumo_hartree,

                "HOMO_eV":
                    homo_ev,

                "LUMO_eV":
                    lumo_ev,

                "HOMO_LUMO_gap_Hartree":
                    gap_hartree,

                "HOMO_LUMO_gap_eV":
                    gap_ev,

                "DFT_time_seconds":
                    dft_time,

                "CUBE_time_seconds":
                    cube_time,

                "total_time_seconds":
                    compound_time,

                "HOMO_cube":
                    homo_cube.name,

                "LUMO_cube":
                    lumo_cube.name

            }

        )


        print(
            "\n[7/7] Finalizado"
        )


        print(
            f"Tempo total de {compound_label}: "
            f"{format_seconds(compound_time)}"
        )


        print(
            f"Fim: {timestamp()}"
        )


        # salvar resultados parciais após CADA molécula
        partial_df = pd.DataFrame(
            results
        )


        partial_df.to_csv(

            OUTPUT_DIR /
            "HOMO_LUMO_summary.csv",

            index=False
        )


        # liberar memória
        del mf_final
        del mol
        del molH

        gc.collect()


    except Exception as exc:


        compound_time = (
            time.perf_counter()
            -
            compound_start
        )


        print("\n")

        print("!" * 76)

        print(
            f"ERRO EM {sdf_file.name}"
        )

        print("!" * 76)


        print(
            str(exc)
        )


        print(
            "\nDetalhes:"
        )


        traceback.print_exc()


        print(
            "\nO script continuará para "
            "o próximo SDF."
        )


        print(
            f"Tempo antes do erro: "
            f"{format_seconds(compound_time)}"
        )


        gc.collect()


# ======================================================================
# 10. RESULTADOS FINAIS
# ======================================================================

overall_time = (
    time.perf_counter()
    -
    overall_start
)


print("\n\n")

print("=" * 76)

print(
    "PROCESSAMENTO FINALIZADO"
)

print("=" * 76)


if len(results) == 0:

    print(
        "Nenhum composto foi processado "
        "com sucesso."
    )

    sys.exit(1)


results_df = pd.DataFrame(
    results
)


summary_file = (

    OUTPUT_DIR /

    "HOMO_LUMO_summary.csv"
)


results_df.to_csv(

    summary_file,

    index=False
)


# ----------------------------------------------------------------------
# tabela enxuta no terminal
# ----------------------------------------------------------------------

display_columns = [

    "compound",

    "formula",

    "number_electrons",

    "HOMO_eV",

    "LUMO_eV",

    "HOMO_LUMO_gap_eV",

    "total_time_seconds"
]


print("\n")

print(

    results_df[
        display_columns
    ].round(4).to_string(
        index=False
    )

)


print("\n")

print(
    f"Tempo total da execução: "
    f"{format_seconds(overall_time)}"
)


print(
    f"\nCSV:\n{summary_file}"
)


print(
    f"\nResultados:\n{OUTPUT_DIR}"
)


print("\n")

print(
    "Arquivos principais de cada composto:"
)

print(
    "  *_MMFF94.sdf"
)

print(
    "  *_MMFF94.xyz"
)

print(
    "  *_HOMO.cube"
)

print(
    "  *_LUMO.cube"
)

print(
    "  *.chk"
)


# ======================================================================
# 11. README
# ======================================================================

readme = (

    OUTPUT_DIR /

    "README_HOMO_LUMO.txt"
)


with open(
    readme,
    "w",
    encoding="utf-8"
) as f:


    f.write(

f"""
HOMO / LUMO ANALYSIS
====================

Environment:
WSL Ubuntu / Linux

Quantum chemistry:
PySCF

Geometry preparation:
RDKit

CPU threads:
{N_THREADS}


WORKFLOW
========

SDF
-> sanitization
-> explicit hydrogen addition
-> formula validation
-> formal charge validation
-> 3D generation if required
-> MMFF94 optimization
-> B3LYP/cc-pVDZ single-point
-> HOMO / LUMO extraction
-> Gaussian CUBE generation


DFT METHOD
==========

Functional:
{METHOD}

Basis set:
{BASIS}

Charge:
{CHARGE}

Multiplicity:
{SPIN + 1}

SCF convergence tolerance:
{SCF_CONV_TOL}

Numerical grid level:
{DFT_GRID_LEVEL}


IMPORTANT
=========

No DFT geometry optimization was performed.

The quantum calculation is a B3LYP/cc-pVDZ
single-point calculation using the MMFF94
preoptimized geometry.


COMPOUNDS
=========

LIN
C10H18O
86 electrons

NER
C15H26O
124 electrons

DZP
C16H13ClN2O
148 electrons

PHB
C12H12N2O3
122 electrons

GABA
C4H9NO2
56 electrons


HOMO-LUMO GAP
=============

Delta E = E(LUMO) - E(HOMO)


CONVERSION
==========

1 Hartree = {HARTREE_TO_EV} eV


CUBE
====

Grid:
{CUBE_GRID} x {CUBE_GRID} x {CUBE_GRID}

For comparative orbital visualization,
use the same isovalue for every compound.

Suggested starting isovalue:

+0.03
-0.03

The positive and negative orbital values
represent opposite phases of the orbital
wavefunction.

They do NOT represent positive and negative
electrostatic charges.


GABA NOTE
=========

Neutral GABA and zwitterionic GABA both have
formula C4H9NO2 and net charge zero.

Formula validation therefore cannot distinguish
these protonation representations.

Check the isomeric SMILES recorded in
HOMO_LUMO_summary.csv.


OUTPUT
======

HOMO_LUMO_summary.csv

and one subdirectory per compound.

"""
    )


print(
    f"\nREADME:\n{readme}"
)

print("\nConcluído.")
