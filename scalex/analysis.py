import os
import scanpy as sc
import pandas as pd
import numpy as np
from gseapy import barplot, dotplot
import gseapy as gp
import matplotlib.pyplot as plt
from collections import Counter
from anndata import concat

from scalex.data import aggregate_data

cell_type_markers_human = {
    'Macrophage': ['MSR1', 'CSF1R', 'C1QA', 'APOE', 'TREM2', 'MARCO', 'MCR1',  'CTSB', 'RBM47', 'FMN1', 'MS4A6A', 'CD68', 'CD163', 'CD206', 'CCL2', 'CCL3', 'CCL4'],
    'Foamy': ['SPP1', 'GPNMB', 'LPL', 'MGLL', 'LIPA', 'PPARG', 'APOE', 'CAPG', 'CTSD', 'LGALS3', 'LGALS1'], # IL4I1# APOE covers SPP1, LPL is a subset of SPP1, while 'LGMN' is exclusive to SPP1+ macrophages
    'Resident': ['SEPP1', 'SELENOP', 'FOLR2', 'F13A1', 'LYVE1'], # 'C1QA', 'C1QB', 'C1QC'], # 'RNASE1',
    'Proliferating': ['MKI67', 'TOP2A', 'TUBB', 'SMC2'], #, 'CDK1', 'CCNB1', 'CCNB2', 'CCNA2', 'CCNE1', 'CDK2', 'CDK4', 'CDK6'],
    'Inflammatory': ['NFKBIA', 'IL1B', 'CXCL2', 'CXCL8', 'IER3', 'SOD2', ], 
    'NK activating': ['FUCA1', 'ENPP2', 'TIGIT', 'CMKLR1', 'KLRK1', 'RASGRP1', 'NR1H3', 'TIMD4'], # Membrane Lipid Catabolic Process, Interleukin-12 Production, Apoptotic cell clearance
    'MonoMac': ['FCN1', 'S100A9', 'S100A8', 'LYZ', 'S100A4'], # NLRP3, 'PLAC8', 'MSRB1'
    'CD16 Mono': ['FCGR3A', 'PLAC8', 'CEBPD', 'CX3CR1'],
    'CD14 Mono': ['S100A9', 'S100A8', 'VCAN'],
    # 'Monocyte': ['FCGR3A', 'CD14', 'CD16', 'VCAN', 'SELL', 'CDKN1C', 'MTSS1'],
    'neutrophils': ['FCGR3B', 'CSF3R', 'CXCR2', 'IFITM2', 'BASP1', 'GOS2'],
    'DC': ['CLEC9A', 'XCR1', 'CD1C', 'CD1A', 'LILRA4'],
    'cDC1': ['CLEC9A', 'IRF8', 'SNX3', 'XCR1'],
    'cDC2': ['FCER1A', 'CD1C', 'CD1E', 'CLEC10A'],
    'migratoryDC': ['BIRC3', 'CCR7', 'LAMP3'],
    'follicular DC': ['FDCSP'],
    'CD207+ DC': ['CD1A', 'CD207'], # 'FCAR1A'],
    # 'Inflammatory_iMacs': ['SOD2', 'CXCL9', 'ACSL1', 'SLAMF7', 'CD44', 'NAMPT', 'CXCL10', 'GBP1', 'GBP2'],
    # 'Macro FABP4+': ['FABP4'],
    'B': ['MS4A1', 'BANK1', 'CD19', 'CD79A',  'IGHM'], # 'CD37',
    'plasma': ['TENT5C', 'MZB1', 'SLAMF7', 'PRDM1', 'FKBP11'],
    'NK': ['GNLY', 'NKG7', 'PRF1'],
    'NKT': ['DCN', 'MGP','COL1A1'],
    'T': ['IL32', 'CCL5', 'CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B',  'CD7', 'TRAC', 'CD3D', 'TRBC2'],
    'Treg': ['FOXP3', 'CD25'], #'UBC', 'DNAJB1'],
    # 'naive T': ['TPT1'],
    'mast': ['TPSB2', 'CPA3', 'MS4A2', 'KIT', 'GATA2', 'FOS2'],
    'pDC': ['CLEC4C', 'TCL1A'], #['IRF7', 'IRF8', 'PLD4', 'MPEG1'],
    'SMC': ['PRKG1', 'EPS8', 'CALD1', 'RBPMS', 'PDE3A', 'DLC1', 'INPP4B'],
    'vascular smooth muscle': ['MYH11', 'ACTA2', 'CNN1'],
    'endothelial': ['MECOM', 'PTPRB', 'LDB2', 'DOCK9', 'PTPRM', 'SULF1'],
    'vascular endothelial': ['VWF', 'CD93', 'CAVN1', 'CD34'],
    'fibroblast': ['PDGFRA', 'FBN1', 'COL5A2', 'INMT', 'MYLK', 'VCAN'],
    'CAFs': ['POSTN', 'BGN', 'LUM', 'CCN2', 'COL5A1'],
    'epithelial': ['KRT8'],
    'Motile ciliated epithelial cells': ['SNTN', 'TSPAN19', 'CCL16', 'DNAAF1', 'SOX2'],
    'cancer cells': ['EPCAM'],
}

cell_type_markers_mouse = {
    'Macrophage': ['Msr1', 'Csf1r', 'C1qa', 'Apoe', 'Trem2', 'Marco', 'Mcr1', 'Ctsb', 'Rbm47', 'Fmn1', 'Ms4a6c', 'Cd68', 'Cd163', 'Mrc1', 'Ccl2', 'Ccl3', 'Ccl4'],
    'Foamy': ['Spp1', 'Gpnmb', 'Lpl', 'Mgll', 'Lipa', 'Pparg', 'Apoe', 'Capg', 'Ctsd', 'Lgals3', 'Lgals1'],
    'Resident': ['Sepp1', 'Selenop', 'Folr2', 'F13a1', 'Lyve1'],
    'Proliferating': ['Mki67', 'Top2a', 'Tubb', 'Smc2'],
    'Inflammatory': ['Nfkbia', 'Il1b', 'Cxcl2', 'Cxcl15', 'Ier3', 'Sod2'],  # CXCL8 human → Cxcl15 mouse
    'MonoMac': ['Fcn1', 'S100a9', 'S100a8', 'Lyz2', 'S100a4'],
    'CD16 Mono': ['Fcgr3', 'Plac8', 'Cebpd', 'Cx3cr1'],  # FCGR3A → Fcgr3
    'CD14 Mono': ['S100a9', 'S100a8', 'Vcan'],
    'neutrophils': ['Fcgr4', 'Csf3r', 'Cxcr2', 'Ifitm2', 'Basp1', 'G0s2'],  # FCGR3B → Fcgr4
    'DC': ['Clec9a', 'Xcr1', 'Cd1d1', 'Cd1d2', 'Lilra6'],  # CD1C/CD1A/LILRA4 human → Cd1d1/2, Lilra6 in mouse
    'cDC1': ['Clec9a', 'Irf8', 'Snx3', 'Xcr1'],
    'cDC2': ['Fcer1a', 'Cd1d1', 'Cd1d2', 'Clec10a'],
    'migratoryDC': ['Birc3', 'Ccr7', 'Lamp3'],
    'follicular DC': ['Fdcsp'],
    'CD207+ DC': ['Cd1d1', 'Cd207'],
    'B': ['Ms4a1', 'Bank1', 'Cd19', 'Cd79a', 'Ighm'],
    'plasma': ['Tent5c', 'Mzb1', 'Slamf7', 'Prdm1', 'Fkbp11'],
    'NK': ['Gnly', 'Nkg7', 'Prf1'],
    'NKT': ['Dcn', 'Mgp', 'Col1a1'],
    'T': ['Itgb6', 'Itga1', 'Lat', 'Cd3g', 'Gimap4', 'Il32', 'Ccl5', 'Cd3d', 'Cd3e', 'Cd3g', 'Cd4', 'Cd8a', 'Cd8b1', 'Cd7', 'Trac', 'Cd3d', 'Trbc2'],
    'Treg': ['Foxp3', 'Il2ra'],  # CD25 = IL2RA
    'mast': ['Tpsb2', 'Cpa3', 'Ms4a2', 'Kit', 'Gata2', 'Fos'],
    'cardiomyocytes': ['Nppa', 'Tnnt2', 'Actc1', 'Actn2'], 
    'vascular smooth muscle': ['Myh11', 'Acta2', 'Cnn1'],
    'endothelial': ['Mecom', 'Ptprb', 'Ldb2', 'Dock9', 'Ptprm', 'Sulf1'],
    'vascular endothelial': ['Vwf', 'Cd93', 'Cavn1', 'Cd34'],
    'adventitial fibroblasts': ['Pi16'],
    'valve fibroblasts': ['Tbx20', 'Erbb4', 'Wff1', 'Clu', 'Thbs1', 'Cadm1', 'Pdzm4', 'Kcnq5'],
    'adv Fibroblasts-1': ['Ebf2', 'Ngf'],
    'adv Fibroblasts-2': ['Ntrk3', 'Lamc3', 'Eln', 'Prdm6'],
    'fibroblast': ['Pi16', 'Pdgfra', 'Fbn1', 'Col5a2', 'Inmt', 'Mylk', 'Vcan'],
    'epithelial': ['Krt8'],
    'mesothelia': ['Msln', 'Upk3b', 'Ezr', 'Efemp1', 'C2'],
    'cancer cells': ['Epcam'],
    'glial cells': ['Abca8a', 'Itgb4', 'Fabp7', 'Cnp', 'Ncam1', 'Atp1b2'],
    'adipocyte': ['Lipe', 'Fasn', 'Adipoq', 'Cyp2e1', 'Pcx']
}

macrophage_markers = {
    'Foam_SPP1-': ['ACP5', 'ADTRP', 'AFAP1L1', 'ALDH1A1', 'ALOX15B', 'ANKRD29', 'BEAN1', 'CCL18', 'CD22', 'CHIT1', 
                   'CHRNA1', 'CPEB1', 'CPM', 'CYP27A1', 'DNAJC5B', 'FCHO2', 'GALNTL6', 'HS3ST2', 'HS3ST3A1', 'HTRA4', 
                   'ITGAD', 'KCNA2', 'KCNJ5', 'KCNMA1-AS3', 'KLHDC8B', 'LINC01091', 'LINC01500', 'LINC01648', 'LINC01857', 
                   'LRRC39', 'MCOLN3', 'MLPH', 'MYO1D', 'PDE3A', 'PKD2L1', 'PNPLA3', 'RARRES1', 'SLC18B1', 'SLC19A2', 
                   'SLC47A1', 'SULT1C2', 'TFCP2L1', 'TMEM255A', 'ZNF589'],
    'Foam_SPP1+': ['ADCY3', 'ANXA2', 'ARHGAP10', 'ATP6V0D2', 'C4orf45', 'CD109', 'CELSR1', 'CSF1', 'CSTB', 'CT69', 'DPP4', 
                  'FABP5', 'FNIP2', 'HRK', 'ITGAX', 'ITGB3', 'KCP', 'KIAA0319', 'LGALS3', 'LINC01010', 'LINC02099', 'MGLL', 
                  'MIR155HG', 'MYO1E', 'NUPR1', 'PRKCH', 'RAI14', 'RALA', 'RCAN3', 'SCD', 'SH3D21', 'SHC3', 'SLC22A15', 'LPL', 
                  'SLC26A11', 'SLC43A3', 'SLC5A4-AS1', 'SPP1', 'TM4SF19', 'TM4SF19-AS1', 'WHRN', 'ZBTB7C', 'ZFYVE28', 'ZMIZ1-AS1'],
    # 'Foam Shared': ['ACE', 'BBS5', 'CBLB', 'CLIP4', 'DCSTAMP', 'GPNMB', 'KCNMA1', 'MITF', 'PLA2G7', 'PLPP3', 'PPARG', 'SAMD4A', 'SLC38A6', 'SNTB1', 'TPRG1'], 
    'Resident': ['ABCA9-AS1', 'ACSM4', 'C4BPB', 'CD163', 'CD163L1', 'CD209', 'CLEC4G', 'COLEC12', 
                'CR1', 'EDA', 'F13A1', 'FCER2', 'FGF13', 'FOLR2', 'GFRA2', 'IL2RA', 'ITSN1', 
                'LGI2', 'LILRB5', 'LINC01645', 'LINC01839', 'LYPLAL1-AS1', 'LYVE1', 'MAMDC2', 
                'ME1', 'MPPED2', 'MRC1', 'NAV2-AS4', 'NAV2-IT1', 'NEURL2', 'PDGFC', 'PLEKHG5', 
                'PLTP', 'RGL1', 'SCN9A', 'SELENOP', 'SHE', 'SLC39A12', 'SLC40A1', 'STON2', 
                'TBX1', 'TDRD10', 'THBS1', 'TMEM236', 'TRIM50'],
    'MonoMac': ['AATBC', 'AC104809.2', 'ADGRE3', 'APOBEC3A', 'ARHGAP29', 'CD300E', 'CDA', 'CEACAM3', 
                'CFP', 'EIF4E3', 'F5', 'FCN1', 'GBP5', 'GLT1D1', 'GPR174', 'LILRA1', 'LILRA5', 'LIN7A', 
                'LINC01619', 'LINC02085', 'LIPN', 'MCTP2', 'MEFV', 'NEBL', 'NLRP12', 'P2RX1', 'PADI4', 
                'PFKFB4', 'PLAC8', 'PLCB1', 'PRR16', 'PTGER2', 'S100A4', 'SLC2A6', 'SPOCK1', 'TMTC1', 'VCAN', 'VCAN-AS1'],
}

selected_GO_terms = [

]

gene_modules = {
    # --- Foam Cell & Lipid Metabolism ---
    'Foam_SPP1+': ['ADCY3', 'ANXA2', 'ARHGAP10', 'ATP6V0D2', 'C4orf45', 'CD109', 'CELSR1', 'CSF1', 'CSTB', 'CT69', 'DPP4', 
                  'FABP5', 'FNIP2', 'HRK', 'ITGAX', 'ITGB3', 'KCP', 'KIAA0319', 'LGALS3', 'LINC01010', 'LINC02099', 'MGLL', 
                  'MIR155HG', 'MYO1E', 'NUPR1', 'PRKCH', 'RAI14', 'RALA', 'RCAN3', 'SCD', 'SH3D21', 'SHC3', 'SLC22A15', 'LPL', 
                  'SLC26A11', 'SLC43A3', 'SLC5A4-AS1', 'SPP1', 'TM4SF19', 'TM4SF19-AS1', 'WHRN', 'ZBTB7C', 'ZFYVE28', 'ZMIZ1-AS1'],
    'Foam_SPP1-': ['ACP5', 'ADTRP', 'AFAP1L1', 'ALDH1A1', 'ALOX15B', 'ANKRD29', 'BEAN1', 'CCL18', 'CD22', 'CHIT1', 
                   'CHRNA1', 'CPEB1', 'CPM', 'CYP27A1', 'DNAJC5B', 'FCHO2', 'GALNTL6', 'HS3ST2', 'HS3ST3A1', 'HTRA4', 
                   'ITGAD', 'KCNA2', 'KCNJ5', 'KCNMA1-AS3', 'KLHDC8B', 'LINC01091', 'LINC01500', 'LINC01648', 'LINC01857', 
                   'LRRC39', 'MCOLN3', 'MLPH', 'MYO1D', 'PDE3A', 'PKD2L1', 'PNPLA3', 'RARRES1', 'SLC18B1', 'SLC19A2', 
                   'SLC47A1', 'SULT1C2', 'TFCP2L1', 'TMEM255A', 'ZNF589'],
    'Foam Cell Differentiation': [
        'CSF1', 'ITGB3', 'NR1H3', 'PPARG', 'CD36', 'LIPA', 
        'TREM2', 'SPP1', 'GPNMB', 'MSR1', 'LPL', 'ABCG1', 'CD9', 'LGALS3', 'FABP4'
    ],
    'Cell Adhesion & Migration': [
        'ITGA3', 'PALLD', 'MYOF', 'MYO1E', 'ASAP1', 'DOCK3', 
        'ICAM1', 'VCAM1', 'ITGAM', 'ITGAX', 'SELE'
    ],
    'Regulation of Foam Cell Differentiation': ['ADIPOQ', 'MSR1', 'PRKCH', 'PPARA', 'NR1H3', 'PLA2G2A', 'ABCA5', 'NR1H2', 'IL18', 'AGT', 'NFKB1', 
                                                'ALOX15B', 'NFKBIA', 'LCP1', 'PLA2G10', 'ABCG1', 'AGTR1', 'PPARG', 'PF4', 'CRP', 'CETP', 'CSF2', 
                                                'CSF1', 'ITGB3', 'PLA2G3', 'PLA2G5', 'MAPK9', 'ITGAV', 'CD36', 'APOB'],
    'Lipid Metabolism': ['STARD3', 'G6PD', 'HMGCS1', 'APOA2', 'APOA4', 'DHCR24', 'IL4', 'SOAT2', 'APOA1', 'CYP11B2', 'FDX1', 'CYP11B1', 
                         'SOAT1', 'ABCG1', 'DHCR7', 'CES1', 'AKR1D1', 'SH2D2A', 'LCAT', 'HSD17B7', 'CYP7A1', 'CH25H', 'SULT2A1', 'NPC1L1', 
                         'PMVK', 'ARV1', 'LBR', 'APOL2', 'FDFT1', 'CYP51A1', 'OSBPL5', 'ABCA5', 'LSS', 'NR0B2', 'ACLY', 'SQLE', 'CYP27A1', 
                         'SULT2B1', 'GLB1', 'PLPP6', 'SNX17', 'ANGPTL3', 'MVD', 'PPARD', 'OSBPL1A', 'MVK', 'CETP', 'INSIG2', 'INSIG1', 'GBA2', 
                         'HMGCR', 'CYP46A1', 'GBA1', 'MSMO1', 'CLN8', 'TM7SF2', 'CLN6', 'TSKU', 'PIP4P1', 'NSDHL', 'CYP11A1', 'APOE', 
                         'SCARF1', 'APOF', 'LDLRAP1', 'ACAA2', 'APOB'],
    'Lipid Storage': [
        'ITGB3', 'NR1H3', 'PPARG', 'TTC39B', 
        'PLIN2', 'PLIN3', 'DGAT1', 'DGAT2', 'SOAT1', 'ACAT1', 'LIPE', 'PNPLA2'
    ],
    'Cholesterol Efflux': [
        'ABCG8', 'ABCA7', 'ABCA5', 'APOA2', 'ABCA8', 'APOA5', 'APOA4', 'NPC1', 'STX12', 'TSKU', 'APOM', 'APOC1', 'NPC2', 'SOAT2', 'APOA1', 
        'APOC3', 'APOC2', 'SOAT1', 'ABCG1', 'APOE', 'ABCG4', 'ABCG5', 'SCARB1'
    ],
    'EGFR Signaling': ['ITGA1', 'RHBDF2', 'PTPN2', 'RHBDF1', 'DUSP3', 'PDE6H', 'PDE6G', 'WDR54', 'PTPN12', 'PLAUR', 'PTK6', 'MMP9', 'AGT', 
                       'DAB2IP', 'CCDC88A', 'CEACAM1', 'FER', 'RNF126', 'APEX1', 'HIP1R', 'SHKBP1', 'CDH13', 'CNOT9', 'PTPN3', 'ERRFI1', 
                       'HIP1', 'TSG101', 'BCAR3', 'RALB', 'DGKD', 'RALA', 'MVP', 'CBLC', 'IFI6', 'CBL', 'EGFR', 'PTPRJ', 'NEU3', 'RNF115', 
                       'ZFYVE28', 'GPER1', 'NUP62', 'MVB12B', 'MVB12A', 'AGR2', 'FAM83B'],
    # 'Anti-Inflammatory': ['PTPN2', 'NR1H3', 'NR1H2', 'PPARG', 'OTOP1', 'PARP14'],
    'Anti-Lipid Storage': ['CES1', 'CRP', 'PTPN2', 'OSBPL8', 'PPARA', 'ABHD5', 'NR1H3', 'ITGB3', 'NR1H2', 'TNF', 'TREM2', 'TTC39B', 'NFKBIA',
                            'IL6', 'ABCG1', 'PPARG', 'ITGAV', 'PNPLA2'],
    'Phagosome': [
        'TCIRG1', 'TUBA1B', 'CTSL', 'SEC61G', 'RAB7B', 'ITGB1', 'ITGB2', 'ITGB5', 'LAMP1', 
        'MSR1', 'ATP6V0C', 'ATP6V0B', 'TFRC', 'CALR', 'TUBB6', 'TUBA1C', 'MARCO', 'ATP6V1F', 
        'CD36', 'ATP6V1G1', 'RAB5A', 'RAB7A', 'EEA1'
    ],
    'Endocytosis': [
        'DNM3', 'MYO1E', 'CTTN', 'MYO6', 'MICALL1', 'ENTHD1', 'APOE', 'CD36', 'LRP12', 'ATP9A', 
        'CORO1C', 'CLTC', 'CAV1', 'DNM2'
    ],
    'ECM remodeling': [
        'MMP9', 'MMP14', 'TIMP3', 'COL6A1', 
        'MMP12', 'MMP2', 'MMP3', 'CTSS', 'CTSK', 'TIMP1', 'TIMP2'
    ],
    'Membrane remodeling': [
        'DCSTAMP', 'MYOF', 'ANO5', 'TM4SF19', 'TSPAN19', 
        'CHMP4B', 'ESCRT', 'SNX1', 'SNX2'
    ],
    'ECM disassembly': ['GSN', 'MMP7', 'MMP8', 'MMP1', 'EXOC8', 'MMP2', 'CMA1', 'MMP3', 'MMP9', 'MMP11', 'MMP10', 'MMP13', 'MMP12', 'MMP15', 'KIF9',
                         'MMP14', 'LCP1', 'ADAM15', 'MMP19', 'ADAM8', 'PRSS2', 'PRSS1', 'HPN', 'TMPRSS6', 'FURIN', 'LAMC1', 'NOXO1', 'KLK5', 'KLK4', 
                         'CAPG', 'PLG', 'CTSV', 'CTSS', 'KLK7', 'ADAMTS5', 'ADAMTS15', 'ADAMTS4', 'SH3PXD2B', 'CTSK', 'CTSG', 'FLOT1', 'WASHC1', 'TPSAB1', 'ELANE'],
    # 'Membrane Lipid Catabolic Process': [
    #     'FUCA1', 'SGPL1', 'SMPD2', 'MGST2', 'NAGA', 'ENPP2', 'PPT1', 'SMPDL3B', 
    #     'GBA', 'GLA', 'ASAH1'
    # ],

    # --- Energy Metabolism ---
    'Migration': [
        'VCAN', 'VCAN-AS1', 'CEACAM3', 'S100A4', 'ADGRE3', 'SPOCK1', 
        'RAC1', 'RHOA', 'CDC42', 'ACTB'
    ],
    'Oxidative phosphorylation': [
        'TCIRG1', 'UQCR11', 'COX5B', 'COX6A1', 'COX6C', 'COX8A', 'UQCRQ', 'NDUFAB1', 'NDUFB2', 
        'NDUFB7', 'NDUFS5', 'ATP5F1B', 'ATP5MC1', 'ATP5MC3', 'ATP5ME', 'ATP6V0C', 'ATP6V0B', 
        'ATP6V1F', 'ATP6V1G1', 'ATP5MF', 'MT-CO1', 'MT-CO2', 'SDHB'
    ],
    'Glycolysis': [
        'GALM', 'ENO1', 'FBP1', 'ALDOA', 'GAPDH', 'PGAM1', 'PGK1', 'PKM', 'TPI1', 
        'HK2', 'PFKP', 'LDHA', 'SLC2A1'
    ],
    'Glycolytic Process': ['TPI1', 'PGAM1', 'PKLR', 'PGAM2', 'GCK', 'PKM', 'PFKM', 'PFKL', 'HKDC1', 'PGAM4', 'GAPDHS', 'ALDOC', 
                           'ALDOB', 'GAPDH', 'GPI', 'PFKFB1', 'PFKFB2', 'UCP2', 'ENO1', 'ENO2', 'ENO3', 'HK1', 'ENO4', 'HK3', 'HK2', 'LDHA', 'PGK2', 'PGK1' ],
    'MHC Class I': [
        'PDIA3', 'ACE', 'SAR1B', 'ERAP1', 'ERAP2', 'HFE', 'IDE', 'TAP2', 'HLA-A', 'TAP1', 
        'TAPBPL', 'IFI30', 'TAPBP', 'MFSD6', 'CLEC4A', 'B2M', 'CALR', 'FCER1G', 
        'HLA-B', 'HLA-C', 'HLA-E'
    ],
    'MHC Ib': ['HLA-C', 'HLA-A', 'HLA-B', 'HLA-G', 'HLA-E', 'HLA-F', 'RAET1E', 'RAET1G', 'ULBP1', 'ULBP3', 'MICA', 'ULBP2', 'MICB'],
    'MHC I': ['HLA-DRB5', 'HLA-A', 'DNM2', 'HLA-E', 'HLA-F', 'FCGR2B', 'PIKFYVE', 'CLEC4A', 'GNAO1', 'HLA-DRA', 'LGMN', 'HLA-DRB4', 
         'HLA-DRB3', 'HLA-DQB2', 'HLA-DRB1', 'CTSV', 'IFI30', 'CTSS', 'HLA-DMB', 'MFSD6', 'CTSL', 'HLA-DPA1', 'CTSF', 'FCER1G',
           'HLA-DOB', 'CTSE', 'HLA-DMA', 'CTSD', 'HLA-DQA2', 'HLA-DOA', 'HLA-DQA1'],
    #  'Exogenous Peptide Antigen Via MHC Class I': ['IKBKB', 'MFSD6', 'HLA-A', 'LNPEP', 'FCER1G', 'IFI30'],
    # --- Cell Death & Survival ---
    'Apoptosis': [ # Regulation / Early
        'IER3', 'EGR1', 'BTG1', 'BCL2A1', 'PTGER2', 'PTGS2', 'MCL1', 'CDKN1A', 'G0S2', 
        'BCL2', 'BCL2L1', 'FAS', 'TNFRSF10B'
    ],

    # 'Antioxidant defense / lipid peroxide removal': [
    #     'GPX4', 'SLC7A11', 'SLC3A2', 'GSH', 'FSP1', 'GCH1', 
    #     'NQO1', 'TXN', 'TXNRD1', 'SRXN1', 'GCLC', 'GCLM'
    # ],
    # 'Detoxification': [
    #     'ALOX5AP', 'APOE', 'GPX3', 'MGST2', 'MGST3', 'PRDX1', 'AKR1A1', 'SESN1', 'PRXL2A', 
    #     'GSTP1', 'GSTM1', 'EPHX1'
    # ],
        # --- Proliferation & Regulation ---
    'Proliferating': [
        'LINC01572', 'POLQ', 'DIAPH3', 'NSD2', 'CIT', 'STAG1', 'SMC4', 'NCAPG2', 'KNL1', 'EZH2', 
        'CENPP', 'ATAD2', 'BRIP1', 'MELK', 'CENPK', 'GTSE1', 'C21orf58', 'TOP2A', 'ANLN', 'NUSAP1', 
        'SUGP2', 'RRM2', 'CEP128', 'CENPE', 'ASPM', 'BRCA1', 'KIF11', 'KIF18B', 'DEK', 'BUB1B', 
        'CENPF', 'MIR924HG', 'ARHGAP11B', 'NCAPG', 'TACC3', 'BARD1', 'IQGAP3', 'STIL', 'FANCA', 
        'ASPH', 'BRCA2', 'GEN1', 'MKI67', 'Z94721.1', 'LINC00342', 'KIF4A', 'DTL', 'CCDC18', 
        'KIFC1', 'CIP2A', 'PRIM2', 'RTTN', 'TMPO', 'TPX2', 'FANCD2', 'KIF14', 'SGO2', 'CENPI', 
        'AURKB', 'KNTC1', 'FANCI', 'KIF15', 'CEP192', 'NDC80', 'RRM1', 'CLSPN', 'POLE2', 'NCAPD3', 
        'KIF20B', 'FOXM1', 'HMGN2', 'RBL1', 'NCAPH', 'FAM111A', 'ATAD5', 'CDK5RAP2', 'MASTL', 
        'PRC1', 'PRR11', 'SHCBP1', 'KIF2A', 'WDR62', 'PARPBP', 'MIR4435-2HG', 'CKAP2L', 'SMC6', 
        'CYTOR', 'ZGRF1', 'ARHGAP11A', 'CKAP5', 'NUF2', 'PLK4'
    ],

    # --- Inflammation & Immune Signaling ---

    'Inflammatory': [
        'IL1B', 'NLRP3', 'G0S2', 'SOD2', 'CEBPB', 'NFKBIA', 'INHBA', 'TNF', 'TREM1', 
        'CCL2', 'CCL20', 'CCL7', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL8', 
        'IL6', 'IL1A', 'IL18', 'PTGS2'
    ],
    'Inflammatory Response': [
        'CXCL8', 'SLC11A1', 'C5AR2', 'WNT5A', 'C5AR1', 'NLRC4', 'CXCL3', 'CXCL2', 
        'THBS1', 'NFKB1', 'IL6', 'OLR1', 'CD44', 'TLR4', 'TLR2', 'MYD88', 'CD14'
    ],

    'Cytokine Production': [
        'APP', 'SLC11A1', 'WNT5A', 'LAPTM5', 'EIF2AK3', 'HIF1A', 'MALT1', 'EREG', 
        'TNF', 'IL10', 'TGFB1', 'IL12B', 'IL23A'
    ],

    # --- Antigen Presentation ---

    # 'MHC Class II': [
    #     'HLA-DRB5', 'DNM2', 'FCGR2B', 'PIKFYVE', 'GNAO1', 'HLA-DRA', 'LGMN', 'HLA-DRB4', 'HLA-DRB3', 'HLA-DQB2', 'HLA-DRB1', 'MARCHF1', 'CTSV', 'IFI30', 
    #     'CTSS', 'HLA-DMB', 'CTSL', 'HLA-DPA1', 'MARCHF8', 'CTSF', 'FCER1G', 'HLA-DOB', 'CTSE', 'HLA-DMA', 'CTSD', 'HLA-DQA2', 'HLA-DOA', 'HLA-DQA1'
    # ],

   

    # --- Structural & Functional ---
    # 'Phagocytosis': [
    #     'YES1', 'LYN', 'SRC', 'LIMK1', 'PRKCD', 'SYK', 'PRKCE', 'PLA2G6', 'PLD2', 'VAV1', 
    #     'HCK', 'VAV2', 'PTK2', 'VAV3', 'FGR', 'FCGR2B', 'PAK1', 'FYN', 'MYO1G', 
    #     'MERU', 'AXL', 'TYRO3', 'GULP1'
    # ],
    # 'Regulation of Phagocytosis': ['TUB', 'IL15', 'SFTPD', 'APLP2', 'APOA2', 'AZU1', 'MERTK', 'IL1B', 'PTPRC', 'RAB31', 'FCGR2B', 'APOA1',
    #                                 'IFNG', 'SPACA3', 'CALR', 'DOCK2', 'GAS6', 'FCN1', 'FPR2', 'TNF', 'IL2RG', 'PTPRJ', 'TREM2', 'C4A', 
    #                                 'SIRPB1', 'C4B', 'PYCARD', 'UXT', 'CLEC7A', 'CD300LF', 'FCER1G', 'LYAR', 'CAMK1D', 'IL15RA', 'AHSG', 'NCKAP1L'],

    'Angiogenesis': [
        'BTG1', 'CXCL8', 'FLT1', 'CEMIP2', 'WNT5A', 'HIF1A', 'THBS1', 'HIPK2', 'VEGFA', 
        'IL6', 'RGCC', 'AGO2', 'HMOX1', 'CTNNB1', 'GLUL', 
        'KDR', 'FGF2', 'PDGFA'
    ],
    'Iron metabolism': [
        'TFRC', 'FTH1', 'FTL', 'FPN1', 'NCOA4', 'HMOX1', 
        'SLC40A1', 'IREB2', 'ACO1'
    ],
    'Ferroptosis': [
        'FTH1', 'FTL', 'GPX4', 'SLC40A1', 'HMOX1', 
        'ACSL4', 'TFRC', 'NCOA4', 'SLC7A11', 'SAT1', 'GSS'
    ],

    'MonoMac': ['AATBC', 'AC104809.2', 'APOBEC3A', 'ARHGAP29', 'CD300E', 'CDA', 'CEACAM3', 
                'CFP', 'EIF4E3', 'F5', 'FCN1', 'GBP5', 'GLT1D1', 'GPR174', 'LILRA1', 'LILRA5', 'LIN7A', 
                'LINC01619', 'LINC02085', 'LIPN', 'MCTP2', 'MEFV', 'NEBL', 'NLRP12', 'P2RX1', 'PADI4', 
                'PFKFB4', 'PLAC8', 'PLCB1', 'PRR16', 'PTGER2', 'SLC2A6', 'SPOCK1', 'TMTC1', 'VCAN', 'VCAN-AS1'],
    'Myeloid Differentiation': ['LYN', 'TET2', 'JAK2', 'HIPK2'],
    'Response To Cytokine': ['RIPOR2','GBP5','PID1','LRRK2','IRAK3','MNDA','LILRB2','JAK2'],
    'Protein Phosphorylation': ['LYN','USP25','MAP3K1','PRKCB','LRRK2','PTEN','IRAK3','SSH2','HIPK2','PTPRC','STK17B','SIK3','FYN','JAK2'],
    'Negative Regulation Of Immune Response': ['CCR2', 'SPINK5', 'LYN', 'CD300A', 'AMBP', 'MUL1', 'RC3H1', 'LILRB1', 'FOXP3', 'ATG12', 'MAPK14', 
                                                            'COL3A1', 'FCGR2B', 'FER', 'TRAFD1', 'SMCR8', 'PDCD1', 'ALOX15', 'TNFAIP3', 'CGAS', 'DTX4', 'NLRC3', 
                                                            'TREM2', 'AURKB', 'IFNL1', 'PPP6C', 'IFI16', 'IRAK3', 'AKT1', 'BANF1', 'HAVCR2', 'LYAR', 'CR1', 'RHBDF2', 
                                                            'YES1', 'SYK', 'PARP1' ,  ],
    'Resident': ['ABCA9-AS1', 'ACSM4', 'C4BPB', 'CD163', 'CD163L1', 'CD209', 'CLEC4G', 'COLEC12', 
                'CR1', 'EDA', 'F13A1', 'FCER2', 'FGF13', 'FOLR2', 'GFRA2', 'IL2RA', 'ITSN1', 
                'LGI2', 'LILRB5', 'LINC01645', 'LINC01839', 'LYPLAL1-AS1', 'LYVE1', 'MAMDC2', 
                'ME1', 'MPPED2', 'MRC1', 'NAV2-AS4', 'NAV2-IT1', 'NEURL2', 'PDGFC', 'PLEKHG5', 
                'PLTP', 'RGL1', 'SCN9A', 'SELENOP', 'SHE', 'SLC39A12', 'SLC40A1', 'STON2', 
                'TBX1', 'TDRD10', 'THBS1', 'TMEM236', 'TRIM50'],
    'Response to Cytokine': [
        'CIITA', 'PID1', 'IL1R1', 'FLT3', 'AFF3', 
        'JAK1', 'JAK2', 'STAT1', 'STAT3', 'SOCS3', 'IRF1'
    ],
    # 'ADCP_CRIPSRo_Up100':['CD47', 'GNE', 'CMAS', 'NANS', 'C1GALT1C1', 'QPCTL', 'SLC35A1',
    #    'MS4A1', 'CAB39', 'UBE2D3', 'ARID1A', 'PDCD10', 'C1GALT1', 'PTEN',
    #    'APMAP', 'RTN4IP1', 'AIFM1', 'FDX1', 'NDUFA1', 'GTPBP6', 'NDUFS8',
    #    'SMARCC1', 'TACO1', 'CMC1', 'ATP5SL', 'SLC39A9', 'SS18', 'CHMP1A',
    #    'GRSF1', 'C17ORF89', 'NDUFAF7', 'PDE12', 'UQCC1', 'NDUFAF5',
    #    'HIGD2A', 'NDUFB9', 'MECR', 'WDR1', 'COX18', 'RHOH', 'HMGB1',
    #    'SLC25A1', 'NDUFS6', 'FOXO1', 'TMEM261', 'HMGB2', 'CS', 'POU2F2',
    #    'ADAM10', 'NDUFB6', 'MTIF3', 'MTO1', 'UBR4', 'NDUFB4', 'LIPT2',
    #    'HMHA1', 'YBEY', 'ALAD', 'NXT1', 'OTUB1', 'NDUFV1', 'GTPBP3',
    #    'NUBPL', 'NDUFS2', 'NDUFB11', 'LDB1', 'SAMD4B', 'C1ORF233',
    #    'ZBTB7A', 'TIMMDC1', 'STK4', 'AP000721.4', 'STARD7', 'NFIA',
    #    'UBE2K', 'VPS37A', 'NDUFA9', 'NDUFA8', 'ELOVL1', 'COX5B', 'PTPRC',
    #    'NDUFAF3', 'ACTB', 'HMBS', 'NDUFC1', 'ARID1B', 'SPI1', 'TMEM38B',
    #    'NDUFS7', 'PRKCD', 'STUB1', 'P2RY8', 'NDUFS1', 'SMS', 'MRPL24',
    #    'LIPT1', 'SCYL1', 'CLCC1', 'ARHGEF1', 'SMARCD2'],
    # 'ADCP_CRIPSRo_Down100':['SPERT', 'PCGF3', 'AC007401.2', 'RP11-195F19.5', 'TRIM25',
    #    'MAGEA3', 'IGHV1OR21-1', 'PITX1', 'KIR3DL3', 'BPHL', 'PRKAB1',
    #    'PSPC1', 'LILRA1', 'HIGD1C', 'KRT6B', 'SLC16A13', 'GDF15',
    #    'AP001421.1', 'GALNT12', 'OXNAD1', 'FAM111B', 'SPATA8', 'GCLM',
    #    'LY6G6F', 'RPL26L1', 'FMR1NB', 'TUBAL3', 'ARID4B', 'DNAAF3',
    #    'TMEM241', 'CCL4L1', 'HSDL2', 'OR51J1', 'DERA', 'CLGN', 'GLB1L3',
    #    'LRRC57', 'SGCZ', 'ZNF737', 'SVOPL', 'GAN', 'MPST', 'PIK3AP1',
    #    'C14ORF64', 'SIPA1L2', 'FXYD4', 'CEACAM21', 'CASC5', 'KRTAP10-1',
    #    'RPL41', 'AL121963.1', 'CDSN', 'ENTPD4', 'PHF3', 'GPR15',
    #    'SERINC1', 'CDRT4', 'TMEM132C', 'HBCBP', 'C5ORF17', 'ACAT1',
    #    'PPP1R37', 'ESX1', 'ISYNA1', 'KRT84', 'C1QL4', 'TACR3', 'C1ORF112',
    #    'TVP23C', 'C10ORF32', 'TSPY6P', 'PDZD9', 'OR51E1', 'SLC22A7',
    #    'AC109583.1', 'ST6GALNAC5', 'IGHV3-74', 'OR14I1', 'BICD1', 'ABCD4',
    #    'ITCH', 'C11ORF1', 'ARSB', 'SIX4', 'PTGES', 'AC040160.1',
    #    'C9ORF171', 'SCML4', 'ESPNL', 'WDR41', 'ZFP42', 'MGAT4A', 'ALAS2',
    #    'NIT1', 'CNKSR2', 'IL7R', 'RP5-1021I20.4', 'AHNAK', 'ECEL1',
    #    'ZNF77'],
    # 'ADCP_CRIPSRa_Up100':["GFI1", "SMAGP", "MUC21", "ST6GALNAC1", "ITGB2", "OSR2", "MUC1", "CD1C",
    #     "GAL3ST4", "FUT6", "ST3GAL1", "MS4A1", "LRRC15", "TLE3", "PRDM1", "SPN",
    #     "MUC12", "PTPRC", "HDAC9", "NFIA", "ELOVL6", "GFI1B", "POU2F2", "IRX5",
    #     "MS4A7", "MS4A14", "C5AR1", "CD44", "IQGAP2", "CBFA2T3", "JMJD1C", "CD38",
    #     "ALCAM", "PPAP2B", "FCGR2B", "PODXL", "HMHA1", "HIC1", "BCL9L", "MAML2",
    #     "SPIB", "CLIC4", "SLA", "PIK3AP1", "FAR1", "MAML1", "POU2AF1", "ZEB2",
    #     "SASH3", "SLC9A3R1", "DOCK11", "HES7", "BCOR", "PTPN6", "TSPAN15", "GAL3ST2",
    #     "RAC2", "FOXO4", "AXL", "LIMK2", "SLC39A13", "CADM1", "CAPN6", "MAML3",
    #     "CLDN18", "ST3GAL2", "VSIG8", "IKZF3", "ELOVL1", "CEBPE", "SYK", "FMNL3",
    #     "MAP3K3", "ICAM1", "EZR", "FCGR1B", "LCK", "FMNL1", "GPR114", "GRHL1",
    #     "PNMA5", "ZC3HAV1", "VCAM1", "ZNF746", "BTK", "FAM81A", "SLA2", "ZBTB7A",
    #     "ZNF683", "CD79B", "CIITA", "PPP3CA", "ZBTB7B", "BCL6", "MSN", "IER5L",
    #     "TCEB1", "MAK", "MAP3K10", "ZNF311", "MOB3A", "FCRL3", "ZDBF2", "LCA5L",
    #     "CIT", "GCNT1", "TMEM119", "SIX4", "SUV420H1", "GYPA", "ISL2", "ZFX",
    #     "TP73", "SYT1", "MEX3B", "RNF122", "LPIN2", "MED13", "DBR1", "MUC22",
    #     "SIAH2", "PWWP2B", "POGLUT1", "CELF1", "DBN1", "TTC39C", "MYB", "NRG4",
    #     "FAM49B", "PLA2G6", "MSS51", "CRIP2", "SORD", "DGKB", "VSIG1", "FRMD3",
    #     "CD47", "C15ORF59", "S100A11", "STK10", "CSNK1G3", "PRAP1", "ADAD2", "MS4A8",
    #     "TRIM13", "GDPGP1", "LYL1", "LAG3", "NOMO2", "NOMO3", "PNMAL1", "DOCK10",
    #     "KCNS2", "NOMO1", "ZMYND8", "KLHL36", "VPS39", "SLC30A7", "KCNJ6", "SPTBN1",
    #     "ZNF680", "UNC13D", "QSER1", "CHAF1A", "YY1", "IGLL1", "CRYGD", "PRSS36",
    #     "TIMM23B", "RORC", "MYH10", "PIGR", "NANOS3", "LHX9", "LAPTM5", "CIB1",
    #     "LPPR1", "TIMM23", "CD4", "TMEM101", "DEFB1", "KRT23", "FAM129B", "MAG",
    #     "C20ORF196", "ARRDC3", "RAB44", "JARID2", "KRT6A", "UBE2Q2", "WNT9A",],
    # 'ADCP_CRIPSRa_Down100':["DDACH1", "CLCA1", "TLL1", "FLNA", "TMEM45B", "SGPP2", "SUGCT",
    #     "SLC38A2", "PLEKHG4B", "LIF", "TRAPPC11", "NELL2", "PCDHB3",
    #     "PCDH1", "GOLGA4", "CLCNKA", "MCHR2", "TMEM100", "TNFAIP2",
    #     "BNHIP", "PGRMC2", "MEPE", "MAGEH1", "DRD5", "SLC22A17", "GREAM1",
    #     "OREG", "PRELID1", "SPRN", "DRD4", "KCNJ1", "HBB", "TRUB1",
    #     "CDK17", "CLCNK", "MROH5", "CTNNB1", "TCEA1", "MPND", "RBM42",
    #     "SLC5A1", "THOC6", "FIS1", "ARHGAP39", "ARHGEF12", "OR2Z1",
    #     "CLRN1", "ITLN2", "PCDHB2", "MUC15", "TMEM149", "MAML1", "ARHGEF12",
    #     "C2CD4A", "CPED1", "AVE1", "KCNT2", "DGCR2", "SGPL1", "ABRB1",
    #     "FAM211A", "CHIC2", "C3ORF49", "TMEM14", "CRML", "OR2T1",
    #     "CPED1", "ADCC1B", "OR2W1", "UBALD1", "KCNJ12-2", "AFMIRAID",
    #     "CLCNKA", "DRD1", "NELL2", "MROH5", "FAM215B", "C3ORF49",
    #     "C3ORF49-SC", "AL024168.1", "CLCNKA", "LINC00306", "RP11-102895.1",
    #     "C16ORF92", "KCNJ1", "CLCNKA", "OR2B1A", "LILRB2", "SPRN",
    #     "CCDC33", "RBM42", "TAS2R14", "MEPE", "TMEM74", "NEFL", "RABGGTA",
    #     "NELL2", "NIEZ1"]
}

# gene_modules = {
#     'Macrophage Derived Foam Cell Differentiation': ['CSF1', 'ITGB3', 'NR1H3', 'PPARG', 'CD36', 'LIPA'],
#     # 'Lipid': ['ABCA1', 'ABCG1', 'CD36', 'FABP4', 'FABP5', 'PLIN2', 'OLR1'],
#     'Inflammatory': ['IL1B', 'NLRP3', 'G0S2', 'SOD2', 'CEBPB', 'NFKBIA', 'INHBA', 'TNF', 'TREM1', 'CCL2', 'CCL20', 'CCL7', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL8'],
#     'IL-17 signaling pathway': ['FOS', 'FOSB', 'CXCL2', 'CXCL3', 'IL1B', 'CXCL8', 'JUN', 'NFKBIA', 'PTGS2', 'CCL20', 'TNFAIP3'],
#     'Apoptosis': ['IER3', 'EGR1', 'BTG1', 'BCL2A1', 'PTGER2', 'PTGS2', 'MCL1', 'CDKN1A', 'G0S2'],
#     'Apoptosis 2': ['GALM', 'ENO1', 'FBP1', 'ALDOA', 'GAPDH', 'PGAM1', 'PGK1', 'PKM', 'TPI1'],
#     'Oxidative phosphorylation': ['TCIRG1', 'UQCR11', 'COX5B', 'COX6A1', 'COX6C', 'COX8A', 'UQCRQ', 'NDUFAB1', 'NDUFB2', 'NDUFB7', 'NDUFS5', 'ATP5F1B', 
#                                 'ATP5MC1', 'ATP5MC3', 'ATP5ME', 'ATP6V0C', 'ATP6V0B', 'ATP6V1F', 'ATP6V1G1', 'ATP5MF'],
#     'Glycolysis / Gluconeogenesis': ['GALM', 'ENO1', 'FBP1', 'ALDOA', 'GAPDH', 'PGAM1', 'PGK1', 'PKM', 'TPI1'],
#     'NF-kappa B signaling pathway': ['TNFSF13B', 'MALT1', 'GADD45A', 'BIRC3', 'NFKB1', 'RELB', 'CCL19', 'TRAF1', 'UBE2I', 'CFLAR', 'CD40'],
#     'PPAR signaling pathway': ['PLIN2', 'FABP4', 'FABP5', 'ACSL3', 'GK', 'OLR1', 'ANGPTL4', 'SCD', 'CD36'],
#     'Phagosome': ['TCIRG1', 'TUBA1B', 'CTSL', 'SEC61G', 'RAB7B', 'ITGB1', 'ITGB2', 'ITGB5', 'LAMP1', 'MSR1', 'ATP6V0C', 'ATP6V0B',
#                  'TFRC', 'CALR', 'TUBB6', 'TUBA1C', 'MARCO', 'ATP6V1F', 'CD36', 'ATP6V1G1'],
#     'antigen processing and presentation': ['AP1B1','CAPZA2','CAPZB','CD68','CD74','CLTA','CLTC','CTSD','CTSH',
#             'FCGR1A','FCGR2B','HLA-DRB5','LGMN','PSAP','VAMP8','RAB32','TREM2'],
#     'Ferroptosis': ['FTH1','FTL','GPX4','SLC40A1','HMOX1'],
#     'Detoxification': ['ALOX5AP','APOE','GPX3','MGST2','MGST3','PRDX1','AKR1A1','SESN1','PRXL2A'],
#     'Lipid Storage': ['ITGB3', 'NR1H3', 'PPARG', 'TTC39B'],
#     'Lipid peroxidation': ['ACSL4', 'LPCAT3', 'FADS1', 'ELOVL5', 'ALOX15', 'PEBP1', 'POR', 'NOX4'],
#     'Antioxidant defense / lipid peroxide removal': ['GPX4', 'SLC7A11', 'SLC3A2', 'GSH', 'FSP1', 'GCH1'],
#     'Iron metabolism': ['TFRC', 'FTH1', 'FTL', 'FPN1', 'NCOA4', 'HMOX1'],
#     'Lysosomal lipid handling': ['LIPA', 'NPC1', 'CTSD', 'CTSK'],
#     'ECM remodeling': ['MMP9', 'MMP14', 'TIMP3', 'COL6A1'], # ITGBL1, PAPLN, GPC4, SEMA3C
#     'Cell Adhesion & Migration': ['ITGA3', 'PALLD', 'MYOF', 'MYO1E', 'ASAP1', 'DOCK3'],
#     'Membrane remodeling': ['DCSTAMP', 'MYOF', 'ANO5', 'TM4SF19', 'TSPAN19'],
#     'Cholesterol Efflux': ['ABCA7', 'ABCA5', 'APOA2', 'ABCA8', 'APOA5', 'APOA4', 'NPC1', 'STX12', 'APOC1', 'NPC2',  'APOC3', 'APOC2', 'SOAT1', 'ABCG1', 'APOE', 'SCARB1' ],
#     'Endocytosis': ['DNM3', 'MYO1E', 'CTTN', 'MYO6', 'MICALL1', 'ENTHD1', 'APOE', 'CD36', 'LRP12', 'ATP9A', 'CORO1C'],
#     'Response to Cytokine': ['CIITA', 'PID1', 'IL1R1', 'FLT3', 'AFF3'],
#     'Phagocytosis': ['YES1', 'LYN', 'SRC', 'LIMK1', 'PRKCD', 'SYK', 'PRKCE', 'PLA2G6', 'PLD2', 'VAV1', 'HCK', 'VAV2', 'PTK2', 'VAV3', 'FGR', 'FCGR2B', 'PAK1', 'FYN', 'MYO1G'],
#     'Inflammatory Response': ['CXCL8', 'SLC11A1', 'C5AR2', 'WNT5A', 'C5AR1', 'NLRC4', 'CXCL3', 'CXCL2', 
#                               'THBS1', 'NFKB1', 'IL6', 'OLR1', 'CD44'],
#     'Cytokine Production': ['APP', 'SLC11A1', 'WNT5A', 'LAPTM5', 'EIF2AK3', 'HIF1A', 'MALT1', 'EREG'], 
#     'Inflammatory chemotaxis': ['KLRK1', 'CCL7', 'XCL1', 'CCL5', 'CCL4', 'CCL3'],
#     'Immune Response': ['APOBEC3A', 'MEFV', 'NLRP12', 'FCN1', 'GBP5', 'CD300E', 'LILRA1', 'LILRA5', 'PTGER2'],
#     'Migration': ['VCAN', 'VCAN-AS1', 'CEACAM3', 'S100A4', 'ADGRE3', 'SPOCK1'],
#     'Angiogenesis': ['BTG1','CXCL8','FLT1','CEMIP2', 'WNT5A', 'HIF1A', 'THBS1', 'HIPK2', 'VEGFA', 
#                      'IL6', 'RGCC', 'AGO2', 'HMOX1', 'CTNNB1', 'GLUL'],
#     'Regulation of SMC proliferation': ['IL6', 'NR4A3', 'HMOX1', 'CTNNB1', 'IL6R', 'THBS1', 'EREG',
#                             'IL6', 'RGCC', 'CLEC7A', 'PDE4B', 'HMOX1', 'CD226', 'IL6R'],

#     'Proliferating': ['MKI67', 'TOP2A', 'TUBB', 'SMC2'], 
#     'MHC Class I': ['PDIA3', 'ACE', 'SAR1B', 'ERAP1', 'ERAP2', 'HFE', 'IDE', 'TAP2', 'HLA-A', 'TAP1', 'TAPBPL', 'IFI30', 'TAPBP', 'MFSD6', 'CLEC4A', 'B2M', 'CALR', 'FCER1G'],
#     'MHC Class II': ['HLA-DRB5', 'FCER1A', 'HLA-DQA1', 'HLA-DQB2', 'GAPT'],
#     'Membrane Lipid Catabolic Process': ['FUCA1', 'SGPL1', 'SMPD2', 'MGST2', 'NAGA', 'ENPP2', 'PPT1', 'SMPDL3B'],
#     # 'T-helper 1 Cell Cytokine': ['IL1R1', 'IL18R1'],
#     # 'T Cell Activation': ['JAML', 'IRF4', 'RHOH', 'CD1C'],
# }


def enrich_module(adata, gene_sets):
    import omicverse as ov
    for k,v in gene_sets.items():
        # print(k)
        ov.single.geneset_aucell(adata,
                                    geneset_name=k,
                                    geneset=v) #pathway_dict[geneset_name])

    return adata


def plot_enrich_module(adata, gene_sets):
    sc.pl.embedding(adata,
                basis='umap',
          color=["{}_aucell".format(i) for i in gene_sets.keys()])


def plot_radar_module(adata, columns='cell_type', cols=None, save=None):
    from scipy.stats import zscore
    from scalex.plot import plot_radar

    if cols is None:
        cols = [i for i in adata.obs.columns if i.endswith('aucell')]
    else:
        cols = [i+'_aucell' for i in cols if not i.endswith('aucell')]
    avg_score = adata.obs.groupby(columns)[cols].mean()
    avg_score = zscore(avg_score)
    scaled = (avg_score - avg_score.min(axis=0)) / (avg_score.max(axis=0) - avg_score.min(axis=0))

    plot_radar(scaled)
    return scaled


def enrich_analysis(gene_names, organism='hsapiens', gene_sets='GO_Biological_Process_2023', cutoff=0.05, **kwargs): # gene_sets="GO_Biological_Process_2021"
    """
    Perform KEGG pathway analysis and plot the results as a clustermap.

    Parameters:
    - gene_names: A dictionary with group labels as keys and lists of gene names as values.
    - gene_sets: The gene set database to use for enrichment analysis (default is 'KEGG_2021_Human'). 'GO_Biological_Process_2021', could find in gp.get_library_name()
    - organism: Organism for KEGG analysis (default is 'hsapiens
    - top_terms: Number of top terms to consider for the clustermap.
    """
    import gseapy as gp
    from gseapy import Msigdb
    msig = Msigdb()
    if isinstance(gene_names, pd.DataFrame):
         gene_names = gene_names.to_dict(orient='list')
    if gene_sets in msig.list_category(): 
        # ['c1.all', 'c2.all', 'c2.cgp', 'c2.cp.biocarta', 'c2.cp.kegg_legacy', 'c2.cp.kegg_medicus', 'c2.cp.pid', 'c2.cp.reactome', 'c2.cp', 'c2.cp.wikipathways', 'c3.all', 'c3.mir.mir_legacy', 'c3.mir.mirdb', 'c3.mir', 'c3.tft.gtrd', 'c3.tft.tft_legacy', 'c3.tft', 
        # 'c4.3ca', 'c4.all', 'c4.cgn', 'c4.cm', 'c5.all', 'c5.go.bp', 'c5.go.cc', 'c5.go.mf', 'c5.go', 'c5.hpo', 'c6.all', 'c7.all', 'c7.immunesigdb', 'c7.vax', 'c8.all', 'h.all', 'msigdb']
        gene_sets = msig.get_gmt(category = gene_sets, dbver='2024.1.Hs')
         
    results = pd.DataFrame()
    for group, genes in gene_names.items():
        # print(group, genes)
        genes = list(genes)
        enr = gp.enrichr(genes, gene_sets=gene_sets, cutoff=cutoff).results
        enr['cell_type'] = group  # Add the group label to the results
        results = pd.concat([results, enr])

    results_filtered = results[results['Adjusted P-value'] < cutoff]
    # results_pivot = results_filtered.pivot_table(index='Term', columns='cell_type', values='Adjusted P-value', aggfunc='min')
    # results_pivot = results_pivot.sort_values(by=results_pivot.columns.tolist(), ascending=True)

    # return results_pivot, results_filtered
    return results_filtered


def annotate(
    adata, 
    cell_type='leiden',
    color = ['cell_type', 'leiden', 'tissue', 'donor'],
    cell_type_markers='human', #None, 
    show_markers=False,
    gene_sets='GO_Biological_Process_2023',
    additional={},
    go=True,
    out_dir = None, 
    cutoff = 0.05,
    processed=False,
    top_n=300,
    filter_pseudo=True,
):
    color = [i for i in color if i in adata.obs.columns]
    color = color + [cell_type] if cell_type not in color else color
    sc.pl.umap(adata, color=color, legend_loc='on data', legend_fontsize=10)

    var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
    if cell_type_markers is not None:
        if isinstance(cell_type_markers, str):
            if cell_type_markers == 'human':
                cell_type_markers = cell_type_markers_human
            elif cell_type_markers == 'mouse':
                cell_type_markers = cell_type_markers_mouse
                filter_pseudo = False
        cell_type_markers_ = {k: [i for i in v if i in var_names] for k,v in cell_type_markers.items() }
        sc.pl.dotplot(adata, cell_type_markers_, groupby=cell_type, standard_scale='var', cmap='coolwarm')
        # sc.pl.heatmap(adata, cell_type_markers_, groupby=cell_type,  show_gene_labels=True, vmax=6)
    if 'rank_genes_groups' not in adata.uns:
        if not processed:
            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(adata, groupby=cell_type, method='t-test')
    sc.pl.rank_genes_groups_dotplot(adata, groupby=cell_type, n_genes=10)

    marker_genes = get_markers(adata, groupby=cell_type, processed=processed, top_n=top_n, filter_pseudo=filter_pseudo)
        
    try:
        go_df = enrich_and_plot(marker_genes, gene_sets=gene_sets, cutoff=cutoff, out_dir=out_dir)
        return go_df
    except Exception as e:
        print(e)
        return None


def parse_go_file(filepath):
    go_dict = {}
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines
            parts = line.strip().split("\t")
            key = parts[0].strip()
            genes = [g.strip() for g in parts[1:] if g.strip()]
            go_dict[key] = genes
    return go_dict



def check_is_numeric(df, col):
    col_values = df[col].astype(str)
    is_numeric = pd.to_numeric(col_values, errors='coerce').notna().all()
    return is_numeric

def enrich_and_plot(gene_names, organism='hsapiens', gene_sets='GO_Biological_Process_2023', cutoff=0.05, add='', out_dir=None, **kwargs):
    go_results = enrich_analysis(gene_names, organism=organism, gene_sets=gene_sets, cutoff=cutoff, **kwargs)
    if add:
        go_results['cell_type'] = add + go_results['cell_type'].astype(str)
    if check_is_numeric(go_results, 'cell_type'):
        go_results['cell_type'] = 'cluster_'+go_results['cell_type'].astype(str)

    n = go_results['cell_type'].nunique()
    ax = dotplot(go_results,
            column="Adjusted P-value",
            x='cell_type', # set x axis, so you could do a multi-sample/library comparsion
            # size=10,
            top_term=10,
            figsize=(0.7*n, 2*n),
            title = f"GO_BP",  
            xticklabels_rot=45, # rotate xtick labels
            show_ring=False, # set to False to revmove outer ring
            marker='o',
            cutoff=cutoff,
            cmap='viridis'
            )
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        go_results = go_results.sort_values('Adjusted P-value', ascending=True).groupby('cell_type').head(20)
        go_results[['Gene_set','Term','Overlap', 'Adjusted P-value', 'Genes', 'cell_type']].to_csv(os.path.join(out_dir, f'go_results.csv'))
        plt.savefig(os.path.join(out_dir, f'go_results.pdf'))
    plt.show()

    return go_results

def plot_go_enrich(go_results, add='', out_dir=None, cutoff=0.05, top_term=10, figsize=None, **kwargs):
    if add:
        go_results['cell_type'] = add + go_results['cell_type'].astype(str)
    if check_is_numeric(go_results, 'cell_type'):
        go_results['cell_type'] = 'cluster_'+go_results['cell_type'].astype(str)

    n = go_results['cell_type'].nunique()
    if figsize is None:
        figsize=(0.7*n, 2*n)
    ax = dotplot(go_results,
            column="Adjusted P-value",
            x='cell_type', # set x axis, so you could do a multi-sample/library comparsion
            top_term=top_term,
            figsize=figsize,
            title = f"GO_BP",  
            xticklabels_rot=45, # rotate xtick labels
            show_ring=False, # set to False to revmove outer ring
            size=10,
            marker='o',
            cutoff=cutoff,
            cmap='viridis',
            **kwargs
            )
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        go_results = go_results.sort_values('Adjusted P-value', ascending=True).groupby('cell_type').head(20)
        go_results[['Gene_set','Term','Overlap', 'Adjusted P-value', 'Genes', 'cell_type']].to_csv(os.path.join(out_dir, f'go_results.csv'))
        plt.savefig(os.path.join(out_dir, f'go_results.pdf'))
    plt.show()

    return go_results


def get_markers(
        adata, 
        groupby='cell_type',
        pval_cutoff=0.01, 
        logfc_cutoff=1.5,  # ~1.5 in linear scale
        min_cells=10,
        top_n=300,
        processed=False,
        filter_pseudo=True,
        min_cell_per_batch=100,
    ):
    """
    Get markers filtered by both p-value and log fold change
    
    Parameters:
        logfc_cutoff: 0.58 ≈ 1.5 fold change (log2(1.5))
                     1.0 ≈ 2 fold change (log2(2))
        min_cell_per_batch: int, optional (default: 100)
            Minimum number of cells required per batch
    """
    from scalex.pp.annotation import format_rna

    adata = adata.copy()
    if filter_pseudo:
        adata = format_rna(adata)
    
    markers_dict = {}
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    clusters = adata.obs[groupby].cat.categories

    if 'rank_genes_groups' not in adata.uns:
        if not processed:
            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(adata, groupby=groupby, method='t-test')
    
    for cluster in clusters:
        # Get all results for this cluster
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        
        # Apply filters
        filtered = df[
            (df['pvals_adj'] < pval_cutoff) & 
            (df['logfoldchanges'] > logfc_cutoff)
        ].copy()
        
        markers_dict[cluster] = filtered.sort_values('scores', ascending=False).head(top_n)['names'].values
    
    return markers_dict


def flatten_dict(markers):
    flatten_markers = np.unique([item for sublist in markers.values() for item in sublist])
    return flatten_markers

def flatten_list(lists):
    return [item for sublist in lists for item in sublist]


def filter_marker_dict(markers, var_names):
    marker_dict = {}
    for cluster, genes in markers.items():
        marker_dict[cluster] = [i for i in genes if i in var_names]
    return marker_dict

def rename_marker_dict(markers, rename_dict):
    """
    Rename dictionary keys and merge values if multiple keys map to the same new key.
    
    Parameters:
    -----------
    markers : dict
        Dictionary mapping original keys to lists of values
    rename_dict : dict
        Dictionary mapping original keys to new keys
        
    Returns:
    --------
    dict
        Dictionary with renamed keys and merged values
    """
    marker_dict = {}
    for cluster, genes in markers.items():
        if cluster not in rename_dict:
            marker_dict[cluster] = genes
            continue
        new_key = rename_dict[cluster]
        if new_key in marker_dict:
            # If key exists, extend the list with new values
            marker_dict[new_key].extend(genes)
        else:
            # If key doesn't exist, create new list
            marker_dict[new_key] = genes.copy()
    
    # Remove duplicates while preserving order
    for key in marker_dict:
        marker_dict[key] = list(dict.fromkeys(marker_dict[key]))
        
    return marker_dict


def cluster_program(adata_avg, n_clusters=None, method='kmeans'):
    if n_clusters is None:
        n_clusters = adata_avg.shape[0] #+ 2

    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import dendrogram, linkage
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        adata_avg.var['cluster'] = np.array(kmeans.fit_predict(adata_avg.X.T)).astype(str)
    elif method == 'hclust':
        linkage_matrix = linkage(adata_avg.X.T, method='ward')
        cluster_labels = linkage_matrix[:, 2]  # Use the third column for cluster labels
        adata_avg.var['cluster'] = cluster_labels.astype(str)

    gene_cluster_dict = adata_avg.var.groupby('cluster').groups
    gene_cluster_dict = {k: v.tolist() for k, v in gene_cluster_dict.items()}

    return gene_cluster_dict


def find_gene_program(adata, groupby='cell_type', processed=False, n_clusters=None, top_n=300, filter_pseudo=True, **kwargs):
    """
    Find gene program for each cell type
    """
    adata = adata.copy()
    
    adata_avg = aggregate_data(adata, groupby=groupby, processed=processed, scale=True)

    markers = get_markers(adata, groupby=groupby, processed=processed, top_n=top_n, filter_pseudo=filter_pseudo, **kwargs)
    for cluster, genes in markers.items():
        print(cluster, len(genes))

    marker_list = flatten_dict(markers)

    # sc.pp.scale(adata_avg, zero_center=True)
    adata_avg_ = adata_avg[:, marker_list].copy()

    gene_cluster_dict = cluster_program(adata_avg_, n_clusters=n_clusters)

    return gene_cluster_dict, adata_avg


def find_peak_program(adata, groupby='cell_type', processed=False, n_clusters=None, top_n=-1, pval_cutoff=0.05, logfc_cutoff=1., filter_pseudo=False, **kwargs):
    """
    Find peak program for each cell type
    """
    return find_gene_program(adata, groupby=groupby, processed=processed, top_n=top_n, filter_pseudo=filter_pseudo, pval_cutoff=pval_cutoff, logfc_cutoff=logfc_cutoff, **kwargs)


def _process_group(args):
    """Helper function for multiprocessing"""
    adata_, groupby, set_type, top_n, filter_pseudo, kwargs = args
    if set_type == 'gene':
        filter_pseudo = True
    elif set_type == 'peak':
        filter_pseudo = False
        if not 'pval_cutoff' in kwargs:
            kwargs['pval_cutoff'] = 0.05
        if not 'logfc_cutoff' in kwargs:
            kwargs['logfc_cutoff'] = 1.
        
    # Filter groups with less than min_samples
    group_counts = adata_.obs[groupby].value_counts()
    valid_groups = group_counts[group_counts >= 2].index
    if len(valid_groups) < 2:
        print(f"Skipping as it has less than 2 groups with 2 or more samples")
        return None
        
    adata_ = adata_[adata_.obs[groupby].isin(valid_groups)].copy()
    markers = get_markers(adata_, groupby=groupby, top_n=top_n, filter_pseudo=filter_pseudo, **kwargs)
    # print(len(flatten_dict(markers)))
    return flatten_dict(markers)


def find_consensus_program(adata, groupby='cell_type', across=None, set_type='gene', processed=False, top_n=-1, occurance=None, min_samples=2, n_jobs=None, n_clusters=None,**kwargs):
    """
    Find consensus program for each cell type
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    groupby : str, optional (default: 'cell_type')
        Column name in adata.obs to group by
    across : str, optional (default: None)
        Column name in adata.obs to split data across
    set_type : str, optional (default: 'gene')
        Type of features to analyze ('gene' or 'peak')
    processed : bool, optional (default: False)
        Whether the data is already processed
    top_n : int, optional (default: -1)
        Number of top markers to select per group
    occurance : int, optional (default: None)
        Minimum number of occurrences across groups
    min_samples : int, optional (default: 2)
        Minimum number of samples required per group
    n_jobs : int, optional (default: None)
        Number of jobs to run in parallel. If None, uses all available cores.
    """
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    if n_clusters is None:
        n_clusters = len(adata.obs[groupby].cat.categories)
    occurance = occurance or max(2, len(np.unique(adata.obs[across])) // 2)
    filter_pseudo = True if set_type == 'gene' else False

    if across is not None:
        # Prepare arguments for parallel processing
        args_list = []
        adata_avg_list = []
        for c in np.unique(adata.obs[across]):
            adata_ = adata[adata.obs[across] == c].copy()
            adata_ = adata_[adata_.obs.dropna(subset=[groupby]).index].copy()
            args_list.append((adata_, groupby, set_type, top_n, filter_pseudo, kwargs))
            adata_avg_c = aggregate_data(adata_, groupby=groupby, processed=processed, scale=True)
            adata_avg_c.obs[across] = c
            adata_avg_list.append(adata_avg_c)

        adata_avg = concat(adata_avg_list)

        # Process groups in parallel or sequentially based on n_jobs
        if n_jobs == 1:
            results = [_process_group(args) for args in args_list]
        else:
            from multiprocessing import Pool, cpu_count
            if n_jobs is None:
                n_jobs = min(cpu_count(), 32)
            with Pool(n_jobs) as pool:
                results = pool.map(_process_group, args_list)
            
        # Filter out None results and combine markers
        markers_list = [r for r in results if r is not None]
        if not markers_list:
            raise ValueError("No valid groups found with sufficient samples")
            
        markers_list = np.concatenate(markers_list)
        gene_counts = Counter(markers_list)
        markers_list = np.array([gene for gene, count in gene_counts.items() if count >= occurance])
        print('There are {} {set_type}s with at least {} occurrences'.format(len(markers_list), occurance, set_type=set_type))

        # adata_avg_list = []
        # # gene_cluster_dict = {}
        # for c in np.unique(adata.obs[across]):
        #     adata_ = adata[adata.obs[across] == c].copy()
        #     adata_avg = aggregate_data(adata_, groupby=groupby, processed=processed, scale=True)
        #     adata_avg_list.append(adata_avg)
        #     adata_avg.obs[across] = c

        # adata_avg = concat(adata_avg_list)

    # adata_avg = aggregate_data(adata, groupby=groupby, processed=processed, scale=True)
    adata_avg.obs[groupby+'_'+across] = adata_avg.obs[groupby].astype(str) + '_' + adata_avg.obs[across].astype(str)
    # groupby = groupby+'_'+across


    # adata_avg = aggregate_data(adata, groupby=groupby, processed=processed, scale=True)
    adata_avg_ = adata_avg[:, markers_list].copy()

    gene_cluster_dict = cluster_program(adata_avg_, n_clusters=n_clusters)

    return gene_cluster_dict, adata_avg




def find_go_term_gene(df, term):
    """
    df: df = pd.read_csv(go_results, index_col=0)
    term: either Term full name or Go number: GO:xxxx
    """
    if term.startswith('GO'):
        df['GO'] = df['Term'].str.split('(').str[1].str.replace(')', '')
        select = df[df['GO'] == term].copy()
    else:
        select = df[df['Term'] == term].copy()
    gene_set = set(gene for sublist in select['Genes'].str.split(';') for gene in sublist)
    # gene_set = set(select['Genes'].str.split(';'))
    # print(select['Term'].head(1).values[0])
    # print('\n'.join(gene_set))
    return gene_set

def format_dict_of_list(d, out='table'):
    if out == 'matrix':
        data = []
        for k, lt in d.items():
            for v in lt:
                data.append({'Gene': v, 'Pathway': k})

        # Step 2: Create a DataFrame from the list
        df = pd.DataFrame(data)

        # Step 3: Use crosstab to pivot the DataFrame
        df = pd.crosstab(df['Gene'], df['Pathway'])
    elif out == 'table':
        df = pd.DataFrame.from_dict(d, orient='index').transpose()

    return df


def parse_go_results(df, cell_type='cell_type', top=20, out='table', tag='', dataset=''):
    """
    Return:
        a term gene dataframe: each column is a term
        a term cluster dataframe: each column is a term
    """
    term_genes = {}
    term_clusters = {}
    for c in np.unique(df[cell_type]):
        terms = df[df[cell_type]==c]['Term'].values
        for term in terms[:top]:
            if term not in term_clusters:
                term_clusters[term] = []
            
            term_clusters[term].append(c)

            if term not in term_genes:
                term_genes[term] = find_go_term_gene(df, term)

    tag = tag + ':' if tag else ''

    if out == 'dict':
        return term_genes, term_clusters
    else:
        term_genes = format_dict_of_list(term_genes, out=out)
        index = [(k, dataset, tag+';'.join(v)) for k, v in term_clusters.items()]
        term_genes.columns = pd.MultiIndex.from_tuples(index, names=['Pathway', 'Dataset', 'Cluster'])
        return term_genes


def merge_all_go_results(path, datasets=None, top=20, out_dir=None, add_ref=False, union=True, reference='GO_Biological_Process_2023', organism='human'):
    """
    The go results should organized by path/datasets/go_results.csv
    Args: 
        path is the input to store all the go results
        datasets are selected to merge
    """
    df_list = []
    if datasets is None:
        datasets = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
    for dataset in datasets:
        path2 = os.path.join(path, dataset)
        for filename in os.listdir(path2):
            if 'go_results' in filename:
                name = filename.replace('.csv', '')
                path3 = os.path.join(path2, filename)
                df = pd.read_csv(path3, index_col=0)
                term_genes = parse_go_results(df, dataset=dataset, tag=name, top=top)
                df_list.append(term_genes)
    concat_df = pd.concat(df_list, axis=1)

    if add_ref and not union: 
        go_ref = gp.get_library(name=reference, organism=organism)
        go_ref = format_dict_of_list(go_ref)
        pathways = [i for i in concat_df.columns.get_level_values('Pathway').unique() if i in go_ref.columns]
        go_ref = go_ref.loc[:, pathways]
        index_tuples = [ (i, 'GO_Biological_Process_2023', 'reference') for i in go_ref.columns ] 
        go_ref.columns = pd.MultiIndex.from_tuples(index_tuples, names=['Pathway', 'Dataset', 'Cluster'])
        concat_df = pd.concat([concat_df, go_ref], axis=1)

    concat_df = concat_df.sort_index(axis=1, level='Pathway')

    if union:
        concat_df = concat_df.groupby(level=["Pathway"], axis=1)
        concat_dict = {name:  [i for i in set(group.values.flatten()) if pd.notnull(i)] for name, group in concat_df}
        concat_df = pd.DataFrame.from_dict(concat_dict, orient='index').transpose()

    if out_dir is not None:
        dirname = os.path.dirname(out_dir)
        os.makedirs(dirname, exist_ok=True)

        if not union:
            if not out_dir.endswith('xlsx'):
                out_dir = out_dir + '.xlsx'
            with pd.ExcelWriter(out_dir, engine='openpyxl') as writer:
                concat_df.to_excel(writer, sheet_name='Sheet1')
        else:
            concat_df.to_csv(out_dir, index=False)
    return concat_df


from sklearn.metrics import auc

def aucell_scores(expr: pd.DataFrame, gene_sets: dict, top=0.05):
    """
    AUCell implementation in Python.
    
    expr: cells x genes expression matrix (DataFrame).
    gene_sets: dict of {program_name: [genes]}.
    top: fraction of top-ranked genes to consider.
    """
    n_top = int(expr.shape[1] * top)
    
    scores = {}
    for prog, genes in gene_sets.items():
        prog_scores = []
        # intersect with available genes
        valid_genes = [g for g in genes if g in expr.columns]
        if len(valid_genes) == 0:
            prog_scores = [np.nan] * expr.shape[0]
        else:
            for _, row in expr.iterrows():
                # rank genes in descending order
                ranked_genes = row.sort_values(ascending=False).index[:n_top]
                hits = [1 if g in ranked_genes else 0 for g in valid_genes]
                # compute cumulative sum (enrichment curve)
                x = np.linspace(0, 1, len(hits))
                y = np.cumsum(hits) / max(1, sum(hits))
                score = auc(x, y)
                prog_scores.append(score)
        scores[prog] = prog_scores
    
    return pd.DataFrame(scores, index=expr.index)

import scanpy as sc

def get_rank_dict(adata, cell_type='cell_type', n_top=100, to_dict=True):
    if 'rank_genes_groups' not in adata.uns:
        sc.tl.rank_genes_groups(adata, cell_type)
    df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(n_top)
    if to_dict:
        return df.to_dict(orient='list')
    return df

def reorder(adata, groupby='cell_type', order=None):
    # if order is None:
    #     order = adata.obs[groupby].value_counts().index.tolist()
    adata.obs[groupby] = adata.obs[groupby].astype(str).astype('category').cat.reorder_categories(order)
    return adata