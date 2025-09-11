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
                  'MIR155HG', 'MYO1E', 'NUPR1', 'PRKCH', 'RAI14', 'RALA', 'RCAN3', 'SCD', 'SH3D21', 'SHC3', 'SLC22A15', 
                  'SLC26A11', 'SLC43A3', 'SLC5A4-AS1', 'SPP1', 'TM4SF19', 'TM4SF19-AS1', 'WHRN', 'ZBTB7C', 'ZFYVE28', 'ZMIZ1-AS1'],
    'Foam Shared': ['ACE', 'BBS5', 'CBLB', 'CLIP4', 'DCSTAMP', 'GPNMB', 'KCNMA1', 'MITF', 'PLA2G7', 'PLPP3', 'PPARG', 'SAMD4A', 'SLC38A6', 'SNTB1', 'TPRG1'], 
    'Macrophage Derived Foam Cell Differentiation': ['CSF1', 'ITGB3', 'NR1H3', 'PPARG', 'CD36'],
    'Lipid Storage': ['ITGB3', 'NR1H3', 'PPARG', 'TTC39B'],
    'Cholesterol Efflux': ['ABCA7', 'ABCA5', 'APOA2', 'ABCA8', 'APOA5', 'APOA4', 'NPC1', 'STX12', 'APOC1', 'NPC2',  'APOC3', 'APOC2', 'SOAT1', 'ABCG1', 'APOE', 'SCARB1' ],
    'Endocytosis': ['DNM3', 'MYO1E', 'CTTN', 'MYO6', 'MICALL1', 'ENTHD1', 'APOE', 'CD36', 'LRP12', 'ATP9A', 'CORO1C'],
    'Response to Cytokine': ['CIITA', 'PID1', 'IL1R1', 'FLT3', 'AFF3'],
    'Phagocytosis': ['YES1', 'LYN', 'SRC', 'LIMK1', 'PRKCD', 'SYK', 'PRKCE', 'PLA2G6', 'PLD2', 'VAV1', 'HCK', 'VAV2', 'PTK2', 'VAV3', 'FGR', 'FCGR2B', 'PAK1', 'FYN', 'MYO1G'],
    'Inflammatory Response': ['CXCL8', 'SLC11A1', 'C5AR2', 'WNT5A', 'C5AR1', 'NLRC4', 'CXCL3', 'CXCL2', 
                              'THBS1', 'NFKB1', 'IL6', 'OLR1', 'CD44'],
    'Cytokine Production': ['APP', 'SLC11A1', 'WNT5A', 'LAPTM5', 'EIF2AK3', 'HIF1A', 'MALT1', 'EREG'], 
    'Inflammatory chemotaxis': ['KLRK1', 'CCL7', 'XCL1', 'CCL5', 'CCL4', 'CCL3'],
    'Response To Type II Interferon (GO:0034341)': ['CCL26', 'CCL25', 'GBP6', 'CCL24', 'CD74', 'CCL23', 'CCL22', 'CCL21', 'CCL20', 'WNT5A', 'XCL1', 'RPL13A', 'TLR2', 'BST2', 
                                                    'IRF8', 'XCL2', 'TLR4', 'CALM1', 'SLC26A6', 'IFITM2', 'CCL3L1', 'CD40', 'CITED1', 'IFITM3', 'MEFV', 'IFITM1', 'SP100', 
                                                    'AQP4', 'CX3CL1', 'CXCL16', 'GBP1', 'CASP1', 'CIITA', 'GBP2', 'IL12RB1', 'GBP5', 'SNCA', 'GBP4', 'STAT1', 'DAPK1',
                                                    'GCH1', 'SLC11A1', 'DAPK3', 'EPRS1', 'PDE12', 'CD47', 'SHFL', 'GAPDH', 'TDGF1', 'CCL15', 'SLC22A5', 'CCL14', 'CALCOCO2', 
                                                    'CCL13', 'CCL11', 'CCL4L1', 'IL23R', 'CAMK2A', 'AIF1', 'NUB1', 'CYP27B1', 'SYNCRIP', 'CCL8', 'CCL7', 'CCL5', 'KYNU', 'UBD', 
                                                    'CCL4', 'HLA-DPA1', 'SIRPA', 'CCL3', 'CCL2', 'CCL1', 'CCL19', 'CD58', 'LGALS9', 'TRIM21', 'CCL18', 'CCL17', 'CCL16'],
    'Angiogenesis': ['BTG1','CXCL8','FLT1','CEMIP2', 'WNT5A', 'HIF1A', 'THBS1', 'HIPK2', 'VEGFA', 
                     'IL6', 'RGCC', 'AGO2', 'HMOX1', 'CTNNB1', 'GLUL'],
    'Regulation of SMC proliferation': ['IL6', 'NR4A3', 'HMOX1', 'CTNNB1', 'IL6R', 'THBS1', 'EREG',
                            'IL6', 'RGCC', 'CLEC7A', 'PDE4B', 'HMOX1', 'CD226', 'IL6R'],
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
    'Proferating': ['ACTB', 'ACTG1', 'ACTN4', 'AIF1', 'ALCAM', 'ANLN', 'ANPEP', 'ARHGAP11A', 'ARHGAP11B', 'ARHGDIA', 'ARL6IP1', 'ASPM', 'ATAD2', 'ATAD5', 'ATP5F1E', 'ATP6V0B', 
                'AURKB', 'BARD1', 'BLM', 'BRCA1', 'BRIP1', 'BSG', 'BUB1B', 'C15orf48', 'C21orf58', 'CALM1', 'CALR', 'CD52', 'CDCA2', 'CENPE', 'CENPF', 'CENPI', 'CENPK', 'CEP55', 
                'CFL1', 'CHEK2', 'CIP2A', 'CIT', 'CKAP2', 'CKAP2L', 'CLIC1', 'CLSPN', 'COX4I1', 'COX6A1', 'COX8A', 'CRIP1', 'CYTOR', 'DDX11', 'DEPDC1B', 'DIAPH3', 'DTL', 'EEF1A1', 
                'EEF1G', 'EIF3K', 'EMP3', 'ENO1', 'ESPL1', 'EZH2', 'FAM111B', 'FANCA', 'FANCD2', 'FANCI', 'FAU', 'FLNA', 'FLNB', 'FOXM1', 'GABARAP', 'GAPDH', 'GEN1', 'GPSM2', 
                'GTSE1', 'HCST', 'HLA-A', 'HLA-B', 'HMGB2', 'HMGN2', 'HSP90AB1', 'HSP90B1', 'HSPA5', 'IQGAP3', 'ITGB2', 'KCNK13', 'KIF11', 'KIF14', 'KIF15', 'KIF18A', 'KIF18B', 
                'KIF20B', 'KIF2C', 'KIF4A', 'KIFC1', 'KNL1', 'KNTC1', 'LDHA', 'LDHB', 'LGALS1', 'LIN54', 'LINC01572', 'LSM7', 'LSP1', 'LYZ', 'MELK', 'MIF', 'MKI67', 'MYL6', 'NACA', 
                'NCAPD3', 'NCAPG', 'NCAPG2', 'NCAPH', 'NCL', 'NDC80', 'NME2', 'NPC2', 'NSD2', 'NUF2', 'NUSAP1', 'OAZ1', 'ORC6', 'P4HB', 'PARPBP', 'PCBP1', 'PDIA3', 'PDIA6', 'PFN1', 
                'PGAM1', 'PKM', 'PLK4', 'PLP2', 'POLE2', 'POLQ', 'PPIA', 'PPIB', 'PRC1', 'PRDX1', 'PRKCSH', 'PRR11', 'PTMA', 'PTPN22', 'PTPN7', 'RACK1', 'RAD51AP1', 'RAN', 'REEP5', 
                'RPL10', 'RPL10A', 'RPL11', 'RPL12', 'RPL13', 'RPL13A', 'RPL14', 'RPL15', 'RPL18', 'RPL19', 'RPL22', 'RPL23', 'RPL23A', 'RPL24', 'RPL26', 'RPL27A', 'RPL28', 'RPL29', 
                'RPL3', 'RPL30', 'RPL32', 'RPL35', 'RPL35A', 'RPL36', 'RPL36A', 'RPL37', 'RPL37A', 'RPL38', 'RPL39', 'RPL4', 'RPL41', 'RPL5', 'RPL7', 'RPL7A', 'RPL8', 'RPLP0', 'RPLP1', 
                'RPLP2', 'RPS10', 'RPS11', 'RPS12', 'RPS13', 'RPS14', 'RPS15', 'RPS15A', 'RPS17', 'RPS18', 'RPS19', 'RPS2', 'RPS20', 'RPS21', 'RPS23', 'RPS24', 'RPS25', 'RPS26', 'RPS27A', 
                'RPS28', 'RPS29', 'RPS3', 'RPS4X', 'RPS5', 'RPS6', 'RPS7', 'RPS8', 'RPSA', 'RRM1', 'RRM2', 'S100A10', 'S100A4', 'SAE1', 'SERF2', 'SET', 'SGO2', 'SH3BGRL3', 'SHCBP1', 
                'SLC25A6', 'SMC2', 'SMC4', 'SNHG29', 'SNRPD1', 'SPC25', 'ST14', 'STIL', 'TACC3', 'TAGLN2', 'TMBIM6', 'TMED9', 'TMSB10', 'TMSB4X', 'TNFRSF11A', 'TOP2A', 'TPI1', 'TPX2', 
                'TREM2', 'TSPO', 'TUBA1B', 'TUBB', 'TUBB4B', 'UBA52', 'UHRF1', 'VAMP8', 'VIM', 'WDHD1', 'WDR62', 'WDR76', 'ZGRF1', 'ZNF367'],
    'MHC Class I': ['PDIA3', 'ACE', 'SAR1B', 'ERAP1', 'ERAP2', 'HFE', 'IDE', 'TAP2', 'HLA-A', 'TAP1', 'TAPBPL', 'IFI30', 'TAPBP', 'MFSD6', 'CLEC4A', 'B2M', 'CALR', 'FCER1G'],
    'MHC Class II': ['HLA-DRB5', 'FCER1A', 'HLA-DQA1', 'HLA-DQB2', 'GAPT'],
    'Membrane Lipid Catabolic Process': ['FUCA1', 'SGPL1', 'SMPD2', 'MGST2', 'NAGA', 'ENPP2', 'PPT1', 'SMPDL3B'],
    'T-helper 1 Cell Cytokine': ['IL1R1', 'IL18R1'],
    'T Cell Activation': ['JAML', 'IRF4', 'RHOH', 'CD1C'],


    # 'metabolic adaptation': ['ALDOA', 'LDHA', 'SLC2A1', 'IDH1', 'VHL'],
    #             'angiogenesis': ['VEGFA', 'TGFB1', 'ADM', 'PLOD1', 'ID2', 'SPP1'],
    #             'autophagy': ['BNIP3'],
    #             'apoptosis': ['ENO1'],
    #             'TAM chemotaxos': ['MIF', 'LGALS3', 'IL1B'],
    #             'lipid response': ['ACAT2', 'CAV1', 'NAMPT', 'VLDLR', 'DGAT1'],
    #             'immune regulation': ['CD163', 'CD163L1', 'FCGR2B', 'TGFB2'],
    #             'matrix remodeling': ['DSE', 'MMP7', 'PLAU', 'PLAUR'],
    #             'mesenchymal like': ['CD44', 'NAMPT', 'PLAUR', 'VIM'],
    #             'staining markers': ['TREM2', 'CD36', 'MRC1', 'CD9', 'MSR1']
}


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


def cluster_program(adata_avg, n_clusters=None):
    if n_clusters is None:
        n_clusters = adata_avg.shape[0] #+ 2

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    adata_avg.var['cluster'] = np.array(kmeans.fit_predict(adata_avg.X.T)).astype(str)
    # print(adata_avg_.var)

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

# Suppose adata is your single-cell dataset (normalized log1p CPM/TPM)
# expr = pd.DataFrame(
#     adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X,
#     index=adata.obs_names,
#     columns=adata.var_names
# )

# gene_sets = {
#     "anti_tumor": ["CXCL9", "CXCL10", "CD40", "STAT1"],
#     "pro_tumor": ["VEGFA", "MMP9", "ARG1"],
# }

# scores = aucell_scores(expr, gene_sets, top=0.05)

# # Add back to AnnData
# for prog in scores.columns:
#     adata.obs[f"{prog}_AUCell"] = scores[prog]