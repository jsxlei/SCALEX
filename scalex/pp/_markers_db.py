cell_type_markers_human = {
    'Macrophage': ['MSR1', 'CSF1R', 'C1QA', 'APOE', 'TREM2', 'MARCO', 'MCR1',  'CTSB', 'RBM47', 'FMN1', 'MS4A6A', 'CD68', 'CD163', 'CD206', 'CCL2', 'CCL3', 'CCL4'],
    'Foamy': ['SPP1', 'GPNMB', 'LPL', 'MGLL', 'LIPA', 'PPARG', 'APOE', 'CAPG', 'CTSD', 'LGALS3', 'LGALS1'],
    'Resident': ['SEPP1', 'SELENOP', 'FOLR2', 'F13A1', 'LYVE1'],
    'Proliferating': ['MKI67', 'TOP2A', 'TUBB', 'SMC2'],
    'Inflammatory': ['NFKBIA', 'IL1B', 'CXCL2', 'CXCL8', 'IER3', 'SOD2', ],
    'NK activating': ['FUCA1', 'ENPP2', 'TIGIT', 'CMKLR1', 'KLRK1', 'RASGRP1', 'NR1H3', 'TIMD4'],
    'MonoMac': ['FCN1', 'S100A9', 'S100A8', 'LYZ', 'S100A4'],
    'CD16 Mono': ['FCGR3A', 'PLAC8', 'CEBPD', 'CX3CR1'],
    'CD14 Mono': ['S100A9', 'S100A8', 'VCAN'],
    'neutrophils': ['FCGR3B', 'CSF3R', 'CXCR2', 'IFITM2', 'BASP1', 'GOS2'],
    'DC': ['CLEC9A', 'XCR1', 'CD1C', 'CD1A', 'LILRA4'],
    'cDC1': ['CLEC9A', 'IRF8', 'SNX3', 'XCR1'],
    'cDC2': ['FCER1A', 'CD1C', 'CD1E', 'CLEC10A'],
    'migratoryDC': ['BIRC3', 'CCR7', 'LAMP3'],
    'follicular DC': ['FDCSP'],
    'CD207+ DC': ['CD1A', 'CD207'],
    'B': ['MS4A1', 'BANK1', 'CD19', 'CD79A',  'IGHM'],
    'plasma': ['TENT5C', 'MZB1', 'SLAMF7', 'PRDM1', 'FKBP11'],
    'NK': ['GNLY', 'NKG7', 'PRF1'],
    'NKT': ['DCN', 'MGP','COL1A1'],
    'T': ['IL32', 'CCL5', 'CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B',  'CD7', 'TRAC', 'CD3D', 'TRBC2'],
    'Treg': ['FOXP3', 'CD25'],
    'mast': ['TPSB2', 'CPA3', 'MS4A2', 'KIT', 'GATA2', 'FOS2'],
    'pDC': ['CLEC4C', 'TCL1A'],
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
    'Inflammatory': ['Nfkbia', 'Il1b', 'Cxcl2', 'Cxcl15', 'Ier3', 'Sod2'],
    'MonoMac': ['Fcn1', 'S100a9', 'S100a8', 'Lyz2', 'S100a4'],
    'CD16 Mono': ['Fcgr3', 'Plac8', 'Cebpd', 'Cx3cr1'],
    'CD14 Mono': ['S100a9', 'S100a8', 'Vcan'],
    'neutrophils': ['Fcgr4', 'Csf3r', 'Cxcr2', 'Ifitm2', 'Basp1', 'G0s2'],
    'DC': ['Clec9a', 'Xcr1', 'Cd1d1', 'Cd1d2', 'Lilra6'],
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
    'Treg': ['Foxp3', 'Il2ra'],
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
    'adipocyte': ['Lipe', 'Fasn', 'Adipoq', 'Cyp2e1', 'Pcx'],
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

selected_GO_terms = []

gene_modules = {
    'Foam Cell Differentiation': ['TGFB1', 'SOAT2', 'STAT1', 'WNT5A', 'SOAT1', 'PPARG'],
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
                           'ALDOB', 'GAPDH', 'GPI', 'PFKFB1', 'PFKFB2', 'UCP2', 'ENO1', 'ENO2', 'ENO3', 'HK1', 'ENO4', 'HK3', 'HK2', 'LDHA', 'PGK2', 'PGK1'],
    'MHC Class I': [
        'PDIA3', 'ACE', 'SAR1B', 'ERAP1', 'ERAP2', 'HFE', 'IDE', 'TAP2', 'HLA-A', 'TAP1',
        'TAPBPL', 'IFI30', 'TAPBP', 'MFSD6', 'CLEC4A', 'B2M', 'CALR', 'FCER1G',
        'HLA-B', 'HLA-C', 'HLA-E'
    ],
    'MHC Ib': ['HLA-C', 'HLA-A', 'HLA-B', 'HLA-G', 'HLA-E', 'HLA-F', 'RAET1E', 'RAET1G', 'ULBP1', 'ULBP3', 'MICA', 'ULBP2', 'MICB'],
    'MHC I': ['HLA-DRB5', 'HLA-A', 'DNM2', 'HLA-E', 'HLA-F', 'FCGR2B', 'PIKFYVE', 'CLEC4A', 'GNAO1', 'HLA-DRA', 'LGMN', 'HLA-DRB4',
         'HLA-DRB3', 'HLA-DQB2', 'HLA-DRB1', 'CTSV', 'IFI30', 'CTSS', 'HLA-DMB', 'MFSD6', 'CTSL', 'HLA-DPA1', 'CTSF', 'FCER1G',
           'HLA-DOB', 'CTSE', 'HLA-DMA', 'CTSD', 'HLA-DQA2', 'HLA-DOA', 'HLA-DQA1'],
    'Apoptosis': [
        'IER3', 'EGR1', 'BTG1', 'BCL2A1', 'PTGER2', 'PTGS2', 'MCL1', 'CDKN1A', 'G0S2',
        'BCL2', 'BCL2L1', 'FAS', 'TNFRSF10B'
    ],
}
