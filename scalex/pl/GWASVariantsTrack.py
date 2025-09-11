# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .GenomeTrack import GenomeTrack

class GWASVariantsTrack(GenomeTrack):
    """
    A track that plots GWAS variants within a specified region.
    This version hardcodes the input file path and processes data inside the plot method.
    The init_data() step is removed and no file input is read from configuration.
    """

    SUPPORTED_ENDINGS = []
    TRACK_TYPE = 'gwasvariants'
    OPTIONS_TXT = """
    height = 3
    title =
    file_type = gwasvariants
    color = blue
    highlight_color = crimson
    highlight_rsid =
    highlight_threshold = 5e-8
    significance_threshold = 5e-8
    point_size = 10
    highlight_point_size = 40
    draw_significance_line = true
    draw_vertical_lines = false
    vertical_line_offset = 500
    """

    # Default values for properties
    DEFAULTS_PROPERTIES = {
        'color': 'blue',
        'highlight_color': 'crimson',
        'highlight_rsid': '',
        'highlight_threshold': '5e-8',
        'significance_threshold': '5e-8',
        'point_size': '10',
        'highlight_point_size': '40',
        'draw_significance_line': 'true',
        'draw_vertical_lines': 'false',
        'vertical_line_offset': '500',
        'alpha': '1.0'
    }

    # With no file in configuration, we don't need it as necessary property
    NECESSARY_PROPERTIES = []

    SYNONYMOUS_PROPERTIES = {}
    POSSIBLE_PROPERTIES = {}
    BOOLEAN_PROPERTIES = ['draw_significance_line', 'draw_vertical_lines']
    STRING_PROPERTIES = ['title', 'file_type', 'color', 'highlight_color', 'highlight_rsid', 'file', 'pval_col', 'variant_id_col']
    FLOAT_PROPERTIES = {
        'height': [0, np.inf],
        'highlight_threshold': [0, np.inf],
        'significance_threshold': [0, np.inf],
        'vertical_line_offset': [0, np.inf],
        'min_value': [-np.inf, np.inf],
        'max_value': [-np.inf, np.inf],
        'alpha': [0, 1]
    }
    INTEGER_PROPERTIES = {
        'point_size': [0, np.inf],
        'highlight_point_size': [0, np.inf],
    }

    def plot(self, ax, chrom, region_start, region_end):
        print("Plotting GWAS variants...")

        variant_file_path = self.properties['file']
        pval_col = self.properties['pval_col']
        variant_id_col = self.properties['variant_id_col']
        variant_data = pd.read_table(variant_file_path)
        variant_data['-log10_gwas_pval'] = -np.log10(variant_data[pval_col])
        variant_data.dropna(subset=['-log10_gwas_pval'], inplace=True)

        if variant_id_col:
            variant_data['chr_hg38'] = variant_data[variant_id_col].str.split(':').str[0]
            variant_data['pos_hg38'] = variant_data[variant_id_col].str.split(':').str[1].astype(int)

        region_data = variant_data[
            (variant_data['chr_hg38'] == chrom) &
            (variant_data['pos_hg38'] >= region_start) &
            (variant_data['pos_hg38'] <= region_end)
        ]

        if region_data.empty:
            print("No variants found in this region.")
            return ax

        color = self.properties['color']
        highlight_color = self.properties['highlight_color']
        highlight_rsid = self.properties['highlight_rsid']
        highlight_threshold = float(self.properties['highlight_threshold'])
        significance_threshold = float(self.properties['significance_threshold'])
        point_size = int(self.properties['point_size'])
        highlight_point_size = int(self.properties['highlight_point_size'])
        draw_significance_line = self.properties['draw_significance_line']
        draw_vertical_lines = self.properties['draw_vertical_lines']
        vertical_line_offset = float(self.properties['vertical_line_offset'])
        alpha = float(self.properties['alpha'])

        # Plot all variants
        ax.scatter(region_data['pos_hg38'], region_data['-log10_gwas_pval'],
                   c=color, s=point_size, alpha=alpha)

        # Highlight a specific variant if provided
        if highlight_rsid:
            highlight_variants = region_data[region_data['rsid'] == highlight_rsid]
            if not highlight_variants.empty:
                ax.scatter(
                    highlight_variants['pos_hg38'],
                    highlight_variants['-log10_gwas_pval'],
                    c=highlight_color, s=highlight_point_size
                )
                if draw_vertical_lines == 'true':
                    pos_val = highlight_variants['pos_hg38'].values[0]
                    vline_start = pos_val - vertical_line_offset
                    vline_end = pos_val + vertical_line_offset
                    ax.axvline(vline_start, color='black', linestyle='--', linewidth=0.7)
                    ax.axvline(vline_end, color='black', linestyle='--', linewidth=0.7)

                    with open('/users/soumyak/crc/scripts/plot_data/vlines.bed', 'w') as f:
                        f.write(f'{chrom}\t{vline_start - 1}\t{vline_start}\n')
                        f.write(f'{chrom}\t{vline_end - 1}\t{vline_end}\n')

        if highlight_threshold:
            highlight_variants = region_data[region_data[pval_col] < highlight_threshold]
            if not highlight_variants.empty:
                ax.scatter(
                    highlight_variants['pos_hg38'],
                    highlight_variants['-log10_gwas_pval'],
                    c=highlight_color, s=highlight_point_size
                )

        if draw_significance_line:
            print('here')
            thresh_line = -np.log10(significance_threshold)
            # ax.axhline(y=thresh_line, color='red', linestyle='--', linewidth=0.9, label="GWAS Significance Threshold")
            ax.axhline(y=thresh_line, color='red', linestyle='--', linewidth=0.9)

        ax.set_xlim(region_start, region_end)
        if 'min_value' in self.properties and 'max_value' in self.properties:
            ax.set_ylim(float(self.properties['min_value']), float(self.properties['max_value']))
        elif 'min_value' in self.properties:
            ax.set_ylim(bottom=float(self.properties['min_value']))
        ax.set_ylabel("-log10(P-value)")
        # ax.legend()

        return ax

