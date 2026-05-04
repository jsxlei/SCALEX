## Adapt from BPCell
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pysam
import pyranges as pr
from matplotlib.patches import FancyArrow
from matplotlib.ticker import FuncFormatter
import re

def trackplot_theme(base_size=11):
    """Set the theme for trackplots"""
    plt.rcParams['font.size'] = base_size
    plt.rcParams['axes.grid'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'


def wrap_trackplot(plot, height=None, takes_sideplot=False, region=None, keep_vertical_margin=False):
    """Wrap a plot with trackplot attributes"""
    if height is not None:
        assert isinstance(height, (int, float))
    if region is not None:
        assert all(key in region for key in ["chr", "start", "end"])
        region_str = f"{region['chr']}:{region['start']}-{region['end']}"
    else:
        region_str = None
    
    assert isinstance(keep_vertical_margin, bool)
    
    # Add trackplot attributes to the plot object
    plot.trackplot = {
        'height': height,
        'takes_sideplot': takes_sideplot,
        'region': region_str,
        'keep_vertical_margin': keep_vertical_margin
    }
    
    return plot

def trackplot_empty(region, label):
    """Create an empty track plot when there's no data"""
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.set_xlim(region['start'], region['end'])
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel(label)
    ax.axis('off')
    return fig

def get_patchwork_plots(patchwork):
    """Extract plots from a patchwork object"""
    assert hasattr(patchwork, 'patches')
    ret = patchwork.patches
    patchwork.patches = None
    return ret

def set_trackplot_label(plot, labels):
    """Set trackplot labels"""
    assert hasattr(plot, 'axes')
    assert isinstance(labels, (list, tuple))
    
    if len(plot.axes) != len(labels):
        raise ValueError(f"set_trackplot_label(): Plot has {len(plot.axes)} facets, but {len(labels)} labels were provided")
    
    for ax, label in zip(plot.axes, labels):
        ax.set_ylabel(label)
    
    return plot

def set_trackplot_height(plot, height):
    """Set trackplot height"""
    if not hasattr(plot, 'trackplot'):
        plot = wrap_trackplot(plot)
    
    if not isinstance(height, (int, float)):
        height = float(height)
    
    plot.trackplot['height'] = height
    return plot

def get_trackplot_height(plot):
    """Get trackplot height"""
    if hasattr(plot, 'trackplot'):
        return plot.trackplot['height']
    else:
        return 1.0

def trackplot_calculate_segment_height(data):
    """Calculate y positions for trackplot segments to avoid overlap"""
    if len(data) == 0:
        return []
    
    data = data.copy()
    data['row_number'] = range(len(data))
    
    # Sort by start position for efficient overlap checking
    data = data.sort_values('Start').reset_index(drop=True)
    
    # Initialize y positions
    y_pos = [0] * len(data)
    used_positions = []  # Track which y positions are occupied
    
    for i, row in data.iterrows():
        current_start = row['Start']
        current_end = row['End']
        
        # Check if this gene overlaps with any existing genes at each y position
        best_y = 0
        for y in range(len(used_positions) + 1):
            overlap_found = False
            
            # Check overlap with genes at this y position
            for j, (y_pos_j, start_j, end_j) in enumerate(used_positions):
                if y == y_pos_j:
                    # Check if current gene overlaps with gene at this y position
                    if not (current_end <= start_j or current_start >= end_j):
                        overlap_found = True
                        break
            
            if not overlap_found:
                best_y = y
                break
        
        # Assign y position
        y_pos[i] = best_y
        
        # Add to used positions
        used_positions.append((best_y, current_start, current_end))
    
    return y_pos

def trackplot_create_arrow_segs(data, region, size=50, head_only=False):
    """Break up segments into smaller segments for arrows"""
    arrow_spacing = (region['end'] - region['start']) / size
    
    arrow_list = []
    for _, row in data.iterrows():
        if row['strand']:
            endpoints = np.arange(row['start'], row['end'], arrow_spacing)
        else:
            endpoints = np.arange(row['end'], row['start'], -arrow_spacing)
        
        if len(endpoints) > 1:
            new_arrow = pd.DataFrame({
                'start': endpoints[:-1],
                'end': endpoints[1:]
            })
            
            # Add other columns from original data
            for col in data.columns:
                if col not in ['start', 'end']:
                    new_arrow[col] = row[col]
            
            arrow_list.append(new_arrow)
        else:
            arrow_list.append(pd.DataFrame([row]))
    
    arrows = pd.concat(arrow_list, ignore_index=True)
    
    if head_only:
        # Set segment size small enough to be invisible if head_only
        for idx, row in arrows.iterrows():
            if row['strand']:
                arrows.loc[idx, 'start'] = row['end'] - 1e-4
            else:
                arrows.loc[idx, 'end'] = row['start'] + 1e-4
    
    # Filter arrows within region
    arrows = arrows[
        (arrows['start'] >= region['start']) & 
        (arrows['start'] < region['end']) & 
        (arrows['end'] >= region['start']) & 
        (arrows['end'] < region['end'])
    ]
    
    return arrows

def trackplot_normalize_ranges_with_metadata(data, metadata):
    """Normalize trackplot ranges data with metadata"""
    metadata_column_names = []
    metadata_values = {}
    
    # Check which columns need to be fetched
    for key, value in metadata.items():
        if isinstance(value, str) and len(value) == 1:
            metadata_column_names.append(value)
    
    # Normalize ranges (assuming data is already in correct format)
    data = data.copy()
    
    # Collect the metadata
    for key, value in metadata.items():
        if value is None:
            continue
        
        if isinstance(value, str) and len(value) == 1:
            metadata_values[key] = data[value]
        else:
            if len(value) != len(data):
                raise ValueError(f"Metadata for '{key}' must match length of input data frame")
            metadata_values[key] = value
        
        if key == "label":
            metadata_values[key] = [str(x) for x in metadata_values[key]]
        if key == "color":
            if not all(isinstance(x, (int, float)) for x in metadata_values[key]):
                metadata_values[key] = pd.Categorical(metadata_values[key])
    
    # Add metadata columns
    for key, value in metadata_values.items():
        data[key] = value
    
    return data

def render_plot_from_storage(plot, width, height):
    """Render a plot with intermediate disk storage step"""
    assert hasattr(plot, 'savefig')
    
    import tempfile
    import os
    
    image_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    plot.savefig(image_path, dpi=100, bbox_inches='tight')
    
    # Display the image
    img = plt.imread(image_path)
    plt.figure(figsize=(width, height))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    # Clean up
    os.unlink(image_path)


def trackplot_coverage(
    frag_file: str,
    region: str,
    cell_groups: pd.Series,
    cell_counts: pd.Series,
    bins: int = 500,
    clip_quantile: float = 0.999,
    colors: dict = None,
    return_data: bool = False,
    save: str | None = None,   # <-- new
    dpi: int = 300             # <-- new
):
    # --- Parse region ---
    # chrom, coords = region.split(":")
    # start, end = map(int, coords.split("-"))
    chrom, start, end = re.split(r"[:-]", region)
    start, end = int(start), int(end)
    bin_size = max((end - start) // bins, 1)
    bin_edges = np.arange(start, end, bin_size)
    bin_centers = bin_edges[:-1] + (bin_size // 2)

    # --- Metadata ---
    groups = cell_groups.unique()
    n_groups = len(groups)

    # --- Initialize coverage arrays ---
    coverage = {g: np.zeros(len(bin_centers), dtype=int) for g in groups}

    with pysam.TabixFile(frag_file) as tbx:
        for entry in tbx.fetch(chrom, start, end):
            chrom_f, s, e, bc, _ = entry.split("\t")
            s, e = int(s), int(e)
            if bc not in cell_groups.index:
                continue
            group = cell_groups.loc[bc]
            if group not in coverage:
                continue
            idx_start = max(0, (s - start) // bin_size)
            idx_end   = min(len(bin_centers) - 1, (e - start) // bin_size)
            coverage[group][idx_start:idx_end+1] += 1

    # --- Normalize to RPKM ---
    group_read_counts = cell_counts.groupby(cell_groups).sum()
    norm_factors = 1e9 / (group_read_counts * bin_size)

    records = []
    for g, cov in coverage.items():
        norm_cov = cov * norm_factors[g]
        for pos, val in zip(bin_centers, norm_cov):
            records.append((pos, g, val))

    data = pd.DataFrame(records, columns=["pos", "group", "normalized_insertions"])

    # --- Clip extremes ---
    ymax = data["normalized_insertions"].quantile(clip_quantile)
    data["normalized_insertions"] = np.minimum(data["normalized_insertions"], ymax)

    if return_data:
        return data

    # --- Colors ---
    if colors is None:
        unique_groups = data["group"].unique()
        palette = sns.color_palette("tab10", len(unique_groups))
        colors = dict(zip(unique_groups, palette))

    # --- Shared y-axis range across all groups ---
    global_ymin, global_ymax = 0, data["normalized_insertions"].max()

    # --- Plot setup ---
    fig, axes = plt.subplots(
        nrows=len(groups), ncols=1,
        figsize=(10, 1.0 * len(groups)),   # compact height
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0}
    )
    if len(groups) == 1:
        axes = [axes]

    # --- Coverage tracks ---
    for ax, group in zip(axes, groups):
        group_data = data[data["group"] == group]

        ax.fill_between(
            group_data["pos"],
            group_data["normalized_insertions"],
            color=colors[group],
            linewidth=0
        )

        ax.set_ylim(global_ymin, global_ymax)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

        # Group label on left
        ax.text(-0.0, 0.5, group,
                transform=ax.transAxes,
                va="center", ha="right",
                fontsize=11)

        # Add per-plot range label on top-left
        ax.text(0.01, 0.97, f"[{int(global_ymin)}-{int(global_ymax)}]",
                transform=ax.transAxes,
                va="top", ha="left", fontsize=9)

        ax.set_yticks([])
        ax.set_ylabel("")

    # --- Global labels ---
    axes[-1].set_xlabel("Genomic Position (bp)")
    fig.text(0.005, 0.5, "Insertions (RPKM)", va="center", rotation="vertical", fontsize=12)

    # Format xticks as exact numbers with commas
    formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
    axes[-1].xaxis.set_major_formatter(formatter)

    # Compact layout: shift title down
    plt.subplots_adjust(hspace=0, top=0.90, bottom=0.1, left=0.12)
    plt.suptitle(f"Coverage tracks: {region}", y=0.93, fontsize=14) #, fontweight="bold")

    # --- Save if requested ---
    if save is not None:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    plot = fig
    return wrap_trackplot(plot, height=n_groups, takes_sideplot=True, region={'chr': chrom, 'start': start, 'end': end})

    # return fig, axes
    # return fig

def trackplot_gene(transcripts, region, exon_size=2.5, gene_size=0.5, 
                  label_size=11*0.8, track_label="Genes", return_data=False):
    """Plot transcript models"""
    # Parse region
    if isinstance(region, str):
        chrom, start, end = re.split(r"[:-]", region)
        start, end = int(start), int(end)
    elif isinstance(region, (list, tuple)):
        chrom, start, end = region
        start, end = int(start), int(end)
    else:
        raise ValueError(f"Invalid region type: {type(region)}")
    
    # Filter transcripts within region
    if hasattr(transcripts, 'df'):
        df = transcripts.df
    else:
        df = transcripts
    
    df = df[
        (df['Chromosome'] == chrom) & 
        (df['End'] > start) & 
        (df['Start'] < end) & 
        (df['Feature'].isin(['transcript', 'exon']))
    ].copy()
    
    if df.empty:
        if return_data:
            return {'data': pd.DataFrame(), 'arrows': pd.DataFrame()}
        else:
            return trackplot_empty({'start': start, 'end': end, 'chr': chrom}, track_label)
    
    # Merge transcripts with the same gene name into single tracks
    # First, get unique gene names
    unique_genes = df['gene_name'].unique()
    
    # Create consolidated data for each gene
    consolidated_data = []
    consolidated_arrows = []
    
    for gene_name in unique_genes:
        gene_data = df[df['gene_name'] == gene_name].copy()
        
        # Get all exons for this gene
        exons = gene_data[gene_data['Feature'] == 'exon'].copy()
        transcripts = gene_data[gene_data['Feature'] == 'transcript'].copy()
        
        if len(exons) > 0:
            # Find the overall gene boundaries
            gene_start = exons['Start'].min()
            gene_end = exons['End'].max()
            
            # Determine the dominant strand (most common)
            strand_counts = transcripts['Strand'].value_counts()
            dominant_strand = strand_counts.index[0] if len(strand_counts) > 0 else True
            
            # Create consolidated gene track
            consolidated_data.append({
                'gene_name': gene_name,
                'Start': gene_start,  # Use uppercase to match expected column names
                'End': gene_end,      # Use uppercase to match expected column names
                'Strand': dominant_strand,
                'Feature': 'gene',
                'size': gene_size,
                'exons': exons
            })
            
            # Create arrows for the consolidated gene
            transcript_length = gene_end - gene_start
            step = max(transcript_length / 15, 200)
            
            if dominant_strand in ['+', True, 1]:
                # Positive strand: arrows point right
                positions = np.arange(gene_start, gene_end, step)
                for pos in positions:
                    if pos + step <= gene_end:
                        consolidated_arrows.append({
                            'x': pos, 'y': 0,  # y will be set later
                            'dx': step*0.7, 'dy': 0,
                            'strand': '+',
                            'gene_name': gene_name
                        })
            else:
                # Negative strand: arrows point left
                positions = np.arange(gene_end, gene_start, -step)
                for pos in positions:
                    if pos - step >= gene_start:
                        consolidated_arrows.append({
                            'x': pos, 'y': 0,  # y will be set later
                            'dx': -step*0.7, 'dy': 0,
                            'strand': '-',
                            'gene_name': gene_name
                        })
    
    # Convert to DataFrame
    consolidated_df = pd.DataFrame(consolidated_data)
    
    if len(consolidated_df) == 0:
        if return_data:
            return {'data': pd.DataFrame(), 'arrows': pd.DataFrame()}
        else:
            return trackplot_empty({'start': start, 'end': end, 'chr': chrom}, track_label)
    
    # Assign y positions to consolidated genes to avoid overlap
    consolidated_df['y'] = trackplot_calculate_segment_height(consolidated_df)
    
    # Update arrow y positions - ensure proper merge
    arrows_df = pd.DataFrame(consolidated_arrows)
    
    if len(arrows_df) > 0 and len(consolidated_df) > 0:
        # Ensure both DataFrames have the required columns
        if 'gene_name' in arrows_df.columns and 'gene_name' in consolidated_df.columns:
            arrows_df = arrows_df.merge(
                consolidated_df[['gene_name', 'y']], on='gene_name', how='left'
            )
            
            # Fill any missing y values with 0 as fallback
            if 'y' in arrows_df.columns:
                arrows_df['y'] = arrows_df['y'].fillna(0)
            else:
                arrows_df = pd.DataFrame()  # Reset to empty if merge failed
        else:
            arrows_df = pd.DataFrame()
    else:
        arrows_df = pd.DataFrame()
    
    # Final check - ensure arrows_df has required columns
    if len(arrows_df) > 0:
        required_cols = ['x', 'y', 'dx', 'dy', 'strand']
        missing_cols = [col for col in required_cols if col not in arrows_df.columns]
        if missing_cols:
            arrows_df = pd.DataFrame()  # Reset to empty if missing required columns
    
    # Clip elements to region boundaries
    consolidated_df['Start'] = consolidated_df['Start'].clip(start, end)
    consolidated_df['End'] = consolidated_df['End'].clip(start, end)
    
    if return_data:
        return {'data': consolidated_df, 'arrows': arrows_df}
    
    # Plot with compact height - match the image style
    max_y = max(consolidated_df['y']) if len(consolidated_df) > 0 else 1
    fig, ax = plt.subplots(figsize=(12, max(0.8, max_y * 0.2)))  # Much shorter height
    
    # Plot consolidated gene tracks
    for _, row in consolidated_df.iterrows():
        color = 'black' if row['Strand'] in ['+', True, 1] else 'darkgrey'
        
        # Plot the main gene line
        ax.hlines(y=row['y'], xmin=row['Start'], xmax=row['End'], 
                  linewidth=row['size'], color=color)
        
        # Plot individual exons on top
        exons = row['exons']
        for _, exon in exons.iterrows():
            ax.hlines(y=row['y'], xmin=exon['Start'], xmax=exon['End'], 
                      linewidth=exon_size, color=color)
    
    # Plot arrows with proper parameters to match image
    if len(arrows_df) > 0 and 'y' in arrows_df.columns:
        for _, row in arrows_df.iterrows():
            try:
                # Check if all required columns exist
                required_cols = ['x', 'y', 'dx', 'dy', 'strand']
                if not all(col in row.index for col in required_cols):
                    continue
                
                color = 'black' if row['strand'] == '+' else 'darkgrey'
                arrow_width = max(0.12, exon_size * 0.06)  # Thinner arrows like in image
                
                arrow = FancyArrow(
                    row['x'], row['y'], row['dx'], row['dy'],
                    width=arrow_width, 
                    length_includes_head=True, 
                    color=color,
                    head_width=arrow_width * 1.3,  # Smaller head like in image
                    head_length=abs(row['dx']) * 0.2  # Shorter head like in image
                )
                ax.add_patch(arrow)
            except Exception as e:
                # Skip problematic arrows and continue
                continue
    
    # Add gene labels positioned like in the image (without strand arrows)
    for _, row in consolidated_df.iterrows():
        label_x = (row['Start'] + row['End']) / 2
        label_y = row['y'] + 0.1  # Very close to genes for tight spacing
        
        # Just the gene name without strand arrow
        ax.text(label_x, label_y, row['gene_name'], 
                ha='center', va='bottom', fontsize=label_size,
                color='black')
    
    # Set plot limits and styling to match image exactly
    ax.set_xlim(start, end)
    ax.set_ylim(-0.3, max_y + 0.8)  # More space at top for labels to fit in spine
    ax.set_yticks([])
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel(track_label)
    
    # Add strand legend positioned like in image (top right) with arrow lines
    from matplotlib.lines import Line2D
    
    # Create legend with lines that have arrows at the end
    legend_elements = []
    
    # Positive strand: line with arrow at the end pointing right
    pos_line = Line2D([0, 1], [0, 0], color='black', lw=2, marker='>', markersize=8,  markeredgecolor='black')
    legend_elements.append(pos_line)
    
    # Negative strand: line with arrow at the end pointing left  
    neg_line = Line2D([0, 1], [0, 0], color='darkgrey', lw=2, marker='<', markersize=8,  markeredgecolor='darkgrey')
    legend_elements.append(neg_line)
    
    # Position legend outside the gene track frame
    ax.legend(legend_elements, ['+', '-'], loc='upper right', title='strand', 
              title_fontsize=label_size, fontsize=label_size, 
              frameon=False,  # Remove legend frame
              bbox_to_anchor=(1.1, 1.0))  # Move legend further to the right
    
    # Clean appearance matching the image exactly
    ax.grid(False)  # No grid lines
    ax.spines['top'].set_visible(True)  # Show top spine to cover gene tracks
    ax.spines['right'].set_visible(True)  # Show right spine
    ax.spines['left'].set_visible(True)  # Show left spine
    ax.spines['bottom'].set_visible(True)  # Show bottom spine
    
    # Remove the frame around the whole figure
    # fig.patch.set_edgecolor('black')
    # fig.patch.set_linewidth(1.0)
    
    # Format x-axis with comma separators like in image
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    
    # Adjust layout to accommodate legend outside the gene track frame
    plt.subplots_adjust(left=0.12, right=0.75, top=0.95, bottom=0.15)
    
    plot = fig
    return wrap_trackplot(plot, height=1, region={'chr': chrom, 'start': start, 'end': end})
    # return fig

def trackplot_genome_annotation(loci, region, color_by=None, colors=None, 
                               label_by=None, label_size=11*0.8, show_strand=False,
                               annotation_size=2.5, track_label="Peaks", return_data=False,
                               score_column=None, threshold=None):
    """
    Plot range-based annotation tracks (e.g. peaks) with optional score-based sizing.
    
    Parameters:
    -----------
    loci : DataFrame
        Annotation data with columns: Chromosome, Start, End, ...
    region : str
        Genomic region (e.g., "chr1:1000-2000")
    color_by : str, optional
        Column name for coloring annotations
    colors : dict, optional
        Custom color mapping
    label_by : str, optional
        Column name for labels
    label_size : float
        Font size for labels
    show_strand : bool
        Whether to show strand information
    annotation_size : float
        Base line width for annotations
    track_label : str
        Track title
    return_data : bool
        Whether to return data instead of plot
    score_column : str, optional
        Column name to use for line thickness scaling (higher scores = thicker lines)
    threshold : float, optional
        Significance threshold value to show as a red dashed horizontal line
        
    Returns:
    --------
    Wrapped trackplot object
    """
    # Parse region
    if isinstance(region, str):
        chrom, start, end = re.split(r"[:-]", region)
        start, end = int(start), int(end)
    elif isinstance(region, (list, tuple)):
        chrom, start, end = region
        start, end = int(start), int(end)
    else:
        raise ValueError(f"Invalid region type: {type(region)}")
    
    # Filter loci within region
    if hasattr(loci, 'df'):
        df = loci.df
    else:
        df = loci
    
    df = df[
        (df['Chromosome'] == chrom) & 
        (df['End'] > start) & 
        (df['Start'] < end)
    ].copy()
    
    if df.empty:
        if return_data:
            return {'data': pd.DataFrame(), 'arrows': pd.DataFrame()}
        else:
            return trackplot_empty({'start': start, 'end': end, 'chr': chrom}, track_label)
    
    # Calculate y positions based on score if available, otherwise use segment height
    if score_column and score_column in df.columns:
        # Normalize scores to 0-1 range for y positions
        min_score = df[score_column].min()
        max_score = df[score_column].max()
        if max_score > min_score:
            df['y'] = (df[score_column] - min_score) / (max_score - min_score)
        else:
            df['y'] = 0.5  # If all scores are the same, place in middle
    # else:
        # Use segment height calculation for non-overlapping tracks
    # df['y'] = trackplot_calculate_segment_height(df)
    
    # Clip to region boundaries
    df['Start'] = df['Start'].clip(start, end)
    df['End'] = df['End'].clip(start, end)
    
    if return_data:
        return {'data': df, 'arrows': None}
    
    # Plot with fixed height for normalized scores
    if score_column and score_column in df.columns:
        fig_height = 1.5  # Fixed height for normalized view
    else:
        max_y = max(df['y']) if len(df) > 0 else 1
        fig_height = 1
        # fig_height = max(max_y/2, 0.8)  # Variable height for track layout
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Plot annotations
    for _, row in df.iterrows():
        color = 'blue'
        if color_by and color_by in df.columns:
            if df[color_by].dtype in ['int64', 'float64']:
                # Use continuous color scale
                color = plt.cm.viridis((row[color_by] - df[color_by].min()) / (df[color_by].max() - df[color_by].min()))
            else:
                # Use discrete colors
                color = plt.cm.Set3(hash(row[color_by]) % 12)
        
        ax.hlines(y=row['y'], xmin=row['Start'], xmax=row['End'], 
                  linewidth=annotation_size, color=color)
    
    # Add threshold line if specified and using score_column
    if threshold is not None and score_column and score_column in df.columns and len(df) > 0:
        # Convert threshold to normalized position
        min_score = df[score_column].min()
        max_score = df[score_column].max()
        if max_score > min_score:
            threshold_normalized = (threshold - min_score) / (max_score - min_score)
            # Only draw if threshold is within the score range
            if 0 <= threshold_normalized <= 1:
                ax.axhline(y=threshold_normalized, color='red', linestyle='--', 
                          linewidth=2, alpha=0.8, zorder=10)
                # Add threshold label
                ax.text(start + (end - start) * 0.02, threshold_normalized + 0.02, 
                       f'Threshold = {threshold}', 
                       fontsize=9, color='red', ha='left', va='bottom')
    
    # Add labels if specified
    if label_by and label_by in df.columns:
        for _, row in df.iterrows():
            ax.text((row['Start'] + row['End'])/2, row['y'] + 0.25, str(row[label_by]), 
                    ha='center', va='bottom', fontsize=label_size)
    
    ax.set_xlim(start, end)
    
    # Set y-axis based on whether we're using scores
    if score_column and score_column in df.columns and len(df) > 0:
        # Set y-axis to normalized 0-1 range
        ax.set_ylim(-0.05, 1.05)  # Small padding around 0-1 range
        
        # Get original score range for right y-axis labels
        min_score = df[score_column].min()
        max_score = df[score_column].max()
        
        # Create right y-axis with original score values
        ax2 = ax.twinx()
        ax2.set_ylim(-0.05, 1.05)
        
        # Set tick positions and labels to show original score values
        tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
        if max_score > min_score:
            tick_labels = [f"{min_score + pos * (max_score - min_score):.1f}" 
                          for pos in tick_positions]
        else:
            tick_labels = [f"{min_score:.1f}"] * len(tick_positions)
        
        ax2.set_yticks(tick_positions)
        ax2.set_yticklabels(tick_labels)
        ax2.set_ylabel(f"{score_column.title()} Score", rotation=270, labelpad=15)
        ax2.tick_params(axis='y', which='major', labelsize=9)
        
        ax.set_yticks([])
    else:
        # For non-score mode, hide y-axis
        ax.set_yticks([])
    
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_title(track_label)
    
    # Format x-axis with comma separators instead of scientific notation
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
    ax.xaxis.set_major_formatter(formatter)
    
    # Use manual layout adjustment instead of tight_layout to avoid singular matrix issues
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    
    plot = fig
    return wrap_trackplot(plot, height=1, region={'chr': chrom, 'start': start, 'end': end})

def trackplot_loop(loops, region, color_by=None, colors=None, allow_truncated=True, 
                  curvature=0.75, track_label="Links", return_data=False, score_column=None):
    """Plot loops"""
    # Parse region
    if isinstance(region, str):
        chrom, start, end = re.split(r"[:-]", region)
        start, end = int(start), int(end)
    elif isinstance(region, (list, tuple)):
        chrom, start, end = region
        start, end = int(start), int(end)
    else:
        raise ValueError(f"Invalid region type: {type(region)}")
    
    # Filter loops within region
    if hasattr(loops, 'df'):
        df = loops.df
    else:
        df = loops
    
    df = df[
        (df['Chromosome'] == chrom) & 
        ((df['End'] > start) & (df['Start'] < end))
    ].copy()
    
    if df.empty:
        if return_data:
            return pd.DataFrame()
        else:
            return trackplot_empty({'start': start, 'end': end, 'chr': chrom}, track_label)
    # print(df.shape)
    
    if return_data:
        return df
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Determine max score for scaling if using score_column
    max_score = None
    min_score = None
    score_range = 1
    if score_column and score_column in df.columns:
        max_score = df[score_column].max()
        min_score = df[score_column].min()
        if max_score > min_score:
            score_range = max_score - min_score
        else:
            score_range = 1  # Avoid division by zero
    
    # Plot loops as curves
    for _, row in df.iterrows():
        x = np.linspace(row['Start'], row['End'], 100)
        
        # Use score for arc height if available, otherwise use default curvature
        if score_column and score_column in df.columns and score_column in row.index:
            # Normalize score to use as height multiplier
            normalized_score = (row[score_column] - min_score) / score_range if score_range > 0 else 0.5
            arc_height = curvature * 0.9 * (0.1 + 0.9 * normalized_score)  # Max 90% of curvature to fit below frame
        else:
            arc_height = curvature * 0.9  # Leave 10% margin at top
            
        y = arc_height * np.sin(np.pi * (x - row['Start']) / (row['End'] - row['Start']))
        
        color = 'blue'
        if color_by and color_by in df.columns:
            if df[color_by].dtype in ['int64', 'float64']:
                color = plt.cm.viridis((row[color_by] - df[color_by].min()) / (df[color_by].max() - df[color_by].min()))
            else:
                color = plt.cm.Set3(hash(row[color_by]) % 12)
        
        ax.plot(x, y, color=color, linewidth=1)
    
    ax.set_xlim(start, end)
    ax.set_ylim(0, curvature)  # Start from bottom axis (y=0)
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_title(track_label)
    ax.set_yticks([])
    
    # Add score axis on the right if using score_column
    if score_column and score_column in df.columns and max_score is not None:
        # Create secondary y-axis for scores
        ax2 = ax.twinx()
        
        # Set score axis limits and ticks
        ax2.set_ylim(0, curvature)
        
        # Calculate score tick positions and labels
        # Map curvature positions back to score values
        score_ticks = []
        score_labels = []
        
        # Create 5 evenly spaced score ticks
        for i in range(6):  # 0, 20%, 40%, 60%, 80%, 100%
            score_fraction = i / 5.0
            score_value = min_score + score_fraction * score_range
            
            # Calculate corresponding y position
            # Reverse the arc_height calculation
            normalized_score = score_fraction
            y_position = curvature * 0.9 * (0.1 + 0.9 * normalized_score)
            
            score_ticks.append(y_position)
            score_labels.append(f"{score_value:.2f}")
        
        ax2.set_yticks(score_ticks)
        ax2.set_yticklabels(score_labels)
        ax2.set_ylabel("Score", rotation=270, labelpad=15)
        ax2.tick_params(axis='y', labelsize=9)
    
    fig.tight_layout()
    
    plot = fig

    return wrap_trackplot(plot, height=1, region={'chr': chrom, 'start': start, 'end': end})

def trackplot_scalebar(region, font_pt=11):
    """Plot scale bar"""
    # Parse region
    if isinstance(region, str):
        chrom, start, end = re.split(r"[:-]", region)
        start, end = int(start), int(end)
    elif isinstance(region, (list, tuple)):
        chrom, start, end = region
        start, end = int(start), int(end)
    else:
        raise ValueError(f"Invalid region type: {type(region)}")
    
    # Calculate scale bar
    width = end - start
    bar_width = width * 0.1
    
    fig, ax = plt.subplots(figsize=(12, 1))
    
    # Add region text
    ax.text(start, 0, f"{chrom}: {start:,} - {end:,}", 
            fontsize=font_pt, ha='left', va='center')
    
    # Add scale bar with vertical lines at ends
    bar_start = end - bar_width - width * 0.05
    bar_end = end - width * 0.05
    
    # Horizontal scale bar line (thinner)
    ax.plot([bar_start, bar_end], [0, 0], 'k-', linewidth=1)
    
    # Vertical lines at the ends (thinner)
    ax.plot([bar_start, bar_start], [-0.05, 0.05], 'k-', linewidth=1)
    ax.plot([bar_end, bar_end], [-0.05, 0.05], 'k-', linewidth=1)
    
    # Distance text on the left of the scale bar (in kb)
    bar_width_kb = bar_width / 1000
    ax.text(bar_start - width * 0.01, 0, f"{bar_width_kb:.0f}k", 
            fontsize=font_pt, ha='right', va='center')
    
    ax.set_xlim(start, end)
    ax.set_ylim(-0.2, 0.2)
    
    # Remove axes completely
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # Remove frame/spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    plot = fig
    return wrap_trackplot(plot, height=font_pt*1.1/72, region={'chr': chrom, 'start': start, 'end': end}, 
                         keep_vertical_margin=False)
    # return plot

# Deprecated functions for backward compatibility
def draw_trackplot_grid(*plots, labels, title=None, heights=None, label_width=0.2, label_style=None):
    """Deprecated: Use trackplot_combine instead"""
    print("Warning: draw_trackplot_grid is deprecated. Use trackplot_combine instead.")
    return trackplot_combine(plots, title=title)

def trackplot_bulk(*args, **kwargs):
    """Deprecated: Use trackplot_coverage instead"""
    print("Warning: trackplot_bulk is deprecated. Use trackplot_coverage instead.")
    return trackplot_coverage(*args, **kwargs)


def trackplot_combine(
    frag_file: str,
    region: str,
    cell_groups: pd.Series,
    cell_counts: pd.Series,
    extend: int = 0,
    up: int = None,
    down: int = None,
    transcripts = None,
    loops = None,
    loci = None,
    bins: int = 500,
    clip_quantile: float = 0.999,
    colors: dict = None,
    exon_size: float = 2.5,
    gene_size: float = 0.5,
    label_size: float = 11*0.8,
    coverage_height: float = 2.0,
    gene_height: float = 1.0,
    loop_height: float = 1.0,
    annotation_height: float = 1.0,
    loop_score_column: str = None,
    loci_score_column: str = None,
    loci_threshold: float = None,
    loci_color_by: str = None,
    hspace: float = 0.05,
    return_data: bool = False,
    save: str | None = None,
    dpi: int = 300
):
    """
    Combined function to plot coverage, annotations, loops, and gene tracks together in one figure.
    
    Parameters:
    -----------
    frag_file : str
        Path to fragment file
    region : str
        Genomic region to plot (e.g., "chr1:1000-2000")
    cell_groups : pd.Series
        Cell group assignments
    cell_counts : pd.Series
        Cell read counts for normalization
    transcripts : DataFrame or similar, optional
        Transcript/gene annotation data
    loops : DataFrame or similar, optional
        Loop/interaction data with columns: Chromosome, Start, End, optional Score
    loci : DataFrame or similar, optional
        Genomic annotation data (peaks, variants, etc.) with columns: Chromosome, Start, End
    bins : int
        Number of bins for coverage calculation
    clip_quantile : float
        Quantile for clipping extreme values
    colors : dict
        Color mapping for cell groups
    exon_size : float
        Line width for exons
    gene_size : float
        Line width for gene bodies
    label_size : float
        Font size for labels
    coverage_height : float
        Height ratio for coverage track
    gene_height : float
        Height ratio for gene track
    loop_height : float
        Height ratio for loop track
    annotation_height : float
        Height ratio for annotation track
    loop_score_column : str, optional
        Column name in loops DataFrame to use for arc heights
    annotation_score_column : str, optional
        Column name in loci DataFrame to use for y-positioning
    annotation_threshold : float, optional
        Significance threshold for annotation track
    annotation_color_by : str, optional
        Column name in loci DataFrame for coloring annotations
    hspace : float
        Space between tracks
    return_data : bool
        Whether to return data instead of plotting
    save : str, optional
        Path to save the figure
    dpi : int
        DPI for saved figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        Combined plot with coverage, annotations, loops, and gene tracks
    """
    
    # Parse region
    chrom, start, end = re.split(r"[:-]", region)
    start, end = int(start), int(end)
    if up is not None:
        start -= up
    else:
        start -= extend
    if down is not None:
        end += down
    else:
        end += extend
    
    if return_data:
        # Return data from both functions
        coverage_data = trackplot_coverage(
            frag_file, region, cell_groups, cell_counts, bins, clip_quantile, 
            colors, return_data=True
        )
        gene_data = trackplot_gene(transcripts, region, return_data=True)
        return {'coverage': coverage_data, 'genes': gene_data}
    
    # Create figure with separate tracks like original individual functions
    groups = cell_groups.unique()
    n_groups = len(groups)
    
    # Check if transcripts, loops, and loci are available
    has_transcripts = transcripts is not None and len(transcripts) > 0
    has_loops = loops is not None and len(loops) > 0
    has_loci = loci is not None and len(loci) > 0
    
    # Calculate number of rows and height ratios
    n_rows = n_groups + 1  # Base: coverage tracks + scale bar
    height_ratios = [0.3] + [1.0] * n_groups  # Scale bar + coverage tracks
    
    if has_loci:
        n_rows += 1
        height_ratios.append(annotation_height)  # Add annotation track
    
    if has_loops:
        n_rows += 1
        height_ratios.append(loop_height)  # Add loop track
    
    if has_transcripts:
        n_rows += 1
        height_ratios.append(gene_height)  # Add gene track
    
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=1,
        figsize=(12, n_groups + gene_height + 0.5),
        height_ratios=height_ratios,
        sharex=True,
        gridspec_kw={"hspace": 0}  # No gap between tracks
    )
    
    # Assign axes based on available tracks
    current_idx = 0
    
    # Scale bar is always first
    scalebar_ax = axes[current_idx]
    current_idx += 1
    
    # Coverage axes
    coverage_axes = axes[current_idx:current_idx + n_groups]
    current_idx += n_groups
    
    # Annotation axis (if available)
    if has_loci:
        annotation_ax = axes[current_idx]
        current_idx += 1
    
    # Loop axis (if available)
    if has_loops:
        loop_ax = axes[current_idx]
        current_idx += 1
    
    # Gene axis (if available)
    if has_transcripts:
        gene_ax = axes[current_idx]
        current_idx += 1
    
    # Calculate coverage data
    bin_size = max((end - start) // bins, 1)
    bin_edges = np.arange(start, end, bin_size)
    bin_centers = bin_edges[:-1] + (bin_size // 2)
    
    if pd.api.types.is_categorical_dtype(cell_groups):
        groups = cell_groups.cat.categories
    else:
        groups = cell_groups.unique()
    n_groups = len(groups)
    coverage = {g: np.zeros(len(bin_centers), dtype=int) for g in groups}
    
    # Read fragment data
    with pysam.TabixFile(frag_file) as tbx:
        for entry in tbx.fetch(chrom, start, end):
            chrom_f, s, e, bc, _ = entry.split("\t")
            s, e = int(s), int(e)
            if bc not in cell_groups.index:
                continue
            group = cell_groups.loc[bc]
            if group not in coverage:
                continue
            idx_start = max(0, (s - start) // bin_size)
            idx_end = min(len(bin_centers) - 1, (e - start) // bin_size)
            coverage[group][idx_start:idx_end+1] += 1
    
    # Normalize to RPKM
    group_read_counts = cell_counts.groupby(cell_groups).sum()
    norm_factors = 1e9 / (group_read_counts * bin_size)
    
    # Plot coverage for each group (like original trackplot_coverage)
    if colors is None:
        unique_groups = list(groups)
        palette = sns.color_palette("tab10", len(unique_groups))
        colors = dict(zip(unique_groups, palette))
    
    # Shared y-axis range across all groups (like original)
    global_ymin, global_ymax = 0, 0
    for group in groups:
        norm_cov = coverage[group] * norm_factors[group]
        global_ymax = max(global_ymax, norm_cov.max())
    
    # Clip extremes (like original)
    ymax = np.percentile([cov * norm_factors[g] for g, cov in coverage.items()], clip_quantile * 100)
    global_ymax = min(global_ymax, ymax)
    
    # Add scale bar track first (at the top)
    # Calculate scale bar
    width = end - start
    bar_width = width * 0.1
    
    # Add region text
    scalebar_ax.text(start, 0, f"{chrom}: {start:,} - {end:,}", 
                    fontsize=11, ha='left', va='center')
    
    # Add scale bar with vertical lines at ends
    bar_start = end - bar_width - width * 0.05
    bar_end = end - width * 0.05
    
    # Horizontal scale bar line (thinner)
    scalebar_ax.plot([bar_start, bar_end], [0, 0], 'k-', linewidth=1)
    
    # Vertical lines at the ends (thinner)
    scalebar_ax.plot([bar_start, bar_start], [-0.05, 0.05], 'k-', linewidth=1)
    scalebar_ax.plot([bar_end, bar_end], [-0.05, 0.05], 'k-', linewidth=1)
    
    # Distance text on the left of the scale bar (in kb)
    bar_width_kb = bar_width / 1000
    scalebar_ax.text(bar_start - width * 0.01, 0, f"{bar_width_kb:.0f}k", 
                    fontsize=11, ha='right', va='center')
    
    scalebar_ax.set_xlim(start, end)
    scalebar_ax.set_ylim(-0.2, 0.2)
    
    # Remove axes completely for scale bar
    scalebar_ax.set_xticks([])
    scalebar_ax.set_yticks([])
    scalebar_ax.set_xlabel("")
    scalebar_ax.set_ylabel("")
    
    # Remove frame/spines for scale bar
    for spine in scalebar_ax.spines.values():
        spine.set_visible(False)
    
    # Plot each group separately (like original trackplot_coverage)
    for i, group in enumerate(groups):
        norm_cov = coverage[group] * norm_factors[group]
        norm_cov = np.minimum(norm_cov, global_ymax)
        
        # Get the appropriate axis for this group
        if n_groups == 1:
            ax = coverage_axes[0] if 'coverage_axes' in locals() else axes[0]
        else:
            ax = coverage_axes[i]
        
        ax.fill_between(
            bin_centers,
            norm_cov,
            color=colors[group],
            linewidth=0
        )
        
        ax.set_ylim(global_ymin, global_ymax)
        
        # Styling like original trackplot_coverage
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        
        # Group label on left (like original)
        ax.text(-0.0, 0.5, group,
                transform=ax.transAxes,
                va="center", ha="right",
                fontsize=11)
        
        # Add per-plot range label on top-left (like original)
        ax.text(0.015, 0.97, f"[{int(global_ymin)}-{int(global_ymax)}]",
                transform=ax.transAxes,
                va="top", ha="left", fontsize=9)
        
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xticklabels([])  # Hide x-axis labels
    
    # Annotation track (only if loci are available)
    if has_loci:
        # Filter loci within region
        if hasattr(loci, 'df'):
            loci_df = loci.df
        else:
            loci_df = loci
        
        loci_df = loci_df[
            (loci_df['Chromosome'] == chrom) & 
            (loci_df['End'] > start) & 
            (loci_df['Start'] < end)
        ].copy()
        
        if not loci_df.empty:
            # Calculate y positions based on score if available
            if loci_score_column and loci_score_column in loci_df.columns:
                # Normalize scores to 0-1 range for y positions
                min_score = loci_df[loci_score_column].min()
                max_score = loci_df[loci_score_column].max()
                if max_score > min_score:
                    loci_df['y'] = (loci_df[loci_score_column] - min_score) / (max_score - min_score)
                else:
                    loci_df['y'] = 0.5  # If all scores are the same, place in middle
            else:
                # Use segment height calculation for non-overlapping tracks
                loci_df['y'] = trackplot_calculate_segment_height(loci_df)
            
            # Clip to region boundaries
            loci_df['Start'] = loci_df['Start'].clip(start, end)
            loci_df['End'] = loci_df['End'].clip(start, end)
            
            # Plot annotations
            for _, row in loci_df.iterrows():
                color = 'blue'
                if loci_color_by and loci_color_by in loci_df.columns:
                    if loci_df[annotation_color_by].dtype in ['int64', 'float64']:
                        # Use continuous color scale
                        color = plt.cm.viridis((row[annotation_color_by] - loci_df[annotation_color_by].min()) / 
                                             (loci_df[annotation_color_by].max() - loci_df[annotation_color_by].min()))
                    else:
                        # Use discrete colors
                        color = plt.cm.Set3(hash(row[loci_color_by]) % 12)
                
                annotation_ax.hlines(y=row['y'], xmin=row['Start'], xmax=row['End'], 
                                   linewidth=2.5, color=color)
            
            # Add threshold line if specified
            if (loci_threshold is not None and loci_score_column and 
                loci_score_column in loci_df.columns):
                # Convert threshold to normalized position
                if max_score > min_score:
                    threshold_normalized = (loci_threshold - min_score) / (max_score - min_score)
                    # Only draw if threshold is within the score range
                    if 0 <= threshold_normalized <= 1:
                        annotation_ax.axhline(y=threshold_normalized, color='red', linestyle='--', 
                                            linewidth=2, alpha=0.8, zorder=10)
            
            # Set annotation track properties
            annotation_ax.set_xlim(start, end)
            if loci_score_column and loci_score_column in loci_df.columns:
                annotation_ax.set_ylim(-0.05, 1.05)
            else:
                max_y = loci_df['y'].max() if len(loci_df) > 0 else 1
                annotation_ax.set_ylim(-0.1, max_y + 0.1)
            
            annotation_ax.set_yticks([])
            annotation_ax.set_ylabel("Loci")
            annotation_ax.set_xticklabels([])
        else:
            # Empty annotation track
            annotation_ax.set_xlim(start, end)
            annotation_ax.set_ylim(0, 1)
            annotation_ax.set_yticks([])
            annotation_ax.set_ylabel("Loci")
            annotation_ax.set_xticklabels([])
    
    # Loop track (only if loops are available)
    if has_loops:
        # Filter loops within region
        if hasattr(loops, 'df'):
            loop_df = loops.df
        else:
            loop_df = loops
        
        loop_df = loop_df[
            (loop_df['Chromosome'] == chrom) & 
            ((loop_df['End'] > start) & (loop_df['Start'] < end))
        ].copy()
        
        if not loop_df.empty:
            # Determine max score for scaling if using score_column
            max_score = None
            min_score = None
            score_range = 1
            curvature = 0.75  # Default curvature for loops
            
            if loop_score_column and loop_score_column in loop_df.columns:
                max_score = loop_df[loop_score_column].max()
                min_score = loop_df[loop_score_column].min()
                if max_score > min_score:
                    score_range = max_score - min_score
                else:
                    score_range = 1
            
            # Plot loops as curves
            for _, row in loop_df.iterrows():
                x = np.linspace(row['Start'], row['End'], 100)
                
                # Use score for arc height if available
                if loop_score_column and loop_score_column in loop_df.columns and loop_score_column in row.index:
                    normalized_score = (row[loop_score_column] - min_score) / score_range if score_range > 0 else 0.5
                    arc_height = curvature * 0.9 * (0.1 + 0.9 * normalized_score)
                else:
                    arc_height = curvature * 0.9
                    
                y = arc_height * np.sin(np.pi * (x - row['Start']) / (row['End'] - row['Start']))
                
                # Default color for loops
                color = 'red'
                loop_ax.plot(x, y, color=color, linewidth=1)
            
            # Set loop track properties
            loop_ax.set_xlim(start, end)
            loop_ax.set_ylim(0, curvature)
            loop_ax.set_yticks([])
            loop_ax.set_ylabel("Loops")
            loop_ax.set_xticklabels([])
            
            # Add score axis on the right if using score_column
            if loop_score_column and loop_score_column in loop_df.columns and max_score is not None:
                loop_ax2 = loop_ax.twinx()
                loop_ax2.set_ylim(0, curvature)
                
                # Calculate score tick positions and labels
                score_ticks = []
                score_labels = []
                
                for i in range(6):  # 0, 20%, 40%, 60%, 80%, 100%
                    score_fraction = i / 5.0
                    score_value = min_score + score_fraction * score_range
                    normalized_score = score_fraction
                    y_position = curvature * 0.9 * (0.1 + 0.9 * normalized_score)
                    
                    score_ticks.append(y_position)
                    score_labels.append(f"{score_value:.2f}")
                
                loop_ax2.set_yticks(score_ticks)
                loop_ax2.set_yticklabels(score_labels)
                loop_ax2.set_ylabel("Loop Score", rotation=270, labelpad=15)
                loop_ax2.tick_params(axis='y', labelsize=9)
        else:
            # Empty loop track
            loop_ax.set_xlim(start, end)
            loop_ax.set_ylim(0, 0.75)
            loop_ax.set_yticks([])
            loop_ax.set_ylabel("Loops")
            loop_ax.set_xticklabels([])
    
    # Gene track (only if transcripts are available)
    if has_transcripts:
        # Filter transcripts within region
        if hasattr(transcripts, 'df'):
            df = transcripts.df
        else:
            df = transcripts
    
        df = df[
            (df['Chromosome'] == chrom) & 
            (df['End'] > start) & 
            (df['Start'] < end) & 
            (df['Feature'].isin(['transcript', 'exon']))
        ].copy()
        
        # if not df.empty:
        # Merge transcripts with the same gene name into single tracks
        unique_genes = df['gene_name'].unique()
        consolidated_data = []
        
        for gene_name in unique_genes:
            gene_data = df[df['gene_name'] == gene_name].copy()
            exons = gene_data[gene_data['Feature'] == 'exon'].copy()
            transcripts = gene_data[gene_data['Feature'] == 'transcript'].copy()
            
            if len(exons) > 0:
                gene_start = exons['Start'].min()
                gene_end = exons['End'].max()
                strand_counts = transcripts['Strand'].value_counts()
                dominant_strand = strand_counts.index[0] if len(strand_counts) > 0 else True
                
                consolidated_data.append({
                    'gene_name': gene_name,
                    'Start': gene_start,
                    'End': gene_end,
                    'Strand': dominant_strand,
                    'exons': exons
                })
        
        # Assign y positions to avoid overlap
        if consolidated_data:
            consolidated_df = pd.DataFrame(consolidated_data)
            consolidated_df['y'] = trackplot_calculate_segment_height(consolidated_df)
            
            # Clip elements to region boundaries
            consolidated_df['Start'] = consolidated_df['Start'].clip(start, end)
            consolidated_df['End'] = consolidated_df['End'].clip(start, end)
            
            # Plot genes
            for _, row in consolidated_df.iterrows():
                color = 'black' if row['Strand'] in ['+', True, 1] else 'darkgrey'
                
                # Plot the main gene line
                gene_ax.hlines(y=row['y'], xmin=row['Start'], xmax=row['End'], 
                              linewidth=gene_size, color=color)
                
                # Plot individual exons
                exons = row['exons']
                for _, exon in exons.iterrows():
                    gene_ax.hlines(y=row['y'], xmin=exon['Start'], xmax=exon['End'], 
                                  linewidth=exon_size, color=color)
                
                # Add gene label
                label_x = (row['Start'] + row['End']) / 2
                label_y = row['y'] + 0.1
                gene_ax.text(label_x, label_y, row['gene_name'], 
                            ha='center', va='bottom', fontsize=label_size, color='black')
            
            max_y = consolidated_df['y'].max() if len(consolidated_df) > 0 else 1
            gene_ax.set_ylim(-0.3, max_y + 0.8)
        else:
            gene_ax.set_ylim(0, 1)
    # else:
    #     gene_ax.set_ylim(0, 1)
    
    # Styling like original trackplot_gene (only if transcripts are available)
    # if has_transcripts:
        gene_ax.set_yticks([])
        gene_ax.set_xlabel("Genomic Position (bp)")
        gene_ax.set_ylabel("Genes")
        
        # Clean appearance matching the original
        gene_ax.grid(False)  # No grid lines
        gene_ax.spines['top'].set_visible(True)
        gene_ax.spines['right'].set_visible(True)
        gene_ax.spines['left'].set_visible(True)
        gene_ax.spines['bottom'].set_visible(True)
    
    # Set x-axis limits for all axes
    for i in range(n_rows):
        axes[i].set_xlim(start, end)
    
    # Format x-axis with comma separators like original (only if transcripts are available)
    if has_transcripts:
        from matplotlib.ticker import FuncFormatter
        formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
        gene_ax.xaxis.set_major_formatter(formatter)
    
    # No legend for gene track (removed as requested)
    
    # Add global labels like original trackplot_coverage
    fig.text(0.005, 0.5, "Insertions (RPKM)", va="center", rotation="vertical", fontsize=12)
    
    # No title (removed as requested)
    
    # Layout adjustment - no gaps between tracks
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.12, right=0.88)
    
    # Save if requested
    if save is not None:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    
    return fig






