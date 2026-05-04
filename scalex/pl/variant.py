import numpy as np
import torch
import matplotlib.pyplot as plt
import tangermeme.plot
from scipy.special import softmax as scipy_softmax
import pyfaidx

from chrombpnet.data_utils import dna_to_one_hot
from chrombpnet.interpret import _deep_lift_shap
from chrombpnet.model_wrappers import CountWrapper, ProfileWrapper, init_chrombpnet_wo_bias

INPUTLEN = 2114
OUTPUTLEN = 1000


def _load_model(model, device):
    if isinstance(model, str):
        if model.endswith('.ckpt'):
            from chrombpnet.model_wrappers import ChromBPNetWrapper
            wrapper = ChromBPNetWrapper.load_from_checkpoint(model, map_location='cpu')
            model = wrapper.chrombpnet_wo_bias
        elif model.endswith('.h5') or model.endswith('.pt'):
            model = init_chrombpnet_wo_bias(model, freeze=False)
        else:
            raise ValueError(f"Unsupported model file format: {model!r}. Expected .ckpt, .h5, or .pt")
    model.eval().to(device)
    return model


def _prepare_sequences(chrom, pos, ref, alt, fasta):
    flank = INPUTLEN // 2
    pos0 = pos - 1  # convert 1-based (VCF) to 0-based for pyfaidx
    genome = pyfaidx.Fasta(fasta)
    window = str(genome[chrom][pos0 - flank : pos0 + flank])
    alt_seq = window[:flank] + alt + window[flank + len(ref):]
    return window, alt_seq


def _predict(model, x, device):
    with torch.no_grad():
        profile_logits, logcount = model(x.to(device))
    total = np.exp(logcount.squeeze().cpu().numpy())
    pred = scipy_softmax(profile_logits.squeeze().cpu().numpy()) * total
    return pred


def _run_shap(model, x, task, n_shuffles, device):
    shap_model = CountWrapper(model) if task == 'counts' else ProfileWrapper(model)
    attr = _deep_lift_shap(
        shap_model, x, n_shuffles=n_shuffles, device=device, verbose=True,
        warning_threshold=1e8,
    )
    return attr.squeeze(0).cpu().numpy()  # (4, INPUTLEN)


def _parse_variant(variant):
    parts = variant.split(':')
    if len(parts) != 4:
        raise ValueError(f"variant must be 'chrom:pos:ref:alt', got: {variant!r}")
    chrom, pos, ref, alt = parts
    return chrom, int(pos), ref, alt


def plot_variant_effect(
    model,
    chrom=None,
    pos=None,
    ref=None,
    alt=None,
    fasta=None,
    variant=None,
    device=None,
    task='counts',
    n_shuffles=20,
    contrib_flank=70,
    cell_type=None,
    title=None,
    output=None,
):
    if variant is not None:
        chrom, pos, ref, alt = _parse_variant(variant)
    elif None in (chrom, pos, ref, alt):
        raise ValueError("Provide either 'variant' or all of chrom, pos, ref, alt")

    if device is None:
        device = 'cpu'

    model = _load_model(model, device)

    ref_seq, alt_seq = _prepare_sequences(chrom, pos, ref, alt, fasta)

    ohe = dna_to_one_hot([ref_seq, alt_seq])              # (2, INPUTLEN, 4)
    x = torch.tensor(ohe, dtype=torch.float32).permute(0, 2, 1)  # (2, 4, INPUTLEN)
    x_ref, x_alt = x[0:1], x[1:2]

    pred_ref = _predict(model, x_ref, device)
    pred_alt = _predict(model, x_alt, device)

    attr_ref = _run_shap(model, x_ref, task, n_shuffles, device)  # (4, INPUTLEN)
    attr_alt = _run_shap(model, x_alt, task, n_shuffles, device)

    var_pos = INPUTLEN // 2  # 1057
    half_out = OUTPUTLEN // 2  # 500
    attr_ref_short = attr_ref.T[var_pos - half_out : var_pos + half_out]  # (1000, 4)
    attr_alt_short = attr_alt.T[var_pos - half_out : var_pos + half_out]

    fig, ax = plt.subplots(4, 1, figsize=(11, 6))
    alpha = 0.3

    ax[0].plot(pred_ref, linewidth=0.5, label="REF", color="gray")
    ax[0].plot(pred_alt, linewidth=0.5, label="ALT", color="red")
    ax[0].set_ylabel("Pred.\naccessibility")
    ax[0].axvspan(half_out - contrib_flank, half_out + contrib_flank, alpha=alpha, color='yellow')
    ax[0].legend(fontsize=8)

    ax[1].plot(attr_ref_short.sum(axis=1), linewidth=0.5, label="REF", color="gray")
    ax[1].plot(attr_alt_short.sum(axis=1), linewidth=0.5, label="ALT", color="red")
    ax[1].set_ylabel("Contrib.\nscores")
    ax[1].axvspan(half_out - contrib_flank, half_out + contrib_flank, alpha=alpha, color='yellow')

    tangermeme.plot.plot_logo(
        np.float64(attr_ref), start=var_pos - contrib_flank,
        end=var_pos + contrib_flank, ax=ax[2],
    )
    ax[2].set_ylim(-0.05, 0.15)
    ax[2].axvspan(contrib_flank - 0.5, contrib_flank + 0.5, alpha=alpha - 0.1, color='gray')
    ax[2].set_ylabel("Contrib.\nREF")

    tangermeme.plot.plot_logo(
        np.float64(attr_alt), start=var_pos - contrib_flank,
        end=var_pos + contrib_flank, ax=ax[3],
    )
    ax[3].set_ylim(-0.05, 0.15)
    ax[3].axvspan(contrib_flank - 0.5, contrib_flank + 0.5, alpha=alpha - 0.1, color='red')
    ax[3].set_ylabel("Contrib.\nALT")

    if title is None:
        variant_label = f"{chrom}:{pos}:{ref}:{alt}"
        title = f"{cell_type} — {variant_label}" if cell_type else variant_label
    plt.suptitle(title, fontsize=12)
    plt.xlabel("Relative genomic position (bp)")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')

    return fig
