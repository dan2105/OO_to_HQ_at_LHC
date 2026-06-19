import ROOT as root
import atlasplots as aplt
import math
import os
import re

MASS_THRESHOLDS = [10, 15, 20, 25, 30, 40]


PDF_RATIO_VARIABLES = [
]

for thr in MASS_THRESHOLDS:
    PDF_RATIO_VARIABLES.append(
        (f"h_ptbb{thr}", f"ptbb_mbb{thr}", "p_{T}(b#bar{b}) [GeV]", (0, 50), (1e2, 10000000000), 10, "d#sigma/dp_{T}(b#bar{b}) [nb/GeV]")
    )
    PDF_RATIO_VARIABLES.append(
        (f"h_ybb{thr}", f"ybb_mbb{thr}", "y_{b#bar{b}}", (-3, 3), (1e2, 10000000000), 10, "d#sigma/dy_{b#bar{b}}")
    )
    PDF_RATIO_VARIABLES.append(
        (f"h_ptbb{thr}_aftercuts",f"ptbb_mbb{thr}_ac","p_{T}(b#bar{b}) [GeV]",(0, 50),(1e2, 10000000000), 10, "d#sigma/dp_{T}(b#bar{b}) [nb/GeV]")
    )
    PDF_RATIO_VARIABLES.append(
        (f"h_ybb{thr}_aftercuts", f"ybb_mbb{thr}_ac", "y_{b#bar{b}}", (-3, 3), (1e2, 10000000000), 10, "d#sigma/dy_{b#bar{b}}")
    )
    PDF_RATIO_VARIABLES.append(
        (f"h_mbb{thr}", f"mbb{thr}", "M_{b#bar{b}} [GeV]", (0, 160), (1e2, 10000000000), 20, "d#sigma/dM_{b#bar{b}} [nb/GeV]")
    )
    PDF_RATIO_VARIABLES.append(
        (f"h_mbb{thr}_aftercuts", f"mbb{thr}_ac", "M_{b#bar{b}} [GeV]", (0, 160), (1e2, 10000000000), 20, "d#sigma/dM_{b#bar{b}} [nb/GeV]")
    )

   

def extract_mbb_threshold(hist_name):
    match = re.match(r"h_(?:mbb|ptbb|ybb)(\d+)(?:_aftercuts)?$", hist_name)
    return float(match.group(1)) if match else None


def fetch_hist(fobj, hist_name, rebin, clone_tag):
    h = fobj.Get(hist_name)
    if not h:
        return None
    h = h.Clone(f"{hist_name}_{clone_tag}")
    h.SetDirectory(0)
    h.Rebin(rebin)
    return h


def _make_band(h, h_up, h_down, col, sty):
    """Constrói TGraphAsymmErrors com incerteza stat (+PDF se fornecido) e suavização dos erros."""
    nbins = h.GetNbinsX()
    
    # Calcular erros antes da suavização
    y_values = []
    ey_up_raw = []
    ey_down_raw = []
    x_values = []
    ex_values = []
    
    for b in range(1, nbins + 1):
        x = h.GetBinCenter(b)
        y = h.GetBinContent(b)
        ex = 0.5 * h.GetBinWidth(b)
        stat_err = h.GetBinError(b)
        pdf_up = abs(h_up.GetBinContent(b) - y) if h_up else 0.0
        pdf_down = abs(y - h_down.GetBinContent(b)) if h_down else 0.0
        ey_up = math.sqrt(stat_err ** 2 + pdf_up ** 2)
        ey_down = math.sqrt(stat_err ** 2 + pdf_down ** 2)
        
        y_values.append(y)
        ey_up_raw.append(ey_up)
        ey_down_raw.append(ey_down)
        x_values.append(x)
        ex_values.append(ex)
    
    # Suavizar erros com média móvel ponderada (kernel: 1, 2, 1)
    ey_up_smooth = []
    ey_down_smooth = []
    for i in range(nbins):
        if y_values[i] == 0.0:  # Bin vazio
            ey_up_smooth.append(0.0)
            ey_down_smooth.append(0.0)
        else:
            # Média móvel ponderada: (1*anterior + 2*atual + 1*próximo) / 4
            up_sum = 0.0
            down_sum = 0.0
            weight_sum = 0.0
            
            for j, w in [(-1, 1.0), (0, 2.0), (1, 1.0)]:
                idx = i + j
                if 0 <= idx < nbins and y_values[idx] > 0.0:
                    up_sum += w * ey_up_raw[idx]
                    down_sum += w * ey_down_raw[idx]
                    weight_sum += w
            
            if weight_sum > 0:
                ey_up_smooth.append(up_sum / weight_sum)
                ey_down_smooth.append(down_sum / weight_sum)
            else:
                ey_up_smooth.append(ey_up_raw[i])
                ey_down_smooth.append(ey_down_raw[i])
    
    # Preencher TGraphAsymmErrors com erros suavizados
    band = root.TGraphAsymmErrors(nbins)
    for b in range(nbins):
        band.SetPoint(b, x_values[b], y_values[b])
        band.SetPointError(b, ex_values[b], ex_values[b], ey_down_smooth[b], ey_up_smooth[b])
    
    band.SetFillColorAlpha(col, 0.22)
    band.SetLineColor(col)
    band.SetLineStyle(sty)
    band.SetLineWidth(3)
    return band


def plot_ratio_with_envelope(file_nuclear, file_baseline, hist_name, plot_key, xlabel, xlim, ylim, rebin, ylabel, outpath):
    fig, (ax, rax) = aplt.ratio_plot(name=f"fig_ratio_{plot_key}", figsize=(800, 800), hspace=0.1)
    mbb_threshold = extract_mbb_threshold(hist_name)

    h_nuclear = fetch_hist(file_nuclear, hist_name, rebin, "nuclear")
    h_baseline = fetch_hist(file_baseline, hist_name, rebin, "baseline")
    if not h_nuclear or not h_baseline:
        print(f"  AVISO: histograma '{hist_name}' ausente para ratio nPDF")
        return

    if h_nuclear.GetSumw2N() == 0:
        h_nuclear.Sumw2()
    if h_baseline.GetSumw2N() == 0:
        h_baseline.Sumw2()

    col_nuclear = root.kRed + 1
    col_baseline = root.kBlue - 5 
    
    h_nuclear.SetLineColor(col_nuclear)
    h_nuclear.SetLineWidth(3)
    h_baseline.SetLineColor(col_baseline)
    h_baseline.SetLineStyle(2)
    h_baseline.SetLineWidth(3)

    h_up = fetch_hist(file_nuclear, f"{hist_name}_pdf_up", rebin, "up")
    h_down = fetch_hist(file_nuclear, f"{hist_name}_pdf_down", rebin, "down")
    use_pdf = h_up is not None and h_down is not None

    band_nuclear = _make_band(h_nuclear, h_up if use_pdf else None, h_down if use_pdf else None, col_nuclear, 1)
    band_baseline = _make_band(h_baseline, None, None, col_baseline, 2)

    # painel superior: banda depois do plot para ficar sobre o frame
    ax.plot(h_nuclear)
    ax.plot(h_baseline)
    band_nuclear.Draw("2 SAME")
    h_nuclear.Draw("HIST SAME")
    band_baseline.Draw("2 SAME")
    h_baseline.Draw("HIST SAME")

    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    ax.frame.GetXaxis().SetLabelSize(0)
    ax.frame.GetXaxis().SetTitleSize(0)
    ax.add_margins(top=0.16)
    

    ax.text(0.20, 0.92, "#bf{Pythia8: Generation level}", size=20, align=13)
    ax.text(0.20, 0.83, "O-O #rightarrow b#bar{b}, #sqrt{s_{NN}} = 5.36 TeV", size=20, align=13)
    ax.text(0.20, 0.75, "#it{OO nuclear / OO baseline}", size=18, align=13)
    
    if "_ac" in plot_key:    
        ax.text(0.20, 0.66, "#it{After selection cuts}", size=18, align=13)
    else:
        ax.text(0.20, 0.66, "#it{Before selection cuts}", size=18, align=13)
    if mbb_threshold is not None:
        ax.text(0.20, 0.57, f"#it{{Selection:}} m_{{b#bar{{b}}}} > {int(mbb_threshold)} GeV", size=18, align=13)

    legend = ax.legend(loc=(0.56, 0.60, 0.90, 0.90), textsize=17)
    legend.AddEntry(band_nuclear, "OO nuclear (EPPS21)", "lf")
    legend.AddEntry(band_baseline, "OO baseline (CT18ANLO)", "lf")

   

    # painel de ratio com banda de incerteza propagada
    ratio = h_nuclear.Clone(f"ratio_{plot_key}")
    ratio.Divide(h_baseline)
    ratio.SetLineColor(root.kBlack)
    ratio.SetLineWidth(3)

    ratio_band = root.TGraphAsymmErrors(h_nuclear.GetNbinsX())
    for b in range(1, h_nuclear.GetNbinsX() + 1):
        x = h_nuclear.GetBinCenter(b)
        y_nuc = h_nuclear.GetBinContent(b)
        y_base = h_baseline.GetBinContent(b)
        ex = 0.5 * h_nuclear.GetBinWidth(b)
        if y_base == 0 or y_nuc == 0:
            ratio_band.SetPoint(b - 1, x, 0)
            ratio_band.SetPointError(b - 1, ex, ex, 0, 0)
            continue
        y_ratio = y_nuc / y_base
        ey_nuc_up = band_nuclear.GetErrorYhigh(b - 1)
        ey_nuc_down = band_nuclear.GetErrorYlow(b - 1)
        ey_base_up = band_baseline.GetErrorYhigh(b - 1)
        ey_base_down = band_baseline.GetErrorYlow(b - 1)
        ey_ratio_up = y_ratio * math.sqrt((ey_nuc_up / y_nuc) ** 2 + (ey_base_down / y_base) ** 2)
        ey_ratio_down = y_ratio * math.sqrt((ey_nuc_down / y_nuc) ** 2 + (ey_base_up / y_base) ** 2)
        ratio_band.SetPoint(b - 1, x, y_ratio)
        ratio_band.SetPointError(b - 1, ex, ex, ey_ratio_down, ey_ratio_up)
    ratio_band.SetFillColorAlpha(root.kViolet, 0.22)
    ratio_band.SetLineColor(root.kBlack)
    ratio_band.SetLineWidth(3)

    rax.plot(ratio, options="HIST")
    ratio_band.Draw("2 SAME")
    ratio.Draw("HIST SAME")

    line = root.TLine(xlim[0], 1.0, xlim[1], 1.0)
    line.SetLineColor(root.kBlack)
    line.SetLineStyle(2)
    line.Draw()

    rax.set_xlim(*xlim)
    rax.set_ylim(0.2, 1.5)
    rax.set_ylabel("R_{OO}", loc="center")
    rax.set_xlabel(xlabel)
    rax.frame.GetXaxis().SetNdivisions(506)
    rax.frame.GetYaxis().SetNdivisions(505)
    rax.frame.GetXaxis().SetTitleOffset(1.01)
    
    if "ptbb_mbb" in plot_key:
        subdir = "ptbb"
    elif "ybb_mbb" in plot_key or "bbbar_y" in plot_key:
        subdir = "ybb"
    elif "mbb" in plot_key:
        subdir = "mbb"
    else:
        subdir = "misc"
        print(f"  AVISO: plot_key '{plot_key}' não corresponde a padrão esperado; salvando em '{subdir}'")

    save_dir = os.path.join(outpath, subdir)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"npdf_epps21_ratio_{plot_key}.pdf"))
    root.gROOT.GetListOfCanvases().Clear()
 
 

def main():
    aplt.set_atlas_style()
    root.gROOT.SetBatch()
    root.gROOT.ForceStyle()

    path = "/media/danbiajujumiguel/T7/HI_analysis/OO_bbbar_analysis/output/"
    file_nuclear = root.TFile(path + "analysis_HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16.root")
    file_baseline = root.TFile(path + "analysis_HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16_pp_reference.root")

    outpath = "/media/danbiajujumiguel/T7/HI_analysis/OO_bbbar_analysis/plots"

    for hist_name, plot_key, xlabel, xlim, ylim, rebin, ylabel in PDF_RATIO_VARIABLES:
        print(f"Plotando ratio nPDF: {plot_key} ({hist_name})")
        plot_ratio_with_envelope(
            file_nuclear,
            file_baseline,
            hist_name,
            plot_key,
            xlabel,
            xlim,
            ylim,
            rebin,
            ylabel,
            outpath,
        )

    file_nuclear.Close()
    file_baseline.Close()
    print("Plots nPDF ratio salvos em:", outpath)


if __name__ == "__main__":
    main()
