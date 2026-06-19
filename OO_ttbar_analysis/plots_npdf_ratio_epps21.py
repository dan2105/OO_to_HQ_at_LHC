import ROOT as root
import atlasplots as aplt
import math


PDF_RATIO_VARIABLES = [
    ("h_mtt", "mtt", "M_{t#bar{t}} [GeV]", (360, 900), (1e-3, 1), 10, "d#sigma/dM_{t#bar{t}} [nb/GeV]"),
    ("h_ttbar_y", "ttbar_y", "y_{t#bar{t}}", (-3, 3), (1e-4, 1), 10, "d#sigma/dy_{t#bar{t}}"),
    ("h_ttbar_pt", "ttbar_pt", "p_{T}(t#bar{t}) [GeV]", (0, 350), (1e-3, 1), 10, "d#sigma/dp_{T}(t#bar{t}) [nb/GeV]"),
    ("h_mtt_aftercuts", "mtt_ac", "M_{t#bar{t}} [GeV]", (360, 900), (1e-3, 1), 10, "d#sigma/dM_{t#bar{t}} [nb/GeV]"),
    ("h_ttbar_y_aftercuts", "ttbar_y_ac", "y_{t#bar{t}}", (-3, 3), (1e-4, 1), 10, "d#sigma/dy_{t#bar{t}}"),
    ("h_ttbar_pt_aftercuts", "ttbar_pt_ac", "p_{T}(t#bar{t}) [GeV]", (0, 350), (1e-3, 1), 10, "d#sigma/dp_{T}(t#bar{t}) [nb/GeV]"),
]


def fetch_hist(fobj, hist_name, rebin, clone_tag):
    h = fobj.Get(hist_name)
    if not h:
        return None
    h = h.Clone(f"{hist_name}_{clone_tag}")
    h.SetDirectory(0)
    h.Rebin(rebin)
    return h


def _make_band(h, h_up, h_down, col, sty):
    """Constrói TGraphAsymmErrors com incerteza stat (+PDF se fornecido)."""
    band = root.TGraphAsymmErrors(h.GetNbinsX())
    for b in range(1, h.GetNbinsX() + 1):
        x = h.GetBinCenter(b)
        y = h.GetBinContent(b)
        ex = 0.5 * h.GetBinWidth(b)
        stat_err = h.GetBinError(b)
        pdf_up = abs(h_up.GetBinContent(b) - y) if h_up else 0.0
        pdf_down = abs(y - h_down.GetBinContent(b)) if h_down else 0.0
        ey_up = math.sqrt(stat_err ** 2 + pdf_up ** 2)
        ey_down = math.sqrt(stat_err ** 2 + pdf_down ** 2)
        band.SetPoint(b - 1, x, y)
        band.SetPointError(b - 1, ex, ex, ey_down, ey_up)
    band.SetFillColorAlpha(col, 0.22)
    band.SetLineColor(col)
    band.SetLineStyle(sty)
    band.SetLineWidth(3)
    return band


def plot_ratio_with_envelope(file_nuclear, file_baseline, hist_name, plot_key, xlabel, xlim, ylim, rebin, ylabel, outpath):
    fig, (ax, rax) = aplt.ratio_plot(name=f"fig_ratio_{plot_key}", figsize=(800, 800), hspace=0.1)

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

    ax.set_yscale("linear")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    ax.frame.GetXaxis().SetLabelSize(0)
    ax.frame.GetXaxis().SetTitleSize(0)
    ax.add_margins(top=0.16)
    

    ax.text(0.20, 0.92, "#bf{Pythia8: Generation level}", size=20, align=13)
    ax.text(0.20, 0.83, "O-O, #sqrt{s_{NN}} = 5.36 TeV", size=20, align=13)
    ax.text(0.20, 0.74, "t#bar{t} #rightarrow l#nubjjb", size=20, align=13)
    ax.text(0.20, 0.65, "#it{OO nuclear / OO baseline}", size=18, align=13)
    
    if "_ac" in plot_key:    
        ax.text(0.20, 0.56, "#it{After selection cuts}", size=18, align=13)
    else:
        ax.text(0.20, 0.56, "#it{Before selection cuts}", size=18, align=13)

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
    rax.set_ylim(0.7, 1.3)
    rax.set_ylabel("R_{OO}", loc="center")
    rax.set_xlabel(xlabel)
    rax.frame.GetXaxis().SetNdivisions(506)
    rax.frame.GetYaxis().SetNdivisions(505)
    rax.frame.GetXaxis().SetTitleOffset(1.01)
    

    fig.savefig(f"{outpath}/npdf_epps21_ratio_{plot_key}.pdf")
    root.gROOT.GetListOfCanvases().Clear()


def main():
    aplt.set_atlas_style()
    root.gROOT.SetBatch()
    root.gROOT.ForceStyle()

    path = "/media/danbiajujumiguel/T7/HI_analysis/OO_ttbar_analysis/output/"
    file_nuclear = root.TFile(path + "analysis_HI_ttbar_OO_EPPS21nlo_CT18Anlo_O16.root")
    file_baseline = root.TFile(path + "analysis_HI_ttbar_OO_EPPS21nlo_CT18Anlo_O16_pp_reference.root")

    outpath = "/media/danbiajujumiguel/T7/HI_analysis/OO_ttbar_analysis/plots"

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
