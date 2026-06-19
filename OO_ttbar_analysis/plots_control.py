import ROOT as root
import atlasplots as aplt
import math


# (hist_name, plot_key, xlabel, xlim, ylim, rebin, ylabel)
CONTROL_VARIABLES = [
    ("h_bestjet1_pt", "bestjet1pt", "p_{T}(j_{1}) [GeV]", (0, 200), (1e-3, 10), 5, "d#sigma/dp_{T}(j_{1}) [nb/GeV]"),
    ("h_bestjet2_pt", "bestjet2pt", "p_{T}(j_{2}) [GeV]", (0, 200), (1e-3, 10), 5, "d#sigma/dp_{T}(j_{2}) [nb/GeV]"),
    ("h_bestbjet_pt", "bestbjetpt", "p_{T}(b) [GeV]", (0, 150), (1e-3, 15), 5, "d#sigma/dp_{T}(b) [nb/GeV]"),
    ("h_lep_pt_reco", "lepptreco", "p_{T}(l) [GeV]", (0, 200), (1e-3, 3), 5, "d#sigma/dp_{T}(l) [nb/GeV]"),
    ("h_reco_top_mass", "recohadtopmass", "M_{jjb} [GeV]", (100, 250), (1e-3, 8), 10, "d#sigma/dM_{jjb} [nb/GeV]"),
    ("h_reco_top_lep_mass", "recoleptopmass", "M_{l#nub} [GeV]", (100, 250), (1e-3, 1), 10, "d#sigma/dM_{l#nub} [nb/GeV]"),
    ("h_lep_pt_aftercuts", "lep_pt_ac", "p_{T}(l) [GeV]", (0, 250), (1e-2, 1e2), 4, "d#sigma/dp_{T}(l) [nb/GeV]"),
    ("h_bestjet1_pt_aftercuts", "bestjet1pt_ac", "p_{T}(j_{1}) [GeV]", (0, 250), (1e-2, 1e2), 4, "d#sigma/dp_{T}(j_{1}) [nb/GeV]"),
    ("h_bestjet2_pt_aftercuts", "bestjet2pt_ac", "p_{T}(j_{2}) [GeV]", (0, 250), (1e-2, 1e2), 4, "d#sigma/dp_{T}(j_{2}) [nb/GeV]"),
    ("h_bestbjet_pt_aftercuts", "bestbjetpt_ac", "p_{T}(b) [GeV]", (0, 150), (1e-2, 1e2), 4, "d#sigma/dp_{T}(b) [nb/GeV]"),
    ("h_reco_top_mass_aftercuts", "recohadtopmass_ac", "M_{jjb} [GeV]", (100, 250), (1e-2, 1e2), 2, "d#sigma/dM_{jjb} [nb/GeV]"),
    ("h_reco_top_lep_mass_aftercuts", "recoleptopmass_ac", "M_{l#nub} [GeV]", (100, 250), (1e-2, 1e2), 2, "d#sigma/dM_{l#nub} [nb/GeV]"),
]


def plot_control(files, labels, colors, styles, hist_name, plot_key, xlabel, xlim, ylim, rebin, ylabel, outpath):
    canvas = root.TCanvas(f"c_{plot_key}", f"c_{plot_key}", 800, 700)
    canvas.SetFillColor(root.kWhite)
    canvas.SetFrameFillColor(root.kWhite)
    canvas.SetTicks(1, 1)

    hists = []
    uncertainty_bands = []
    for i, (fobj, col, sty) in enumerate(zip(files, colors, styles)):
        h = fobj.Get(hist_name)
        if not h:
            print(f"  AVISO: histograma '{hist_name}' nao encontrado em {fobj.GetName()}")
            canvas.Close()
            return

        h = h.Clone(f"{hist_name}_ctrl_{i}")
        h.SetDirectory(0)
        h.Rebin(rebin)
        if h.GetSumw2N() == 0:
            h.Sumw2()
        h.SetLineColor(col)
        h.SetLineStyle(sty)
        h.SetLineWidth(3)

        up_name = f"{hist_name}_pdf_up"
        down_name = f"{hist_name}_pdf_down"
        h_up = fobj.Get(up_name)
        h_down = fobj.Get(down_name)
        if h_up and h_down:
            h_up = h_up.Clone(f"{up_name}_ctrl_{i}")
            h_down = h_down.Clone(f"{down_name}_ctrl_{i}")
            h_up.SetDirectory(0)
            h_down.SetDirectory(0)
            h_up.Rebin(rebin)
            h_down.Rebin(rebin)
            use_pdf = True
        else:
            h_up = None
            h_down = None
            use_pdf = False

        band = root.TGraphAsymmErrors(h.GetNbinsX())
        for b in range(1, h.GetNbinsX() + 1):
            x = h.GetBinCenter(b)
            y = h.GetBinContent(b)
            ex = 0.5 * h.GetBinWidth(b)
            stat_err = h.GetBinError(b)

            if use_pdf:
                pdf_up = abs(h_up.GetBinContent(b) - y)
                pdf_down = abs(y - h_down.GetBinContent(b))
            else:
                pdf_up = 0.0
                pdf_down = 0.0

            ey_up = math.sqrt(stat_err * stat_err + pdf_up * pdf_up)
            ey_down = math.sqrt(stat_err * stat_err + pdf_down * pdf_down)

            band.SetPoint(b - 1, x, y)
            band.SetPointError(b - 1, ex, ex, ey_down, ey_up)

        band.SetFillColorAlpha(col, 0.22)
        band.SetLineColor(col)
        band.SetLineStyle(sty)
        band.SetLineWidth(3)
        uncertainty_bands.append(band)
        
        hists.append(h)

    h0 = hists[0]
    h0.SetTitle("")
    h0.GetXaxis().SetTitle(xlabel)
    h0.GetYaxis().SetTitle(ylabel)
    h0.GetXaxis().SetRangeUser(xlim[0], xlim[1])
    h0.GetXaxis().SetTitleColor(root.kBlack)
    h0.GetXaxis().SetLabelColor(root.kBlack)
    h0.GetYaxis().SetTitleColor(root.kBlack)
    h0.GetYaxis().SetLabelColor(root.kBlack)
    h0.SetMinimum(ylim[0])
    h0.SetMaximum(ylim[1])
    
    
    h0.Draw("HIST")
    for h in hists[1:]:
        h.Draw("HIST SAME")
    for band in uncertainty_bands:
        band.Draw("2 SAME")
    h0.Draw("HIST SAME")
    for h in hists[1:]:
        h.Draw("HIST SAME")
    
    
    
    latex = root.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextSize(0.043)
    latex.SetTextColor(root.kBlack)
    latex.DrawLatex(0.20, 0.90, "#bf{Pythia8: Generation level}")
    latex.DrawLatex(0.20, 0.84, "O-O, #sqrt{s_{NN}} = 5.36 TeV")
    latex.DrawLatex(0.20, 0.78, "t#bar{t} #rightarrow l#nubjjb")
    if "_ac" in plot_key:
        latex.DrawLatex(0.20, 0.72, "#it{After selection cuts}")

    legend = root.TLegend(0.58, 0.70, 0.90, 0.90)
    legend.SetBorderSize(0)
    legend.SetFillStyle(1001)
    legend.SetFillColor(root.kWhite)
    legend.SetLineColor(root.kWhite)
    legend.SetTextFont(42)
    legend.SetTextSize(0.036)
    legend.SetTextColor(root.kBlack)
    for idx, lab in enumerate(labels):
        legend.AddEntry(uncertainty_bands[idx], lab, "lf")
    legend.Draw()

    canvas.SaveAs(f"{outpath}/control_{plot_key}.pdf")
    canvas.Close()


def main():
    aplt.set_atlas_style()
    root.gStyle.SetTextColor(root.kBlack)
    root.gStyle.SetTitleTextColor(root.kBlack)
    root.gStyle.SetLabelColor(root.kBlack, "XYZ")
    root.gStyle.SetTitleColor(root.kBlack, "XYZ")
    root.gROOT.SetBatch()
    root.gROOT.ForceStyle()

    path = "/media/danbiajujumiguel/T7/HI_analysis/OO_ttbar_analysis/output/"
    files = [
        root.TFile(path + "analysis_HI_ttbar_OO_EPPS21nlo_CT18Anlo_O16.root"),
        root.TFile(path + "analysis_HI_ttbar_OO_EPPS21nlo_CT18Anlo_O16_pp_reference.root"),
    ]

    labels = [
        "t#bar{t} (EPPS21NLO)",
        "t#bar{t} (CT18ANLO)",
    ]

    colors = [root.TColor.GetColor("#E74C3C"), root.TColor.GetColor("#3498DB")]
    styles = [1, 2]

    outpath = "/media/danbiajujumiguel/T7/HI_analysis/OO_ttbar_analysis/plots"

    for hist_name, plot_key, xlabel, xlim, ylim, rebin, ylabel in CONTROL_VARIABLES:
        print(f"Plotando controle: {plot_key} ({hist_name})")
        plot_control(files, labels, colors, styles, hist_name, plot_key, xlabel, xlim, ylim, rebin, ylabel, outpath)

    for fobj in files:
        fobj.Close()

    print("Plots de controle salvos em:", outpath)


if __name__ == "__main__":
    main()
