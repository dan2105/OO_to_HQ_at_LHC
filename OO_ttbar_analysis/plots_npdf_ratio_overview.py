import math
import itertools
import ROOT as root
import atlasplots as aplt


RATIO_OVERVIEW_VARIABLES = [
    ("h_mtt", "mtt", "M_{t#bar{t}} [GeV]", (360, 900), (0.5, 1.5), 10),
    ("h_ttbar_y", "ttbar_y", "y_{t#bar{t}}", (-3, 3), (0.5, 1.5), 10),
    ("h_ttbar_pt", "ttbar_pt", "p_{T}(t#bar{t}) [GeV]", (0, 350), (0.5, 1.5), 10),
]


PDF_CONFIGS = [
    {
        "tag": "EPPS21",
        "label": "EPPS21nlo/CT18ANLO",
        "nuclear": "analysis_HI_ttbar_OO_EPPS21nlo_CT18Anlo_O16.root",
        "baseline": "analysis_HI_ttbar_OO_EPPS21nlo_CT18Anlo_O16_pp_reference.root",
        "color": root.kRed + 1,
        "style": 1,
    },
    {
        "tag": "nCTEQ15HQ",
        "label": "nCTEQ15HQ/CT18NLO",
        "nuclear": "analysis_HI_ttbar_OO_nCTEQ15HQ_FullNucleous_16_8.root",
        "baseline": "analysis_HI_ttbar_OO_nCTEQ15HQ_FullNucleous_16_8_pp_reference.root",
        "color": root.kBlue + 1,
        "style": 2,
    },
    {
        "tag": "TUJU21",
        "label": "TUJU21([A = 16,Z = 8]/[A = 1,Z = 1])",
        "nuclear": "analysis_HI_ttbar_OO_TUJU21_16_8.root",
        "baseline": "analysis_HI_ttbar_OO_TUJU21_16_8_pp_reference.root",
        "color": root.kGreen + 2,
        "style": 3,
    },
]


PAIR_CONFIGS = [
    ("EPPS21", "nCTEQ15HQ"),
    ("EPPS21", "TUJU21"),
    ("nCTEQ15HQ", "TUJU21"),
]


def fetch_hist(fobj, hist_name, rebin, clone_tag):
    h = fobj.Get(hist_name)
    if not h:
        return None
    h = h.Clone(f"{hist_name}_{clone_tag}")
    h.SetDirectory(0)
    h.Rebin(rebin)
    if h.GetSumw2N() == 0:
        h.Sumw2()
    return h


def hist_to_tgraph(h, tag):
    """Converte TH1 em TGraph de pontos centrais (sem erros), para draw smooth 'C'."""
    n = h.GetNbinsX()
    g = root.TGraph(n)
    g.SetName(f"tg_{h.GetName()}_{tag}")
    for b in range(1, n + 1):
        g.SetPoint(b - 1, h.GetBinCenter(b), h.GetBinContent(b))
    return g


def compute_ratio_band(h_nuclear, h_baseline, h_nuclear_up=None, h_nuclear_down=None):
    ratio_h = h_nuclear.Clone(f"ratio_{h_nuclear.GetName()}")
    ratio_h.Divide(h_baseline)

    # TGraph para curva smooth (sem erros)
    ratio_g = hist_to_tgraph(ratio_h, "smooth")

    # TGraphAsymmErrors apenas para a banda de shade (ex=meia-largura de bin)
    ratio_band = root.TGraphAsymmErrors(h_nuclear.GetNbinsX())
    for b in range(1, h_nuclear.GetNbinsX() + 1):
        x = h_nuclear.GetBinCenter(b)
        ex = 0.0

        y_nuc = h_nuclear.GetBinContent(b)
        y_base = h_baseline.GetBinContent(b)
        if y_nuc <= 0.0 or y_base <= 0.0:
            ratio_band.SetPoint(b - 1, x, 0.0)
            ratio_band.SetPointError(b - 1, ex, ex, 0.0, 0.0)
            continue

        y_ratio = y_nuc / y_base

        stat_nuc = h_nuclear.GetBinError(b)
        stat_base = h_baseline.GetBinError(b)

        pdf_nuc_up = abs(h_nuclear_up.GetBinContent(b) - y_nuc) if h_nuclear_up else 0.0
        pdf_nuc_down = abs(y_nuc - h_nuclear_down.GetBinContent(b)) if h_nuclear_down else 0.0

        rel_up = math.sqrt(((stat_nuc ** 2 + pdf_nuc_up ** 2) / (y_nuc ** 2)) + (stat_base ** 2 / (y_base ** 2)))
        rel_down = math.sqrt(((stat_nuc ** 2 + pdf_nuc_down ** 2) / (y_nuc ** 2)) + (stat_base ** 2 / (y_base ** 2)))

        ratio_band.SetPoint(b - 1, x, y_ratio)
        ratio_band.SetPointError(b - 1, ex, ex, y_ratio * rel_down, y_ratio * rel_up)

    return ratio_h, ratio_g, ratio_band


def plot_ratio_pair(config_subset, hist_name, plot_key, xlabel, xlim, ylim, rebin, outpath):
    pair_tag = "_vs_".join(cfg["tag"] for cfg in config_subset)
    fig, ax = aplt.subplots(1, 1, name=f"fig_ratio_overview_{plot_key}_{pair_tag}", figsize=(800, 700))
    keep = []
    entries = []

    for cfg in config_subset:
        h_nuclear = fetch_hist(cfg["f_nuclear"], hist_name, rebin, f"{cfg['tag']}_nuclear")
        h_baseline = fetch_hist(cfg["f_baseline"], hist_name, rebin, f"{cfg['tag']}_baseline")
        if not h_nuclear or not h_baseline:
            print(f"  AVISO: histograma '{hist_name}' ausente para {cfg['tag']}")
            continue

        h_up = fetch_hist(cfg["f_nuclear"], f"{hist_name}_pdf_up", rebin, f"{cfg['tag']}_up")
        h_down = fetch_hist(cfg["f_nuclear"], f"{hist_name}_pdf_down", rebin, f"{cfg['tag']}_down")

        ratio_h, ratio_g, ratio_band = compute_ratio_band(h_nuclear, h_baseline, h_up, h_down)

        ratio_g.SetLineColor(cfg["color"])
        ratio_g.SetLineStyle(cfg["style"])
        ratio_g.SetLineWidth(3)
        ratio_g.SetMarkerSize(0)

        ratio_band.SetFillColorAlpha(cfg["color"], 0.22)
        ratio_band.SetLineColorAlpha(cfg["color"], 0)
        ratio_band.SetLineWidth(0)
        ratio_band.SetMarkerSize(0)
        cfg["ratio_band"] = ratio_band

        entries.append((cfg, ratio_g, ratio_band, ratio_h))
        keep.extend([h_nuclear, h_baseline, h_up, h_down, ratio_h, ratio_g, ratio_band])

    if not entries:
        print(f"  AVISO: nenhum ratio foi desenhado para {hist_name} em {pair_tag}")
        root.gROOT.GetListOfCanvases().Clear()
        return

    first_ratio_h = entries[0][3]
    first_ratio_h.SetLineColorAlpha(root.kWhite, 0)
    first_ratio_h.SetMarkerSize(0)
    ax.plot(first_ratio_h, options="AXIS")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_ylabel("R_{OO} = nuclear / baseline")
    ax.set_xlabel(xlabel)
    ax.add_margins(top=0.16)

    for _cfg, _ratio_g, ratio_band, _ratio_h in entries:
        ratio_band.Draw("3 SAME")

    for _cfg, ratio_g, _ratio_band, _ratio_h in entries:
        ratio_g.Draw("C SAME")

    line = root.TLine(xlim[0], 1.0, xlim[1], 1.0)
    line.SetLineColor(root.kBlack)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw()
    keep.append(line)

    ax.text(0.20, 0.90, "#bf{Pythia8: Generation level}", size=20, align=13)
    ax.text(0.20, 0.82, "O-O, #sqrt{s_{NN}} = 5.36 TeV", size=20, align=13)
    ax.text(0.20, 0.74, "t#bar{t} #rightarrow l#nubjjb", size=20, align=13)
    ax.text(0.20, 0.66, "#it{Nuclear modification factor}", size=18, align=13)

    legend = ax.legend(loc=(0.56, 0.68, 0.90, 0.90), textsize=16)
    for cfg, _ratio_g, ratio_band, _ratio_h in entries:
        legend.AddEntry(ratio_band, cfg["label"], "lf")
    keep.append(legend)

    fig.savefig(f"{outpath}/npdf_ratio_overview_{plot_key}_{pair_tag}.pdf")
    root.gROOT.GetListOfCanvases().Clear()


def plot_ratio_overview(config_data, hist_name, plot_key, xlabel, xlim, ylim, rebin, outpath):
    fig, ax = aplt.subplots(1, 1, name=f"fig_ratio_overview_{plot_key}", figsize=(800, 700))

    drawn_any = False
    objects_to_keep = []

    for cfg in config_data:
        h_nuclear = fetch_hist(cfg["f_nuclear"], hist_name, rebin, f"{cfg['tag']}_nuclear")
        h_baseline = fetch_hist(cfg["f_baseline"], hist_name, rebin, f"{cfg['tag']}_baseline")
        if not h_nuclear or not h_baseline:
            print(f"  AVISO: histograma '{hist_name}' ausente para {cfg['tag']}")
            continue

        h_up = fetch_hist(cfg["f_nuclear"], f"{hist_name}_pdf_up", rebin, f"{cfg['tag']}_up")
        h_down = fetch_hist(cfg["f_nuclear"], f"{hist_name}_pdf_down", rebin, f"{cfg['tag']}_down")

        ratio, ratio_band = compute_ratio_band(h_nuclear, h_baseline, h_up, h_down)

        ratio.SetLineColor(cfg["color"])
        ratio.SetLineStyle(cfg["style"])
        ratio.SetLineWidth(3)

        ratio_band.SetFillColorAlpha(cfg["color"], 0.20)
        ratio_band.SetLineColor(cfg["color"])
        ratio_band.SetLineStyle(cfg["style"])
        ratio_band.SetLineWidth(2)

        if not drawn_any:
            ax.plot(ratio, options="C")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_ylabel("R_{OO} = nuclear / baseline")
            ax.set_xlabel(xlabel)
            ax.add_margins(top=0.16)
            drawn_any = True

        ratio_band.Draw("2 SAME")
        ratio.Draw("HIST SAME")

        objects_to_keep.extend([h_nuclear, h_baseline, h_up, h_down, ratio, ratio_band])

    if not drawn_any:
        print(f"  AVISO: nenhum ratio foi desenhado para {hist_name}")
        root.gROOT.GetListOfCanvases().Clear()
        return

    line = root.TLine(xlim[0], 1.0, xlim[1], 1.0)
    line.SetLineColor(root.kBlack)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw()

    ax.text(0.20, 0.90, "#bf{Pythia8: Generation level}", size=20, align=13)
    ax.text(0.20, 0.82, "O-O, #sqrt{s_{NN}} = 5.36 TeV", size=20, align=13)
    ax.text(0.20, 0.74, "t#bar{t} #rightarrow l#nubjjb", size=20, align=13)
    ax.text(0.20, 0.66, "#it{Nuclear modification factor overview}", size=18, align=13)

    legend = ax.legend(loc=(0.56, 0.62, 0.90, 0.90), textsize=16)
    for cfg in config_data:
        if cfg.get("ratio_band") is not None:
            legend.AddEntry(cfg["ratio_band"], cfg["label"], "lf")

    fig.savefig(f"{outpath}/npdf_ratio_overview_{plot_key}.pdf")
    root.gROOT.GetListOfCanvases().Clear()



def main():
    aplt.set_atlas_style()
    root.gROOT.SetBatch()
    root.gROOT.ForceStyle()

    path = "/media/danbiajujumiguel/T7/HI_analysis/OO_ttbar_analysis/output/"
    outpath = "/media/danbiajujumiguel/T7/HI_analysis/OO_ttbar_analysis/plots"

    config_data = []
    for cfg in PDF_CONFIGS:
        f_nuclear = root.TFile(path + cfg["nuclear"])
        f_baseline = root.TFile(path + cfg["baseline"])
        if not f_nuclear or f_nuclear.IsZombie() or not f_baseline or f_baseline.IsZombie():
            print(f"AVISO: nao foi possivel abrir arquivos para {cfg['tag']}")
            continue

        cfg_runtime = dict(cfg)
        cfg_runtime["f_nuclear"] = f_nuclear
        cfg_runtime["f_baseline"] = f_baseline
        cfg_runtime["ratio_band"] = None
        config_data.append(cfg_runtime)

    if not config_data:
        print("ERRO: nenhum conjunto PDF foi carregado.")
        return

    config_lookup = {cfg["tag"]: cfg for cfg in config_data}

    for hist_name, plot_key, xlabel, xlim, ylim, rebin in RATIO_OVERVIEW_VARIABLES:
        print(f"Plotando ratios em pares: {plot_key} ({hist_name})")
        for tag_a, tag_b in PAIR_CONFIGS:
            subset = [config_lookup[tag] for tag in (tag_a, tag_b) if tag in config_lookup]
            if len(subset) != 2:
                print(f"  AVISO: par {tag_a}/{tag_b} incompleto")
                continue
            plot_ratio_pair(subset, hist_name, plot_key, xlabel, xlim, ylim, rebin, outpath)

    for cfg in config_data:
        cfg["f_nuclear"].Close()
        cfg["f_baseline"].Close()

    print("Plots de ratio overview salvos em:", outpath)


if __name__ == "__main__":
    main()
