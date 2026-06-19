import math
import itertools
import ROOT as root
import atlasplots as aplt

MASS_CUTS = [10, 30, 40]  # GeV

def build_ratio_overview_variables(mass_cuts):
    variables = []
    for cut in mass_cuts:
        variables.extend([
            (f"h_ptbb{cut}_aftercuts", f"ptbb_mbb{cut}", "p_{T}(t#bar{t}) [GeV]", (10, 30), (0.1, 1.5), 20),
            (f"h_ybb{cut}_aftercuts", f"ybb_mbb{cut}", "y(b#bar{b})", (-2.5, 2.5), (0.1, 1.5), 10),
        ])
    return variables


RATIO_OVERVIEW_VARIABLES = build_ratio_overview_variables(MASS_CUTS)
#colocar o corte da massa do par no nome do histograma, para facilitar a comparação entre os diferentes cortes de massa (10, 15, 20, 25, 30, 40 GeV) 
#quando chegar em casa
PDF_CONFIGS = [
    {
        "tag": "EPPS21",
        "label": "EPPS21nlo/CT18ANLO",
        "nuclear": "analysis_HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16.root",
        "baseline": "analysis_HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16_pp_reference.root",
        "color": root.kRed + 1,
        "style": 1,
    },
    {
        "tag": "nCTEQ15HQ",
        "label": "nCTEQ15HQ/CT18NLO",
        "nuclear": "analysis_HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8.root",
        "baseline": "analysis_HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8_pp_reference.root",
        "color": root.kBlue + 1,
        "style": 2,
    },
    {
        "tag": "TUJU21",
        "label": "TUJU21([A = 16,Z = 8]/[A = 1,Z = 1])",
        "nuclear": "analysis_HI_bbbar_OO_TUJU21_16_8.root",
        "baseline": "analysis_HI_bbbar_OO_TUJU21_16_8_pp_reference.root",
        "color": root.kGreen + 2,
        "style": 3,
    },
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
    """Converte TH1 em TGraph usando apenas bins válidos (y > 0 e finito)."""
    g = root.TGraph()
    g.SetName(f"tg_{h.GetName()}_{tag}")
    ip = 0
    for b in range(1, h.GetNbinsX() + 1):
        y = h.GetBinContent(b)
        if y <= 0.0 or not math.isfinite(y):
            continue
        g.SetPoint(ip, h.GetBinCenter(b), y)
        ip += 1
    return g


def compute_ratio_band(h_nuclear, h_baseline, h_nuclear_up=None, h_nuclear_down=None):
    ratio_h = h_nuclear.Clone(f"ratio_{h_nuclear.GetName()}")
    ratio_h.Reset("ICES")

    # TGraphAsymmErrors apenas para a banda de shade (ex=meia-largura de bin)
    ratio_band = root.TGraphAsymmErrors(h_nuclear.GetNbinsX())
    
    # Armazenar erros antes da suavização para cálculo
    nbins = h_nuclear.GetNbinsX()
    y_values = []
    rel_up_raw = []
    rel_down_raw = []
    x_values = []
    
    for b in range(1, nbins + 1):
        x = h_nuclear.GetBinCenter(b)
        ex = 0.0

        y_nuc = h_nuclear.GetBinContent(b)
        y_base = h_baseline.GetBinContent(b)
        if y_nuc <= 0.0 or y_base <= 0.0:
            ratio_h.SetBinContent(b, 0.0)
            ratio_h.SetBinError(b, 0.0)
            ratio_band.SetPoint(b - 1, x, 0.0)
            ratio_band.SetPointError(b - 1, ex, ex, 0.0, 0.0)
            y_values.append(0.0)
            rel_up_raw.append(0.0)
            rel_down_raw.append(0.0)
            x_values.append(x)
            continue

        y_ratio = y_nuc / y_base
        if not math.isfinite(y_ratio):
            y_ratio = 0.0
        ratio_h.SetBinContent(b, y_ratio)
        ratio_h.SetBinError(b, 0.0)

        stat_nuc = h_nuclear.GetBinError(b)
        stat_base = h_baseline.GetBinError(b)

        pdf_nuc_up = abs(h_nuclear_up.GetBinContent(b) - y_nuc) if h_nuclear_up else 0.0
        pdf_nuc_down = abs(y_nuc - h_nuclear_down.GetBinContent(b)) if h_nuclear_down else 0.0

        rel_up = math.sqrt(((stat_nuc ** 2 + pdf_nuc_up ** 2) / (y_nuc ** 2)) + (stat_base ** 2 / (y_base ** 2)))
        rel_down = math.sqrt(((stat_nuc ** 2 + pdf_nuc_down ** 2) / (y_nuc ** 2)) + (stat_base ** 2 / (y_base ** 2)))

        if not math.isfinite(rel_up):
            rel_up = 0.0
        if not math.isfinite(rel_down):
            rel_down = 0.0

        y_values.append(y_ratio)
        rel_up_raw.append(rel_up)
        rel_down_raw.append(rel_down)
        x_values.append(x)

    # Suavizar erros com média móvel ponderada (kernel: 1, 2, 1)
    rel_up_smooth = []
    rel_down_smooth = []
    for i in range(nbins):
        if rel_up_raw[i] == 0.0:  # Bin vazio
            rel_up_smooth.append(0.0)
            rel_down_smooth.append(0.0)
        else:
            # Média móvel ponderada: (1*anterior + 2*atual + 1*próximo) / 4
            up_sum = 0.0
            down_sum = 0.0
            weight_sum = 0.0
            
            for j, w in [(-1, 1.0), (0, 2.0), (1, 1.0)]:
                idx = i + j
                if 0 <= idx < nbins and rel_up_raw[idx] > 0.0:
                    up_sum += w * rel_up_raw[idx]
                    down_sum += w * rel_down_raw[idx]
                    weight_sum += w
            
            if weight_sum > 0:
                rel_up_smooth.append(up_sum / weight_sum)
                rel_down_smooth.append(down_sum / weight_sum)
            else:
                rel_up_smooth.append(rel_up_raw[i])
                rel_down_smooth.append(rel_down_raw[i])

    # Preencher TGraphAsymmErrors com erros suavizados
    for b in range(nbins):
        x = x_values[b]
        y = y_values[b]
        if y > 0.0:
            ratio_band.SetPoint(b, x, y)
            ratio_band.SetPointError(b, 0.0, 0.0, y * rel_down_smooth[b], y * rel_up_smooth[b])

    # TGraph para curva de razão, somente com bins válidos.
    ratio_g = hist_to_tgraph(ratio_h, "smooth")

    return ratio_h, ratio_g, ratio_band

def plot_ratio_overview(config_data, hist_name, plot_key, xlabel, xlim, ylim, rebin, outpath):
    fig, ax = aplt.subplots(1, 1, name=f"fig_ratio_overview_{plot_key}", figsize=(800, 700))

    drawn_any = False
    objects_to_keep = []
    entries = []

    for cfg in config_data:
        cfg["ratio_band"] = None

    for cfg in config_data:
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

        ratio_band.SetFillColorAlpha(cfg["color"], 0.20)
        ratio_band.SetLineColorAlpha(cfg["color"], 0)
        ratio_band.SetLineWidth(0)
        ratio_band.SetMarkerSize(0)
        cfg["ratio_band"] = ratio_band

        if not drawn_any:
            ratio_h.SetLineColorAlpha(root.kWhite, 0)
            ratio_h.SetMarkerSize(0)
            ax.plot(ratio_h, options="AXIS")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_ylabel("R_{OO} = nuclear / baseline")
            ax.set_xlabel(xlabel)
            ax.add_margins(top=0.16)
            drawn_any = True

        entries.append((ratio_band, ratio_g))
        objects_to_keep.extend([h_nuclear, h_baseline, h_up, h_down, ratio_h, ratio_g, ratio_band])

    if not drawn_any:
        print(f"  AVISO: nenhum ratio foi desenhado para {hist_name}")
        root.gROOT.GetListOfCanvases().Clear()
        return

    for ratio_band, _ratio_g in entries:
        ratio_band.Draw("3 SAME")

    for _ratio_band, ratio_g in entries:
        ratio_g.Draw("L SAME")

    line = root.TLine(xlim[0], 1.0, xlim[1], 1.0)
    line.SetLineColor(root.kBlack)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw()

    ax.text(0.20, 0.90, "#bf{Pythia8: Generation level}", size=20, align=13)
    ax.text(0.20, 0.82, "O-O, #sqrt{s_{NN}} = 5.36 TeV", size=20, align=13)
    ax.text(0.20, 0.76, "#it{Nuclear modification factor overview}", size=18, align=13)

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

    path = "/media/danbiajujumiguel/T7/HI_analysis/OO_bbbar_analysis/output/"
    outpath = "/media/danbiajujumiguel/T7/HI_analysis/OO_bbbar_analysis/plots"

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


# Plootar overview geral (todos os PDFs juntos)

    for hist_name, plot_key, xlabel, xlim, ylim, rebin in RATIO_OVERVIEW_VARIABLES:
        print(f"Plotando overview geral: {plot_key} ({hist_name})")
        plot_ratio_overview(config_data, hist_name, plot_key, xlabel, xlim, ylim, rebin, outpath)

    for cfg in config_data:
        cfg["f_nuclear"].Close()
        cfg["f_baseline"].Close()

    print("Plots de ratio overview salvos em:", outpath)


if __name__ == "__main__":
    main()
