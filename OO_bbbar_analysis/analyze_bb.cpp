/*
 * Análise de produção de bb̄ em colisões O-O
 * Aplica pressupostos experimentais inspirados no fluxo de ttbar:
 * - reconstrução de jatos com FastJet
 * - classificação de b-jets por matching em DeltaR com b-quarks de verdade
 * - resposta simplificada de detector (smearing e eficiência)
 * - cutflow ponderado com incerteza estatística
 *
 * Compilar: g++ -o analyze_bb analyze_bb.cpp `root-config --cflags --libs` `fastjet-config --cxxflags --libs --plugins`
 * Executar: ./analyze_bb input.root output.root [cross_section_nb] [txt_name]
 */

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/PseudoJet.hh>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

struct HessianUncertainty
{
  double up = 0.0;
  double down = 0.0;
};

// Função auxiliar para remover espaços em branco no início e fim de string
std::string trim(const std::string &str)
{
  size_t first = str.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return "";
  size_t last = str.find_last_not_of(" \t\r\n");
  return str.substr(first, (last - first + 1));
}

std::vector<TH1F *> createPdfMemberHistograms(const TH1F *templateHist, const std::string &baseName, int nMembers)
{
  std::vector<TH1F *> histograms;
  histograms.reserve(static_cast<size_t>(nMembers));
  for (int iMember = 0; iMember < nMembers; ++iMember)
  {
    TH1F *hist = static_cast<TH1F *>(templateHist->Clone((baseName + "_pdfMember" + std::to_string(iMember)).c_str()));
    hist->Reset();
    hist->SetDirectory(nullptr);
    histograms.push_back(hist);
  }
  return histograms;
}

HessianUncertainty computeHessianUncertainty(const std::vector<double> &memberValues)
{
  HessianUncertainty unc;
  if (memberValues.size() < 3)
    return unc;

  const double central = memberValues[0];
  double up2 = 0.0;
  double down2 = 0.0;

  for (size_t i = 1; i + 1 < memberValues.size(); i += 2)
  {
    const double deltaPlus = memberValues[i] - central;
    const double deltaMinus = memberValues[i + 1] - central;
    const double upVar = std::max({deltaPlus, deltaMinus, 0.0});
    const double downVar = std::max({-deltaPlus, -deltaMinus, 0.0});
    up2 += upVar * upVar;
    down2 += downVar * downVar;
  }

  // EPPS21/nCTEQ Hessian sets are often provided at 90% CL; convert to ~68% CL.
  constexpr double kHessianCLScale = 1.0 / 1.645;
  unc.up = std::sqrt(up2) * kHessianCLScale;
  unc.down = std::sqrt(down2) * kHessianCLScale;
  return unc;
}

std::pair<TH1F *, TH1F *> buildPdfVariationHistograms(const std::vector<TH1F *> &memberHistograms,
                                                      const std::string &upName,
                                                      const std::string &downName)
{
  if (memberHistograms.empty())
    return {nullptr, nullptr};

  TH1F *upHist = static_cast<TH1F *>(memberHistograms[0]->Clone(upName.c_str()));
  TH1F *downHist = static_cast<TH1F *>(memberHistograms[0]->Clone(downName.c_str()));
  upHist->SetDirectory(nullptr);
  downHist->SetDirectory(nullptr);

  for (int iBin = 1; iBin <= memberHistograms[0]->GetNbinsX(); ++iBin)
  {
    std::vector<double> memberValues;
    memberValues.reserve(memberHistograms.size());
    for (const auto *hist : memberHistograms)
      memberValues.push_back(hist->GetBinContent(iBin));

    const double central = memberValues[0];
    const HessianUncertainty unc = computeHessianUncertainty(memberValues);
    upHist->SetBinContent(iBin, central + unc.up);
    downHist->SetBinContent(iBin, std::max(0.0, central - unc.down));
    upHist->SetBinError(iBin, 0.0);
    downHist->SetBinError(iBin, 0.0);
  }

  return {upHist, downHist};
}

double jetEnergyScale(double eta)
{
  return (std::abs(eta) < 2.5) ? 1.00 : 1.02;
}

double jetEnergyResolution(double eta)
{
  return (std::abs(eta) < 2.5) ? 0.10 : 0.15;
}

fastjet::PseudoJet smearJet(const fastjet::PseudoJet &jet, TRandom3 &rng)
{
  const double scale = jetEnergyScale(jet.eta());
  const double sigma = jetEnergyResolution(jet.eta());
  const double factor = rng.Gaus(scale, sigma);

  fastjet::PseudoJet smeared(jet.px() * factor,
                             jet.py() * factor,
                             jet.pz() * factor,
                             jet.E() * factor);
  smeared.set_user_index(jet.user_index());
  return smeared;
}

std::vector<TLorentzVector> collectBQuarks(const std::vector<int> &pdgID,
                                           const std::vector<int> &status,
                                           const std::vector<float> &px,
                                           const std::vector<float> &py,
                                           const std::vector<float> &pz,
                                           const std::vector<float> &e,
                                           bool isHepMC3Format)
{
  std::vector<TLorentzVector> b_quarks;
  for (size_t iPart = 0; iPart < pdgID.size(); ++iPart)
  {
    const bool isBottom = (std::abs(pdgID[iPart]) == 5);
    // Em HepMC3 usamos b-quarks de status 2 (parton-level pós-hard-process);
    // em LHE legado mantemos status 1/23 para preservar o comportamento anterior.
    const bool isValidStatus = isHepMC3Format ? (status[iPart] == 2 || status[iPart] == 23) : (status[iPart] == 1 || status[iPart] == 23);
    if (isBottom && isValidStatus)
    {
      TLorentzVector b_quark;
      b_quark.SetPxPyPzE(px[iPart], py[iPart], pz[iPart], e[iPart]);
      b_quarks.push_back(b_quark);
    }
  }
  return b_quarks;
}

std::vector<bool> tagBJetsByDeltaR(const std::vector<fastjet::PseudoJet> &jets,
                                   const std::vector<TLorentzVector> &b_quarks,
                                   int &n_bjets)
{
  std::vector<bool> is_bjet(jets.size(), false);
  n_bjets = 0;

  for (size_t iJet = 0; iJet < jets.size(); ++iJet)
  {
    const double jet_eta = jets[iJet].eta();
    const double jet_phi = jets[iJet].phi();

    for (const auto &b_quark : b_quarks)
    {
      const double deta = jet_eta - b_quark.Eta();
      double dphi = jet_phi - b_quark.Phi();
      while (dphi > M_PI)
        dphi -= 2 * M_PI;
      while (dphi < -M_PI)
        dphi += 2 * M_PI;

      const double deltaR = std::sqrt(deta * deta + dphi * dphi);
      if (deltaR < 0.4)
      {
        is_bjet[iJet] = true;
        ++n_bjets;
        break;
      }
    }
  }

  return is_bjet;
}

// Contadores
double weight = 0.0;
double xs = 0.0; // nb isospin nPDFSet0
// double xs = 23.4; // nb isospin nPDFSet3
// double xs = 23.4; // nb isospin nCTeq16_8
double lumi = 8.0; // nb-1

int main(int argc, char **argv)
{

  if (argc < 3)
  {
    std::cout << "Uso: " << argv[0] << " <input.root> <output.root> [cross_section_nb] [txt_name] [pdfweights.csv] [pdf_member]" << std::endl;
    std::cout << "  cross_section_nb: seção de choque em nb (opcional)" << std::endl;
    std::cout << "  pdfweights.csv: arquivo CSV com pesos por evento (opcional)" << std::endl;
    std::cout << "  pdf_member: indice do membro Hessiano (ex.: 0..106, opcional)" << std::endl;
    return 1;
  }

  std::string output_name = argv[2];

  // Seção de choque fornecida como argumento (opcional)
  if (argc >= 4)
  {
    xs = std::atof(argv[3]);
    std::cout << "Seção de choque fornecida: " << xs << " nb" << std::endl;
  }
  else
  {
    std::cout << "Seção de choque padrão: " << xs << " nb" << std::endl;
  }

  std::string txt_name = (argc >= 5) ? argv[4] : "out_bbbar.txt";
  std::string pdf_weights_csv = (argc >= 6) ? argv[5] : "";
  int pdf_member = (argc >= 7) ? std::atoi(argv[6]) : 0;

  // Abrir arquivo de entrada
  TFile *input_file = TFile::Open(argv[1]);
  if (!input_file || input_file->IsZombie())
  {
    std::cerr << "Erro ao abrir arquivo: " << argv[1] << std::endl;
    return 1;
  }

  // Obter a árvore (novo formato HepMC3 com fallback para formato legado LHE)
  TTree *tree = (TTree *)input_file->Get("hepmc3_tree");
  if (!tree)
    tree = (TTree *)input_file->Get("lheTree");
  if (!tree)
  {
    std::cerr << "Erro: árvore 'hepmc3_tree' (ou 'lheTree') não encontrada!" << std::endl;
    input_file->Close();
    return 1;
  }

  const bool isHepMC3Format = (tree->GetBranch("particles.pid") != nullptr);
  std::cout << "Formato de entrada detectado: "
            << (isHepMC3Format ? "HepMC3 ROOT tree" : "LHE tree legado")
            << std::endl;

  TTreeReader reader(tree);

  std::unique_ptr<TTreeReaderArray<int>> pdgIDArr;
  std::unique_ptr<TTreeReaderArray<int>> statusArr;
  std::unique_ptr<TTreeReaderArray<float>> pxArrF;
  std::unique_ptr<TTreeReaderArray<float>> pyArrF;
  std::unique_ptr<TTreeReaderArray<float>> pzArrF;
  std::unique_ptr<TTreeReaderArray<float>> eArrF;
  std::unique_ptr<TTreeReaderArray<double>> pxArrD;
  std::unique_ptr<TTreeReaderArray<double>> pyArrD;
  std::unique_ptr<TTreeReaderArray<double>> pzArrD;
  std::unique_ptr<TTreeReaderArray<double>> eArrD;

  if (isHepMC3Format)
  {
    pdgIDArr = std::make_unique<TTreeReaderArray<int>>(reader, "particles.pid");
    statusArr = std::make_unique<TTreeReaderArray<int>>(reader, "particles.status");
    pxArrD = std::make_unique<TTreeReaderArray<double>>(reader, "particles.momentum.m_v1");
    pyArrD = std::make_unique<TTreeReaderArray<double>>(reader, "particles.momentum.m_v2");
    pzArrD = std::make_unique<TTreeReaderArray<double>>(reader, "particles.momentum.m_v3");
    eArrD = std::make_unique<TTreeReaderArray<double>>(reader, "particles.momentum.m_v4");
  }
  else
  {
    pdgIDArr = std::make_unique<TTreeReaderArray<int>>(reader, "pdgID");
    statusArr = std::make_unique<TTreeReaderArray<int>>(reader, "status");
    pxArrF = std::make_unique<TTreeReaderArray<float>>(reader, "px");
    pyArrF = std::make_unique<TTreeReaderArray<float>>(reader, "py");
    pzArrF = std::make_unique<TTreeReaderArray<float>>(reader, "pz");
    eArrF = std::make_unique<TTreeReaderArray<float>>(reader, "e");
  }

  // Histogramas gerais
  const std::vector<double> mbb_thresholds = {10.0, 15.0, 20.0, 25.0, 30.0, 40.0};

  // colocar os cortes em pT para ver  os cortes
  TH1F *h_mbb = new TH1F("h_mbb", "Massa invariante BB;m_{BB} [GeV];Eventos", 100, 0, 500);

  std::vector<TH1F *> h_mbb_threshold_hists;
  h_mbb_threshold_hists.reserve(mbb_thresholds.size());

  for (double thr : mbb_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_mbb" << static_cast<int>(thr);
    htitle << "Massa invariante BB para m_{BB}>" << static_cast<int>(thr) << " GeV;m_{BB} [GeV];Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 500, 0, 500);
    h_mbb_threshold_hists.push_back(h_thr);
  }

  std::vector<TH1F *> h_mbb_threshold_aftercuts_hists;
  h_mbb_threshold_aftercuts_hists.reserve(mbb_thresholds.size());
  for (double thr : mbb_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_mbb" << static_cast<int>(thr) << "_aftercuts";
    htitle << "Massa invariante BB (aftercuts) para m_{BB}>" << static_cast<int>(thr) << " GeV;m_{BB} [GeV];Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 500, 0, 500);
    h_mbb_threshold_aftercuts_hists.push_back(h_thr);
  }

  TH1F *h_ptbb = new TH1F("h_ptbb", "pT do sistema BB;pT_{BB} [GeV];Eventos", 100, 0, 400);
  std::vector<TH1F *> h_ptbb_threshold_hists;
  h_ptbb_threshold_hists.reserve(mbb_thresholds.size());
  for (double thr : mbb_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_ptbb" << static_cast<int>(thr);
    htitle << "pT do sistema BB para m_{BB}>" << static_cast<int>(thr) << " GeV;pT_{BB} [GeV];Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 500, 0, 500);
    h_ptbb_threshold_hists.push_back(h_thr);
  }
  std::vector<TH1F *> h_ptbb_threshold_aftercuts_hists;
  h_ptbb_threshold_aftercuts_hists.reserve(mbb_thresholds.size());
  for (double thr : mbb_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_ptbb" << static_cast<int>(thr) << "_aftercuts";
    htitle << "pT do sistema BB (aftercuts) para m_{BB}>" << static_cast<int>(thr) << " GeV;pT_{BB} [GeV];Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 500, 0, 500);
    h_ptbb_threshold_aftercuts_hists.push_back(h_thr);
  }

  TH1F *h_ybb = new TH1F("h_ybb", "Rapidez do sistema BB;Y_{BB};Eventos", 100, -5, 5);
  std::vector<TH1F *> h_ybb_threshold_hists;
  h_ybb_threshold_hists.reserve(mbb_thresholds.size());
  for (double thr : mbb_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_ybb" << static_cast<int>(thr);
    htitle << "Rapidez do sistema BB para m_{BB}>" << static_cast<int>(thr) << " GeV;Y_{BB};Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 100, -5, 5);
    h_ybb_threshold_hists.push_back(h_thr);
  }
  std::vector<TH1F *> h_ybb_threshold_aftercuts_hists;
  h_ybb_threshold_aftercuts_hists.reserve(mbb_thresholds.size());
  for (double thr : mbb_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_ybb" << static_cast<int>(thr) << "_aftercuts";
    htitle << "Rapidez do sistema BB (aftercuts) para m_{BB}>" << static_cast<int>(thr) << " GeV;Y_{BB};Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 100, -5, 5);
    h_ybb_threshold_aftercuts_hists.push_back(h_thr);
  }

  TH1F *h_bb_deltaR = new TH1F("h_bb_deltaR", "#DeltaR entre hádrons B;#DeltaR(B,B);Eventos", 100, 0, 6);
  TH1F *h_bb_dphi = new TH1F("h_bb_dphi", "#Delta#phi entre hádrons B;#Delta#phi(B,B);Eventos", 100, 0, 3.2);
  TH1F *h_b1_pt = new TH1F("h_b1_pt", "pT do hádron B leading;pT_{B1} [GeV];Eventos", 100, 0, 300);
  TH1F *h_b2_pt = new TH1F("h_b2_pt", "pT do hádron B subleading;pT_{B2} [GeV];Eventos", 100, 0, 300);
  TH1F *h_b1_eta = new TH1F("h_b1_eta", "#eta do hádron B leading;#eta_{B1};Eventos", 100, -5, 5);
  TH1F *h_b2_eta = new TH1F("h_b2_eta", "#eta do hádron B subleading;#eta_{B2};Eventos", 100, -5, 5);

  TH1F *h_B_pt = new TH1F("h_B_pt", "pT de b partons (truth);pT [GeV];b partons", 100, 0, 300);
  TH1F *h_B_eta = new TH1F("h_B_eta", "#eta de b partons (truth);#eta;b partons", 100, -5, 5);
  TH1F *h_B_pt_reco = new TH1F("h_B_pt_reco", "pT de b-jets (reco);pT [GeV];b-jets reco", 100, 0, 300);
  TH1F *h_B_eta_reco = new TH1F("h_B_eta_reco", "#eta de b-jets (reco);#eta;b-jets reco", 100, -5, 5);
  TH1F *h_numberofevents_cutflow = new TH1F("h_numberofevents_cutflow", "Cutflow;Corte;Eventos (ponderados)", 5, 0, 5);
  TH1F *h_mbb_cut_index = new TH1F("h_mbb_cut_index", "Indexed m_{BB} thresholds;threshold index;Eventos (ponderados)",
                                   static_cast<int>(mbb_thresholds.size()), 0.5, static_cast<double>(mbb_thresholds.size()) + 0.5);
  for (size_t i = 0; i < mbb_thresholds.size(); ++i)
  {
    std::ostringstream label;
    label << "mbb" << static_cast<int>(mbb_thresholds[i]);
    h_mbb_cut_index->GetXaxis()->SetBinLabel(static_cast<int>(i + 1), label.str().c_str());
  }

  // Histogramas após todos os cortes
  TH1F *h_mbb_aftercuts = new TH1F("h_mbb_aftercuts", "Massa invariante BB após cortes;m_{BB} [GeV];Eventos", 500, 0, 500);
  TH1F *h_ptbb_aftercuts = new TH1F("h_ptbb_aftercuts", "pT do sistema BB após cortes;pT_{BB} [GeV];Eventos", 500, 0, 500);
  TH1F *h_ybb_aftercuts = new TH1F("h_ybb_aftercuts", "Rapidez do sistema BB após cortes;Y_{BB};Eventos", 100, -5, 5);

  TH1F *h_bb_deltaR_aftercuts = new TH1F("h_bb_deltaR_aftercuts", "#DeltaR entre mésons B após cortes;#DeltaR(B,B);Eventos", 100, 0, 6);
  TH1F *h_bb_dphi_aftercuts = new TH1F("h_bb_dphi_aftercuts", "#Delta#phi entre mésons B após cortes;#Delta#phi(B,B);Eventos", 100, 0, 3.2);
  TH1F *h_b1_pt_aftercuts = new TH1F("h_b1_pt_aftercuts", "pT do méson B leading após cortes;pT_{B1} [GeV];Eventos", 100, 0, 300);
  TH1F *h_b2_pt_aftercuts = new TH1F("h_b2_pt_aftercuts", "pT do méson B subleading após cortes;pT_{B2} [GeV];Eventos", 100, 0, 300);
  TH1F *h_b1_eta_aftercuts = new TH1F("h_b1_eta_aftercuts", "#eta do méson B leading após cortes;#eta_{B1};Eventos", 100, -5, 5);
  TH1F *h_b2_eta_aftercuts = new TH1F("h_b2_eta_aftercuts", "#eta do méson B subleading após cortes;#eta_{B2};Eventos", 100, -5, 5);

  Long64_t nEntries = tree->GetEntries();
  std::cout << "Processando " << nEntries << " eventos..." << std::endl;

  std::vector<double> pdf_event_weight(static_cast<size_t>(nEntries), 1.0);
  std::vector<double> pdf_all_event_weights;
  int nPdfMembers = 0;
  if (!pdf_weights_csv.empty())
  {
    std::ifstream csv_in(pdf_weights_csv.c_str());
    if (!csv_in)
    {
      std::cerr << "Erro ao abrir CSV de pesos PDF: " << pdf_weights_csv << std::endl;
      return 1;
    }

    std::string line;
    if (!std::getline(csv_in, line))
    {
      std::cerr << "Erro: CSV de pesos PDF vazio: " << pdf_weights_csv << std::endl;
      return 1;
    }

    size_t nColsHeader = 0;
    {
      std::stringstream hs(line);
      std::string col;
      while (std::getline(hs, col, ','))
        ++nColsHeader;
    }

    if (nColsHeader < 8)
    {
      std::cerr << "Erro: header do CSV invalido (colunas insuficientes): " << nColsHeader << std::endl;
      return 1;
    }

    // Compatível com layouts legado (pesos a partir da coluna 7) e novo (a partir da 10).
    const int weightsStartCol = (nColsHeader >= 11) ? 10 : 7;
    const int nMembersCsv = static_cast<int>(nColsHeader) - weightsStartCol;
    if (nMembersCsv <= 0)
    {
      std::cerr << "Erro: não foi possível determinar colunas de pesos PDF no CSV." << std::endl;
      return 1;
    }

    if (pdf_member < 0 || pdf_member >= nMembersCsv)
    {
      std::cerr << "Erro: pdf_member=" << pdf_member << " fora do intervalo [0," << (nMembersCsv - 1) << "]" << std::endl;
      return 1;
    }

    nPdfMembers = nMembersCsv;
    pdf_all_event_weights.assign(static_cast<size_t>(nEntries) * static_cast<size_t>(nPdfMembers), 1.0);

    size_t loaded = 0;
    while (std::getline(csv_in, line))
    {
      if (line.empty())
        continue;

      std::vector<std::string> cols;
      cols.reserve(static_cast<size_t>(nColsHeader));
      std::stringstream ss(line);
      std::string cell;
      while (std::getline(ss, cell, ','))
        cols.push_back(trim(cell));

      if (cols.size() < static_cast<size_t>(weightsStartCol + nMembersCsv))
        continue;

      try
      {
        const size_t eventIdx = static_cast<size_t>(std::stoll(cols[0]));
        if (eventIdx >= static_cast<size_t>(nEntries))
          continue;

        for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        {
          const std::string &weightStr = cols[weightsStartCol + iMember];
          if (weightStr.empty())
            continue;
          const double wMember = std::stod(weightStr);
          pdf_all_event_weights[eventIdx * static_cast<size_t>(nPdfMembers) + static_cast<size_t>(iMember)] = wMember;
          if (iMember == pdf_member)
            pdf_event_weight[eventIdx] = wMember;
        }
        ++loaded;
      }
      catch (const std::exception &e)
      {
        std::cerr << "Erro ao parsear linha do CSV: " << e.what() << std::endl;
        std::cerr << "Linha: " << line << std::endl;
        continue;
      }
    }

    std::cout << "CSV de pesos carregado: " << loaded
              << " eventos, membro " << pdf_member
              << " de " << nMembersCsv << " membros." << std::endl;
  }

  std::vector<double> cutflow_sumw2(4, 0.0);
  std::vector<double> mbb_scan_sumw2(mbb_thresholds.size(), 0.0);
  std::vector<std::vector<double>> pdf_cutflow_yields(4);
  if (nPdfMembers > 0)
    pdf_cutflow_yields.assign(4, std::vector<double>(static_cast<size_t>(nPdfMembers), 0.0));

  std::vector<TH1F *> h_mbb_pdf_members;
  std::vector<TH1F *> h_mbb_aftercuts_pdf_members;
  std::vector<TH1F *> h_ybb_pdf_members;
  std::vector<TH1F *> h_ybb_aftercuts_pdf_members;
  std::vector<TH1F *> h_ptbb_pdf_members;
  std::vector<TH1F *> h_ptbb_aftercuts_pdf_members;

  if (nPdfMembers > 0)
  {
    h_mbb_pdf_members = createPdfMemberHistograms(h_mbb, "h_mbb", nPdfMembers);
    h_mbb_aftercuts_pdf_members = createPdfMemberHistograms(h_mbb_aftercuts, "h_mbb_aftercuts", nPdfMembers);
    h_ybb_pdf_members = createPdfMemberHistograms(h_ybb, "h_ybb", nPdfMembers);
    h_ybb_aftercuts_pdf_members = createPdfMemberHistograms(h_ybb_aftercuts, "h_ybb_aftercuts", nPdfMembers);
    h_ptbb_pdf_members = createPdfMemberHistograms(h_ptbb, "h_ptbb", nPdfMembers);
    h_ptbb_aftercuts_pdf_members = createPdfMemberHistograms(h_ptbb_aftercuts, "h_ptbb_aftercuts", nPdfMembers);
  }

  std::cout << "Modo de objeto pesado: b-jets reco com FastJet (matching DeltaR com b partons)" << std::endl;

  // Loop sobre eventos
  Long64_t iEvent = 0;
  while (reader.Next())
  {

    const Long64_t currentEvent = iEvent++;

    std::vector<int> pdgID;
    std::vector<int> status;
    std::vector<float> px;
    std::vector<float> py;
    std::vector<float> pz;
    std::vector<float> e;

    const size_t nPart = static_cast<size_t>(pdgIDArr->GetSize());
    pdgID.reserve(nPart);
    status.reserve(nPart);
    px.reserve(nPart);
    py.reserve(nPart);
    pz.reserve(nPart);
    e.reserve(nPart);

    for (size_t iPart = 0; iPart < nPart; ++iPart)
    {
      pdgID.push_back((*pdgIDArr)[iPart]);
      status.push_back((*statusArr)[iPart]);
      if (isHepMC3Format)
      {
        px.push_back(static_cast<float>((*pxArrD)[iPart]));
        py.push_back(static_cast<float>((*pyArrD)[iPart]));
        pz.push_back(static_cast<float>((*pzArrD)[iPart]));
        e.push_back(static_cast<float>((*eArrD)[iPart]));
      }
      else
      {
        px.push_back((*pxArrF)[iPart]);
        py.push_back((*pyArrF)[iPart]);
        pz.push_back((*pzArrF)[iPart]);
        e.push_back((*eArrF)[iPart]);
      }
    }

    TRandom3 detector_rng(7000 + currentEvent);

    weight = (nEntries > 0) ? (xs / nEntries) : 0.0;
    const double pdf_selected_weight = pdf_event_weight[static_cast<size_t>(currentEvent)];
    double event_weight = weight * pdf_selected_weight;

    // Cutflow bin 1: todos os eventos gerados
    h_numberofevents_cutflow->Fill(0.0, event_weight);
    cutflow_sumw2[0] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(currentEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[0][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
    }

    std::map<int, int> statusCounts;
    for (size_t i = 0; i < pdgID.size(); ++i)
      if (std::abs(pdgID[i]) == 5)
        statusCounts[status[i]]++;
    if (currentEvent == 0)
    {
      for (auto &[s, n] : statusCounts)
        std::cout << "b-quark status=" << s << " count=" << n << std::endl;
    }

    std::vector<TLorentzVector> b_quarks_truth = collectBQuarks(pdgID, status, px, py, pz, e, isHepMC3Format);

    for (const auto &bq : b_quarks_truth)
    {
      h_B_pt->Fill(bq.Pt(), event_weight);
      h_B_eta->Fill(bq.Eta(), event_weight);
    }

    // Reconstrução de jatos com FastJet usando partículas finais visíveis
    std::vector<fastjet::PseudoJet> particles;
    for (size_t iPart = 0; iPart < pdgID.size(); ++iPart)
    {
      const int pdg = pdgID[iPart];
      const int stat = status[iPart];
      if (stat != 1)
        continue;

      // Remover neutrinos do clustering
      if (std::abs(pdg) == 12 || std::abs(pdg) == 14 || std::abs(pdg) == 16)
        continue;

      fastjet::PseudoJet particle(px[iPart], py[iPart], pz[iPart], e[iPart]);
      particle.set_user_index(static_cast<int>(iPart));
      particles.push_back(particle);
    }

    const double R = 0.4;
    const double ptmin = 5.0;
    fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
    fastjet::ClusterSequence cs(particles, jet_def);
    std::vector<fastjet::PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(ptmin));

    std::vector<fastjet::PseudoJet> reco_jets;
    for (const auto &jet : jets)
    {
      fastjet::PseudoJet smeared_jet = smearJet(jet, detector_rng);
      if (smeared_jet.pt() > ptmin)
        reco_jets.push_back(smeared_jet);
    }
    jets = sorted_by_pt(reco_jets);

    int n_bjets = 0;
    std::vector<bool> is_bjet = tagBJetsByDeltaR(jets, b_quarks_truth, n_bjets);

    std::vector<TLorentzVector> bjets_reco;
    bjets_reco.reserve(static_cast<size_t>(n_bjets));
    for (size_t iJet = 0; iJet < jets.size(); ++iJet)
    {
      if (!is_bjet[iJet])
        continue;

      TLorentzVector bjet;
      bjet.SetPxPyPzE(jets[iJet].px(), jets[iJet].py(), jets[iJet].pz(), jets[iJet].E());
      bjets_reco.push_back(bjet);
    }

    for (const auto &bjet : bjets_reco)
    {
      h_B_pt_reco->Fill(bjet.Pt(), event_weight);
      h_B_eta_reco->Fill(bjet.Eta(), event_weight);
    }

    if (bjets_reco.size() < 2)
      continue;
    h_numberofevents_cutflow->Fill(1.0, event_weight);
    cutflow_sumw2[1] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(currentEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[1][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
    }

    const auto &b1 = bjets_reco[0];
    const auto &b2 = bjets_reco[1];

    // Cortes cinemáticos (análogo ao fluxo de ttbar: objetos centrais e com pT mínimo)
    bool pass_kinematic_cuts = (b1.Pt() > 5.0 && b2.Pt() > 5.0 && fabs(b1.Eta()) < 2.5 && fabs(b2.Eta()) < 2.5);

    if (!pass_kinematic_cuts)
      continue;

    h_numberofevents_cutflow->Fill(2.0, event_weight);
    cutflow_sumw2[2] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(currentEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[2][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
    }

    // Sistema BB
    TLorentzVector bb_system = b1 + b2;
    double mbb = bb_system.M();
    double ptbb = bb_system.Pt();
    double deta = b1.Eta() - b2.Eta();
    double dphi = b1.Phi() - b2.Phi();
    while (dphi > M_PI)
      dphi -= 2 * M_PI;
    while (dphi < -M_PI)
      dphi += 2 * M_PI;
    double deltaR_bb = sqrt(deta * deta + dphi * dphi);
    double ybb = bb_system.Rapidity();

    // Histogramas antes do último corte de massa
    h_mbb->Fill(mbb, event_weight);
    h_ptbb->Fill(ptbb, event_weight);
    h_ybb->Fill(ybb, event_weight);
    h_bb_deltaR->Fill(deltaR_bb, event_weight);
    h_bb_dphi->Fill(fabs(dphi), event_weight);
    h_b1_pt->Fill(b1.Pt(), event_weight);
    h_b2_pt->Fill(b2.Pt(), event_weight);
    h_b1_eta->Fill(b1.Eta(), event_weight);
    h_b2_eta->Fill(b2.Eta(), event_weight);

    for (size_t iThr = 0; iThr < mbb_thresholds.size(); ++iThr)
    {
      if (mbb > mbb_thresholds[iThr])
      {
        h_mbb_threshold_hists[iThr]->Fill(mbb, event_weight);
        h_ptbb_threshold_hists[iThr]->Fill(ptbb, event_weight);
        h_ybb_threshold_hists[iThr]->Fill(ybb, event_weight);
        h_mbb_cut_index->Fill(static_cast<double>(iThr + 1), event_weight);
        mbb_scan_sumw2[iThr] += event_weight * event_weight;
      }
    }

    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(currentEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
      {
        const double memberWeight = weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
        h_mbb_pdf_members[static_cast<size_t>(iMember)]->Fill(mbb, memberWeight);
        h_ybb_pdf_members[static_cast<size_t>(iMember)]->Fill(ybb, memberWeight);
        h_ptbb_pdf_members[static_cast<size_t>(iMember)]->Fill(ptbb, memberWeight);
      }
    }

    if (mbb < 10.0)
      continue;

    h_numberofevents_cutflow->Fill(3.0, event_weight);
    cutflow_sumw2[3] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(currentEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[3][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
    }

    // Histogramas após todos os cortes
    h_mbb_aftercuts->Fill(mbb, event_weight);
    for (size_t iThr = 0; iThr < mbb_thresholds.size(); ++iThr)
    {
      if (mbb > mbb_thresholds[iThr])
      {
        h_mbb_threshold_aftercuts_hists[iThr]->Fill(mbb, event_weight);
        h_ptbb_threshold_aftercuts_hists[iThr]->Fill(ptbb, event_weight);
        h_ybb_threshold_aftercuts_hists[iThr]->Fill(ybb, event_weight);
      }
    }
    h_ptbb_aftercuts->Fill(ptbb, event_weight);
    h_bb_deltaR_aftercuts->Fill(deltaR_bb, event_weight);
    h_bb_dphi_aftercuts->Fill(fabs(dphi), event_weight);
    h_b1_pt_aftercuts->Fill(b1.Pt(), event_weight);
    h_b2_pt_aftercuts->Fill(b2.Pt(), event_weight);
    h_b1_eta_aftercuts->Fill(b1.Eta(), event_weight);
    h_b2_eta_aftercuts->Fill(b2.Eta(), event_weight);
    h_ybb_aftercuts->Fill(ybb, event_weight);
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(currentEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
      {
        const double memberWeight = weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
        h_mbb_aftercuts_pdf_members[static_cast<size_t>(iMember)]->Fill(mbb, memberWeight);
        h_ybb_aftercuts_pdf_members[static_cast<size_t>(iMember)]->Fill(ybb, memberWeight);
        h_ptbb_aftercuts_pdf_members[static_cast<size_t>(iMember)]->Fill(ptbb, memberWeight);
      }
    }

    // Progresso
    if (currentEvent % 1000 == 0)
    {
      std::cout << "Processado: " << currentEvent << " / " << nEntries
                << " (" << (100.0 * currentEvent / nEntries) << "%)" << std::endl;
    }

    // currentEvent;
  }

  // Output em .txt
  FILE *bbanalysis;
  bbanalysis = fopen(txt_name.c_str(), "a");
  fprintf(bbanalysis, "%s\n", "===========================================================");
  fprintf(bbanalysis, "%s\n", "========== OO -> BB Analysis (FastJet b-jets) ==============");
  fprintf(bbanalysis, "%s\n", "===========================================================");
  fprintf(bbanalysis, "%s\n", "Object mode: b-jets via FastJet anti-kt R=0.4 + DeltaR(b-parton,jet)<0.4");
  fprintf(bbanalysis, "%s%.3f%s\n", "Cross Section: ", xs, " nb");
  fprintf(bbanalysis, "%s%.1f%s\n", "Integrated Luminosity: ", lumi, " nb^-1");
  fprintf(bbanalysis, "%s%.2f\n", "Expected events: ", xs * lumi);
  fprintf(bbanalysis, "%s%lld\n", "Generated events: ", nEntries);
  fprintf(bbanalysis, "%s\n", "===========================================================");
  fprintf(bbanalysis, "%s\n", "=================== Cut Flow ==============================");
  fprintf(bbanalysis, "%s\n", "===========================================================");
  fprintf(bbanalysis, "%s\n", "Corte                                  | Eventos (xs) +- stat | Eficiência | N(L=8nb^-1) +- stat");
  fprintf(bbanalysis, "%s\n", "---------------------------------------------------------");

  auto printCutflowLine = [&](const char *label, int stageIndex, int binIndex)
  {
    const double nominal = h_numberofevents_cutflow->GetBinContent(binIndex);
    const double reference = h_numberofevents_cutflow->GetBinContent(1);
    const double efficiency = (reference > 0.0) ? 100.0 * nominal / reference : 0.0;
    const double statUnc = std::sqrt(cutflow_sumw2[static_cast<size_t>(stageIndex)]);
    HessianUncertainty pdfUnc;
    if (!pdf_cutflow_yields.empty())
      pdfUnc = computeHessianUncertainty(pdf_cutflow_yields[static_cast<size_t>(stageIndex)]);

    fprintf(bbanalysis,
            "%-38s | %8.5f +- %8.5f (stat) +%8.5f -%8.5f (PDF) | %7.3f%% | %8.4f +- %8.4f (stat) +%8.4f -%8.4f (PDF)\n",
            label,
            nominal,
            statUnc,
            pdfUnc.up,
            pdfUnc.down,
            efficiency,
            nominal * lumi,
            statUnc * lumi,
            pdfUnc.up * lumi,
            pdfUnc.down * lumi);
  };

  printCutflowLine("Total", 0, 1);
  printCutflowLine(">= 2 b-jets reconstruidos", 1, 2);
  printCutflowLine("b1,b2: pT>5 GeV, |eta|<2.5", 2, 3);
  printCutflowLine("m_bb > 10 GeV", 3, 4);

  fprintf(bbanalysis, "%s\n", "---------------------------------------------------------");
  fprintf(bbanalysis, "%s\n", "Indexed mass-threshold scan (after kinematic cuts)");
  fprintf(bbanalysis, "%s\n", "idx  label    Yield [nb]    stat [nb]    Eff(vs pre-mass) [%]");
  const double preMassYield = h_numberofevents_cutflow->GetBinContent(3);
  for (size_t iThr = 0; iThr < mbb_thresholds.size(); ++iThr)
  {
    const int bin = static_cast<int>(iThr + 1);
    const double y = h_mbb_cut_index->GetBinContent(bin);
    const double stat = std::sqrt(mbb_scan_sumw2[iThr]);
    const double eff = (preMassYield > 0.0) ? 100.0 * y / preMassYield : 0.0;
    std::ostringstream label;
    label << "mbb" << static_cast<int>(mbb_thresholds[iThr]);
    fprintf(bbanalysis, "%3d  %-7s %12.4f %12.4f %14.3f\n", bin, label.str().c_str(), y, stat, eff);
  }

  fprintf(bbanalysis, "%s\n", "===========================================================");
  fprintf(bbanalysis, "\n");
  fclose(bbanalysis);

  // Salvar histogramas
  TFile *output_file = new TFile(output_name.c_str(), "RECREATE");

  auto mbbPdfVars = buildPdfVariationHistograms(h_mbb_pdf_members, "h_mbb_pdf_up", "h_mbb_pdf_down");
  auto mbbAftercutsPdfVars = buildPdfVariationHistograms(h_mbb_aftercuts_pdf_members, "h_mbb_aftercuts_pdf_up", "h_mbb_aftercuts_pdf_down");
  auto ptbbPdfVars = buildPdfVariationHistograms(h_ptbb_pdf_members, "h_ptbb_pdf_up", "h_ptbb_pdf_down");
  auto ptbbAftercutsPdfVars = buildPdfVariationHistograms(h_ptbb_aftercuts_pdf_members, "h_ptbb_aftercuts_pdf_up", "h_ptbb_aftercuts_pdf_down");
  auto ybbPdfVars = buildPdfVariationHistograms(h_ybb_pdf_members, "h_ybb_pdf_up", "h_ybb_pdf_down");
  auto ybbAftercutsPdfVars = buildPdfVariationHistograms(h_ybb_aftercuts_pdf_members, "h_ybb_aftercuts_pdf_up", "h_ybb_aftercuts_pdf_down");

  h_mbb->Write();
  if (mbbPdfVars.first)
  {
    mbbPdfVars.first->Write();
    mbbPdfVars.second->Write();
  }
  h_ptbb->Write();
  h_ybb->Write();
  h_bb_deltaR->Write();
  h_bb_dphi->Write();
  h_b1_pt->Write();
  h_b2_pt->Write();
  h_b1_eta->Write();
  h_b2_eta->Write();
  h_B_pt->Write();
  h_B_eta->Write();
  h_B_pt_reco->Write();
  h_B_eta_reco->Write();
  h_numberofevents_cutflow->Write();
  h_mbb_cut_index->Write();
  for (TH1F *h_thr : h_mbb_threshold_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_mbb_threshold_aftercuts_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_ptbb_threshold_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_ptbb_threshold_aftercuts_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_ybb_threshold_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_ybb_threshold_aftercuts_hists)
    h_thr->Write();
  if (mbbAftercutsPdfVars.first)
  {
    mbbAftercutsPdfVars.first->Write();
    mbbAftercutsPdfVars.second->Write();
  }
  if (ptbbPdfVars.first)
  {
    ptbbPdfVars.first->Write();
    ptbbPdfVars.second->Write();
  }
  if (ptbbAftercutsPdfVars.first)
  {
    ptbbAftercutsPdfVars.first->Write();
    ptbbAftercutsPdfVars.second->Write();
  }
  h_ybb_aftercuts->Write();
  h_ptbb_aftercuts->Write();
  h_mbb_aftercuts->Write();
  h_bb_deltaR_aftercuts->Write();
  h_bb_dphi_aftercuts->Write();
  h_b1_pt_aftercuts->Write();
  h_b2_pt_aftercuts->Write();
  h_b1_eta_aftercuts->Write();
  h_b2_eta_aftercuts->Write();
  ;
  if (ybbPdfVars.first)
  {
    ybbPdfVars.first->Write();
    ybbPdfVars.second->Write();
  }
  if (ybbAftercutsPdfVars.first)
  {
    ybbAftercutsPdfVars.first->Write();
    ybbAftercutsPdfVars.second->Write();
  }

  output_file->Close();

  std::cout << "\nHistogramas salvos em: " << output_name << std::endl;
  std::cout << "Arquivo de análise salvo em: " << txt_name << std::endl;

  // Fechar arquivo de entrada
  input_file->Close();
  delete input_file;
  delete output_file;

  return 0;
}