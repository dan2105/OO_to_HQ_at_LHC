/*
 * Análise de produção de cc̄ em colisões O-O
 * Aplica pressupostos experimentais inspirados no fluxo de ttbar:
 * - reconstrução de jatos com FastJet
 * - classificação de c-jets por matching em DeltaR com c-quarks status 2
 * - resposta simplificada de detector (smearing e eficiência)
 * - cutflow ponderado com incerteza estatística
 *
 * Compilar: g++ -o analyze_cc analyze_cc.cpp `root-config --cflags --libs` `fastjet-config --cxxflags --libs --plugins`
 * Executar: ./analyze_cc input.root output.root [cross_section_nb] [txt_name]
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

std::vector<TLorentzVector> collectCQuarksStatus2(const std::vector<int> &pdgID,
                                                  const std::vector<int> &status,
                                                  const std::vector<float> &px,
                                                  const std::vector<float> &py,
                                                  const std::vector<float> &pz,
                                                  const std::vector<float> &e)
{
  std::vector<TLorentzVector> c_quarks;
  for (size_t iPart = 0; iPart < pdgID.size(); ++iPart)
  {
    const bool isCharm = (std::abs(pdgID[iPart]) == 4);
    const bool isStatus2 = (status[iPart] == 2);
    if (isCharm && isStatus2)
    {
      TLorentzVector c_quark;
      c_quark.SetPxPyPzE(px[iPart], py[iPart], pz[iPart], e[iPart]);
      c_quarks.push_back(c_quark);
    }
  }
  return c_quarks;
}

std::vector<bool> tagCJetsByDeltaR(const std::vector<fastjet::PseudoJet> &jets,
                                   const std::vector<TLorentzVector> &c_quarks,
                                   int &n_cjets)
{
  std::vector<bool> is_cjet(jets.size(), false);
  n_cjets = 0;

  for (size_t iJet = 0; iJet < jets.size(); ++iJet)
  {
    const double jet_eta = jets[iJet].eta();
    const double jet_phi = jets[iJet].phi();

    for (const auto &c_quark : c_quarks)
    {
      const double deta = jet_eta - c_quark.Eta();
      double dphi = jet_phi - c_quark.Phi();
      while (dphi > M_PI)
        dphi -= 2 * M_PI;
      while (dphi < -M_PI)
        dphi += 2 * M_PI;

      const double deltaR = std::sqrt(deta * deta + dphi * dphi);
      if (deltaR < 0.4)
      {
        is_cjet[iJet] = true;
        ++n_cjets;
        break;
      }
    }
  }

  return is_cjet;
}

// PDG IDs de mesons D e barions com charme
static const std::vector<int> CHARMED_MESON_PDGIDS = {
    411, -411, // D+-, D-+
    421, -421, // D0, D0bar
    431, -431  // D_s+-, D_s-+
};

static const std::vector<int> CHARMED_BARYON_PDGIDS = {
    4122, -4122, // Lambda_c+, Lambda_c-
    4212, -4212, // Sigma_c+, Sigma_c-
    4222, -4222, // Sigma_c++, Sigma_c--
    4112, -4112, // Sigma_c0, Sigma_c0bar
    4232, -4232, // Xi_c+, Xi_c-
    4132, -4132, // Xi_c0, Xi_c0bar
    4332, -4332  // Omega_c0, Omega_c0bar
};

std::vector<TLorentzVector> collectCharmedHadrons(const std::vector<int> &pdgID,
                                                  const std::vector<int> &status,
                                                  const std::vector<float> &px,
                                                  const std::vector<float> &py,
                                                  const std::vector<float> &pz,
                                                  const std::vector<float> &e)
{
  std::vector<TLorentzVector> charmed_hadrons;
  for (size_t iPart = 0; iPart < pdgID.size(); ++iPart)
  {
    const int pdg = pdgID[iPart];
    const int stat = status[iPart];

    // Status 2: intermediate (decayed hadrons)
    if (stat != 2)
      continue;

    bool isCharmedHadron = false;

    // Check D mesons
    for (int did : CHARMED_MESON_PDGIDS)
    {
      if (pdg == did)
      {
        isCharmedHadron = true;
        break;
      }
    }

    // Check charmed baryons
    if (!isCharmedHadron)
    {
      for (int bid : CHARMED_BARYON_PDGIDS)
      {
        if (pdg == bid)
        {
          isCharmedHadron = true;
          break;
        }
      }
    }

    if (isCharmedHadron)
    {
      TLorentzVector hadron;
      hadron.SetPxPyPzE(px[iPart], py[iPart], pz[iPart], e[iPart]);
      charmed_hadrons.push_back(hadron);
    }
  }
  return charmed_hadrons;
}

std::vector<bool> tagJetsByCharmedHadrons(const std::vector<fastjet::PseudoJet> &jets,
                                          const std::vector<TLorentzVector> &charmed_hadrons,
                                          int &n_cjets)
{
  std::vector<bool> is_cjet(jets.size(), false);
  n_cjets = 0;

  for (size_t iJet = 0; iJet < jets.size(); ++iJet)
  {
    const double jet_eta = jets[iJet].eta();
    const double jet_phi = jets[iJet].phi();

    for (const auto &hadron : charmed_hadrons)
    {
      const double deta = jet_eta - hadron.Eta();
      double dphi = jet_phi - hadron.Phi();
      while (dphi > M_PI)
        dphi -= 2 * M_PI;
      while (dphi < -M_PI)
        dphi += 2 * M_PI;

      const double deltaR = std::sqrt(deta * deta + dphi * dphi);
      if (deltaR < 0.4)
      {
        is_cjet[iJet] = true;
        ++n_cjets;
        break;
      }
    }
  }

  return is_cjet;
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
    std::cout << "Uso: " << argv[0] << " <input.root> <output.root> [cross_section_nb] [txt_name] [pdfweights.csv] [pdf_member] [tag_mode]" << std::endl;
    std::cout << "  cross_section_nb: seção de choque em nb (opcional)" << std::endl;
    std::cout << "  pdfweights.csv: arquivo CSV com pesos por evento (opcional)" << std::endl;
    std::cout << "  pdf_member: indice do membro Hessiano (ex.: 0..106, opcional)" << std::endl;
    std::cout << "  tag_mode: 'quarks' para c-quarks status 2 (default), 'hadrons' para D mesons + barions charme status 2" << std::endl;
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

  std::string txt_name = (argc >= 5) ? argv[4] : "out_ccbar.txt";
  std::string pdf_weights_csv = (argc >= 6) ? argv[5] : "";
  int pdf_member = (argc >= 7) ? std::atoi(argv[6]) : 0;
  std::string tag_mode = (argc >= 8) ? argv[7] : "quarks";

  if (tag_mode != "quarks" && tag_mode != "hadrons")
  {
    std::cerr << "Erro: tag_mode deve ser 'quarks' ou 'hadrons'" << std::endl;
    return 1;
  }

  std::cout << "Modo de tagging: " << tag_mode << std::endl;

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
  const std::vector<double> mcc_thresholds = {10.0, 15.0, 20.0, 25.0, 30.0, 40.0};

  // colocar os cortes em pT para ver  os cortes
  TH1F *h_mcc = new TH1F("h_mcc", "Massa invariante cc;m_{cc} [GeV];Eventos", 100, 0, 500);

  std::vector<TH1F *> h_mcc_threshold_hists;
  h_mcc_threshold_hists.reserve(mcc_thresholds.size());

  for (double thr : mcc_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_mcc" << static_cast<int>(thr);
    htitle << "Massa invariante cc para m_{cc}>" << static_cast<int>(thr) << " GeV;m_{cc} [GeV];Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 500, 0, 500);
    h_mcc_threshold_hists.push_back(h_thr);
  }

  std::vector<TH1F *> h_mcc_threshold_aftercuts_hists;
  h_mcc_threshold_aftercuts_hists.reserve(mcc_thresholds.size());
  for (double thr : mcc_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_mcc" << static_cast<int>(thr) << "_aftercuts";
    htitle << "Massa invariante cc (aftercuts) para m_{cc}>" << static_cast<int>(thr) << " GeV;m_{cc} [GeV];Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 500, 0, 500);
    h_mcc_threshold_aftercuts_hists.push_back(h_thr);
  }

  TH1F *h_ptcc = new TH1F("h_ptcc", "pT do sistema cc;pT_{cc} [GeV];Eventos", 100, 0, 400);
  std::vector<TH1F *> h_ptcc_threshold_hists;
  h_ptcc_threshold_hists.reserve(mcc_thresholds.size());
  for (double thr : mcc_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_ptcc" << static_cast<int>(thr);
    htitle << "pT do sistema cc para m_{cc}>" << static_cast<int>(thr) << " GeV;pT_{cc} [GeV];Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 500, 0, 500);
    h_ptcc_threshold_hists.push_back(h_thr);
  }
  std::vector<TH1F *> h_ptcc_threshold_aftercuts_hists;
  h_ptcc_threshold_aftercuts_hists.reserve(mcc_thresholds.size());
  for (double thr : mcc_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_ptcc" << static_cast<int>(thr) << "_aftercuts";
    htitle << "pT do sistema cc (aftercuts) para m_{cc}>" << static_cast<int>(thr) << " GeV;pT_{cc} [GeV];Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 500, 0, 500);
    h_ptcc_threshold_aftercuts_hists.push_back(h_thr);
  }

  TH1F *h_ycc = new TH1F("h_ycc", "Rapidez do sistema cc;Y_{cc};Eventos", 100, -5, 5);
  std::vector<TH1F *> h_ycc_threshold_hists;
  h_ycc_threshold_hists.reserve(mcc_thresholds.size());
  for (double thr : mcc_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_ycc" << static_cast<int>(thr);
    htitle << "Rapidez do sistema cc para m_{cc}>" << static_cast<int>(thr) << " GeV;Y_{cc};Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 100, -5, 5);
    h_ycc_threshold_hists.push_back(h_thr);
  }
  std::vector<TH1F *> h_ycc_threshold_aftercuts_hists;
  h_ycc_threshold_aftercuts_hists.reserve(mcc_thresholds.size());
  for (double thr : mcc_thresholds)
  {
    std::ostringstream hname, htitle;
    hname << "h_ycc" << static_cast<int>(thr) << "_aftercuts";
    htitle << "Rapidez do sistema cc (aftercuts) para m_{cc}>" << static_cast<int>(thr) << " GeV;Y_{cc};Eventos";
    TH1F *h_thr = new TH1F(hname.str().c_str(), htitle.str().c_str(), 100, -5, 5);
    h_ycc_threshold_aftercuts_hists.push_back(h_thr);
  }

  TH1F *h_cc_deltaR = new TH1F("h_cc_deltaR", "#DeltaR entre c-jets;#DeltaR(c,c);Eventos", 100, 0, 6);
  TH1F *h_cc_dphi = new TH1F("h_cc_dphi", "#Delta#phi entre c-jets;#Delta#phi(c,c);Eventos", 100, 0, 3.2);
  TH1F *h_c1_pt = new TH1F("h_c1_pt", "pT do c-jet leading;pT_{c1} [GeV];Eventos", 100, 0, 300);
  TH1F *h_c2_pt = new TH1F("h_c2_pt", "pT do c-jet subleading;pT_{c2} [GeV];Eventos", 100, 0, 300);
  TH1F *h_c1_eta = new TH1F("h_c1_eta", "#eta do c-jet leading;#eta_{c1};Eventos", 100, -5, 5);
  TH1F *h_c2_eta = new TH1F("h_c2_eta", "#eta do c-jet subleading;#eta_{c2};Eventos", 100, -5, 5);

  TH1F *h_numberofevents_cutflow = new TH1F("h_numberofevents_cutflow", "Cutflow;Corte;Eventos (ponderados)", 5, 0, 5);

  TH1F *h_mcc_cut_index = new TH1F("h_mcc_cut_index", "Indexed m_{cc} thresholds;threshold index;Eventos (ponderados)",
                                   static_cast<int>(mcc_thresholds.size()), 0.5, static_cast<double>(mcc_thresholds.size()) + 0.5);
  for (size_t i = 0; i < mcc_thresholds.size(); ++i)
  {
    std::ostringstream label;
    label << "mcc" << static_cast<int>(mcc_thresholds[i]);
    h_mcc_cut_index->GetXaxis()->SetBinLabel(static_cast<int>(i + 1), label.str().c_str());
  }

  // Histogramas após todos os cortes
  TH1F *h_mcc_aftercuts = new TH1F("h_mcc_aftercuts", "Massa invariante cc após cortes;m_{cc} [GeV];Eventos", 500, 0, 500);
  TH1F *h_ptcc_aftercuts = new TH1F("h_ptcc_aftercuts", "pT do sistema cc após cortes;pT_{cc} [GeV];Eventos", 500, 0, 500);
  TH1F *h_ycc_aftercuts = new TH1F("h_ycc_aftercuts", "Rapidez do sistema cc após cortes;Y_{cc};Eventos", 100, -5, 5);

  TH1F *h_cc_deltaR_aftercuts = new TH1F("h_cc_deltaR_aftercuts", "#DeltaR entre c-jets após cortes;#DeltaR(c,c);Eventos", 100, 0, 6);
  TH1F *h_cc_dphi_aftercuts = new TH1F("h_cc_dphi_aftercuts", "#Delta#phi entre c-jets após cortes;#Delta#phi(c,c);Eventos", 100, 0, 3.2);
  TH1F *h_c1_pt_aftercuts = new TH1F("h_c1_pt_aftercuts", "pT do c-jet leading após cortes;pT_{c1} [GeV];Eventos", 100, 0, 300);
  TH1F *h_c2_pt_aftercuts = new TH1F("h_c2_pt_aftercuts", "pT do c-jet subleading após cortes;pT_{c2} [GeV];Eventos", 100, 0, 300);
  TH1F *h_c1_eta_aftercuts = new TH1F("h_c1_eta_aftercuts", "#eta do c-jet leading após cortes;#eta_{c1};Eventos", 100, -5, 5);
  TH1F *h_c2_eta_aftercuts = new TH1F("h_c2_eta_aftercuts", "#eta do c-jet subleading após cortes;#eta_{c2};Eventos", 100, -5, 5);

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

    // Verificar se tem pelo menos as colunas: event, y1, y2, kfac1, kfac2, mll, pdfset, pdfname, ?, ?, e depois membros
    // Formato: event,y1,y2,kfac1,kfac2,mll,LHAPDF6:...,pdfname,?,?,peso1,peso2,...
    if (nColsHeader < 11)
    {
      std::cerr << "Erro: header do CSV invalido (colunas insuficientes): " << nColsHeader << " (esperado >= 11)" << std::endl;
      return 1;
    }

    const int nMembersCsv = static_cast<int>(nColsHeader) - 10; // Pesos começam na coluna 10
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

      if (cols.size() < static_cast<size_t>(10 + nMembersCsv))
        continue;

      try
      {
        const size_t eventIdx = static_cast<size_t>(std::stoll(cols[0]));
        if (eventIdx >= static_cast<size_t>(nEntries))
          continue;

        for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        {
          const std::string &weightStr = cols[10 + iMember]; // Pesos começam na coluna 10
          if (weightStr.empty())
          {
            std::cerr << "Aviso: peso vazio para evento " << eventIdx << ", membro " << iMember << std::endl;
            continue;
          }
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

  std::vector<double> cutflow_sumw2(5, 0.0);
  std::vector<double> mcc_scan_sumw2(mcc_thresholds.size(), 0.0);
  std::vector<std::vector<double>> pdf_cutflow_yields(5);
  if (nPdfMembers > 0)
    pdf_cutflow_yields.assign(5, std::vector<double>(static_cast<size_t>(nPdfMembers), 0.0));

  std::vector<TH1F *> h_mcc_pdf_members;
  std::vector<TH1F *> h_mcc_aftercuts_pdf_members;
  std::vector<TH1F *> h_ycc_pdf_members;
  std::vector<TH1F *> h_ycc_aftercuts_pdf_members;
  std::vector<TH1F *> h_ptcc_pdf_members;
  std::vector<TH1F *> h_ptcc_aftercuts_pdf_members;

  if (nPdfMembers > 0)
  {
    h_mcc_pdf_members = createPdfMemberHistograms(h_mcc, "h_mcc", nPdfMembers);
    h_mcc_aftercuts_pdf_members = createPdfMemberHistograms(h_mcc_aftercuts, "h_mcc_aftercuts", nPdfMembers);
    h_ycc_pdf_members = createPdfMemberHistograms(h_ycc, "h_ycc", nPdfMembers);
    h_ycc_aftercuts_pdf_members = createPdfMemberHistograms(h_ycc_aftercuts, "h_ycc_aftercuts", nPdfMembers);
    h_ptcc_pdf_members = createPdfMemberHistograms(h_ptcc, "h_ptcc", nPdfMembers);
    h_ptcc_aftercuts_pdf_members = createPdfMemberHistograms(h_ptcc_aftercuts, "h_ptcc_aftercuts", nPdfMembers);
  }

  std::cout << "Modo de objeto pesado: c-jets reco com FastJet (matching DeltaR com "
            << (tag_mode == "hadrons" ? "D mesons + barions charme status 2" : "c-quarks status 2")
            << ")" << std::endl;

  // Loop sobre eventos
  Long64_t iEvent = 0;
  while (reader.Next())
  {
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

    TRandom3 detector_rng(7000 + iEvent);

    weight = (nEntries > 0) ? (xs / nEntries) : 0.0;
    const double pdf_selected_weight = pdf_event_weight[static_cast<size_t>(iEvent)];
    double event_weight = weight * pdf_selected_weight;

    // Cutflow bin 1: todos os eventos gerados
    h_numberofevents_cutflow->Fill(0.0, event_weight);
    cutflow_sumw2[0] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[0][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
    }

    std::vector<TLorentzVector> c_quarks_truth = collectCQuarksStatus2(pdgID, status, px, py, pz, e);
    std::vector<TLorentzVector> charmed_hadrons = (tag_mode == "hadrons") ? collectCharmedHadrons(pdgID, status, px, py, pz, e) : std::vector<TLorentzVector>();

    // Usar a lista apropriada para tagging
    const std::vector<TLorentzVector> &tag_objects = (tag_mode == "hadrons") ? charmed_hadrons : c_quarks_truth;

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

    int n_cjets = 0;
    std::vector<bool> is_cjet = tagCJetsByDeltaR(jets, tag_objects, n_cjets);

    std::vector<TLorentzVector> cjets_reco;
    cjets_reco.reserve(static_cast<size_t>(n_cjets));
    for (size_t iJet = 0; iJet < jets.size(); ++iJet)
    {
      if (!is_cjet[iJet])
        continue;

      TLorentzVector cjet;
      cjet.SetPxPyPzE(jets[iJet].px(), jets[iJet].py(), jets[iJet].pz(), jets[iJet].E());
      cjets_reco.push_back(cjet);
    }

    if (cjets_reco.size() >= 1)
    {
      h_numberofevents_cutflow->Fill(1.0, event_weight);
      cutflow_sumw2[1] += event_weight * event_weight;
      if (nPdfMembers > 0)
      {
        const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
        for (int iMember = 0; iMember < nPdfMembers; ++iMember)
          pdf_cutflow_yields[1][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
      }
    }

    if (cjets_reco.size() < 2)
      continue;
    h_numberofevents_cutflow->Fill(2.0, event_weight);
    cutflow_sumw2[2] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[2][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
    }

    const auto &c1 = cjets_reco[0];
    const auto &c2 = cjets_reco[1];

    // Cortes cinemáticos (análogo ao fluxo de ttbar: objetos centrais e com pT mínimo)
    bool pass_kinematic_cuts = (c1.Pt() > 5.0 && c2.Pt() > 5.0 && fabs(c1.Eta()) < 2.5 && fabs(c2.Eta()) < 2.5);

    if (!pass_kinematic_cuts)
      continue;

    h_numberofevents_cutflow->Fill(3.0, event_weight);
    cutflow_sumw2[3] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[3][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
    }

    // Sistema cc
    TLorentzVector cc_system = c1 + c2;
    double mcc = cc_system.M();
    double ptcc = cc_system.Pt();
    double deta = c1.Eta() - c2.Eta();
    double dphi = c1.Phi() - c2.Phi();
    while (dphi > M_PI)
      dphi -= 2 * M_PI;
    while (dphi < -M_PI)
      dphi += 2 * M_PI;
    double deltaR_cc = sqrt(deta * deta + dphi * dphi);
    double ycc = cc_system.Rapidity();

    // Histogramas antes do último corte de massa
    h_mcc->Fill(mcc, event_weight);
    h_ptcc->Fill(ptcc, event_weight);
    h_ycc->Fill(ycc, event_weight);
    h_cc_deltaR->Fill(deltaR_cc, event_weight);
    h_cc_dphi->Fill(fabs(dphi), event_weight);
    h_c1_pt->Fill(c1.Pt(), event_weight);
    h_c2_pt->Fill(c2.Pt(), event_weight);
    h_c1_eta->Fill(c1.Eta(), event_weight);
    h_c2_eta->Fill(c2.Eta(), event_weight);

    for (size_t iThr = 0; iThr < mcc_thresholds.size(); ++iThr)
    {
      if (mcc > mcc_thresholds[iThr])
      {
        h_mcc_threshold_hists[iThr]->Fill(mcc, event_weight);
        h_ptcc_threshold_hists[iThr]->Fill(ptcc, event_weight);
        h_ycc_threshold_hists[iThr]->Fill(ycc, event_weight);
        h_mcc_cut_index->Fill(static_cast<double>(iThr + 1), event_weight);
        mcc_scan_sumw2[iThr] += event_weight * event_weight;
      }
    }

    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
      {
        const double memberWeight = weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
        h_mcc_pdf_members[static_cast<size_t>(iMember)]->Fill(mcc, memberWeight);
        h_ycc_pdf_members[static_cast<size_t>(iMember)]->Fill(ycc, memberWeight);
        h_ptcc_pdf_members[static_cast<size_t>(iMember)]->Fill(ptcc, memberWeight);
      }
    }

    if (mcc < 10.0)
      continue;

    h_numberofevents_cutflow->Fill(4.0, event_weight);
    cutflow_sumw2[4] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[4][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
    }

    // Histogramas após todos os cortes
    h_mcc_aftercuts->Fill(mcc, event_weight);
    for (size_t iThr = 0; iThr < mcc_thresholds.size(); ++iThr)
    {
      if (mcc > mcc_thresholds[iThr])
      {
        h_mcc_threshold_aftercuts_hists[iThr]->Fill(mcc, event_weight);
        h_ptcc_threshold_aftercuts_hists[iThr]->Fill(ptcc, event_weight);
        h_ycc_threshold_aftercuts_hists[iThr]->Fill(ycc, event_weight);
      }
    }
    h_ptcc_aftercuts->Fill(ptcc, event_weight);
    h_cc_deltaR_aftercuts->Fill(deltaR_cc, event_weight);
    h_cc_dphi_aftercuts->Fill(fabs(dphi), event_weight);
    h_c1_pt_aftercuts->Fill(c1.Pt(), event_weight);
    h_c2_pt_aftercuts->Fill(c2.Pt(), event_weight);
    h_c1_eta_aftercuts->Fill(c1.Eta(), event_weight);
    h_c2_eta_aftercuts->Fill(c2.Eta(), event_weight);
    h_ycc_aftercuts->Fill(ycc, event_weight);
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
      {
        const double memberWeight = weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
        h_mcc_aftercuts_pdf_members[static_cast<size_t>(iMember)]->Fill(mcc, memberWeight);
        h_ycc_aftercuts_pdf_members[static_cast<size_t>(iMember)]->Fill(ycc, memberWeight);
        h_ptcc_aftercuts_pdf_members[static_cast<size_t>(iMember)]->Fill(ptcc, memberWeight);
      }
    }

    // Progresso
    if (iEvent % 1000 == 0)
    {
      std::cout << "Processado: " << iEvent << " / " << nEntries
                << " (" << (100.0 * iEvent / nEntries) << "%)" << std::endl;
    }

    ++iEvent;
  }

  FILE *ccanalysis;
  ccanalysis = fopen(txt_name.c_str(), "a");
  fprintf(ccanalysis, "%s\n", "===========================================================");
  fprintf(ccanalysis, "%s\n", "========== OO -> CC Analysis (FastJet c-jets) ==============");
  fprintf(ccanalysis, "%s\n", "===========================================================");
  const char *tag_desc = (tag_mode == "hadrons") ? "c-jets via FastJet anti-kt R=0.4 + DeltaR(D/Lambda_c/Xi_c/... status2, jet)<0.4" : "c-jets via FastJet anti-kt R=0.4 + DeltaR(c-quark status2, jet)<0.4";
  fprintf(ccanalysis, "%s%s\n", "Object mode: ", tag_desc);
  fprintf(ccanalysis, "%s%.3f%s\n", "Cross Section: ", xs, " nb");
  fprintf(ccanalysis, "%s%.1f%s\n", "Integrated Luminosity: ", lumi, " nb^-1");
  fprintf(ccanalysis, "%s%.2f\n", "Expected events: ", xs * lumi);
  fprintf(ccanalysis, "%s%lld\n", "Generated events: ", nEntries);
  fprintf(ccanalysis, "%s\n", "===========================================================");
  fprintf(ccanalysis, "%s\n", "=================== Cut Flow ==============================");
  fprintf(ccanalysis, "%s\n", "===========================================================");
  fprintf(ccanalysis, "%s\n", "Corte                                  | Eventos (xs) +- stat | Eficiência | N(L=8nb^-1) +- stat");
  fprintf(ccanalysis, "%s\n", "---------------------------------------------------------");

  // Check adicional de eficiencia para >=1 c-jet reconstruido no cutflow.

  auto printCutflowLine = [&](const char *label, int stageIndex, int binIndex)
  {
    const double nominal = h_numberofevents_cutflow->GetBinContent(binIndex);
    const double reference = h_numberofevents_cutflow->GetBinContent(1);
    const double efficiency = (reference > 0.0) ? 100.0 * nominal / reference : 0.0;
    const double statUnc = std::sqrt(cutflow_sumw2[static_cast<size_t>(stageIndex)]);
    HessianUncertainty pdfUnc;
    if (!pdf_cutflow_yields.empty())
      pdfUnc = computeHessianUncertainty(pdf_cutflow_yields[static_cast<size_t>(stageIndex)]);

    fprintf(ccanalysis,
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
  printCutflowLine(">= 1 c-jet reconstruido", 1, 2);
  printCutflowLine(">= 2 c-jets reconstruidos", 2, 3);
  printCutflowLine("c1,c2: pT>5 GeV, |eta|<2.5", 3, 4);
  printCutflowLine("m_cc > 10 GeV", 4, 5);

  fprintf(ccanalysis, "%s\n", "---------------------------------------------------------");
  fprintf(ccanalysis, "%s\n", "Indexed mass-threshold scan (after kinematic cuts)");
  fprintf(ccanalysis, "%s\n", "idx  label    Yield [nb]    stat [nb]    Eff(vs pre-mass) [%]");
  const double preMassYield = h_numberofevents_cutflow->GetBinContent(4);
  for (size_t iThr = 0; iThr < mcc_thresholds.size(); ++iThr)
  {
    const int bin = static_cast<int>(iThr + 1);
    const double y = h_mcc_cut_index->GetBinContent(bin);
    const double stat = std::sqrt(mcc_scan_sumw2[iThr]);
    const double eff = (preMassYield > 0.0) ? 100.0 * y / preMassYield : 0.0;
    std::ostringstream label;
    label << "mcc" << static_cast<int>(mcc_thresholds[iThr]);
    fprintf(ccanalysis, "%3d  %-7s %12.4f %12.4f %14.3f\n", bin, label.str().c_str(), y, stat, eff);
  }

  fprintf(ccanalysis, "%s\n", "===========================================================");
  fprintf(ccanalysis, "\n");
  fclose(ccanalysis);

  // Salvar histogramas
  TFile *output_file = new TFile(output_name.c_str(), "RECREATE");

  auto mccPdfVars = buildPdfVariationHistograms(h_mcc_pdf_members, "h_mcc_pdf_up", "h_mcc_pdf_down");
  auto mccAftercutsPdfVars = buildPdfVariationHistograms(h_mcc_aftercuts_pdf_members, "h_mcc_aftercuts_pdf_up", "h_mcc_aftercuts_pdf_down");
  auto ptccPdfVars = buildPdfVariationHistograms(h_ptcc_pdf_members, "h_ptcc_pdf_up", "h_ptcc_pdf_down");
  auto ptccAftercutsPdfVars = buildPdfVariationHistograms(h_ptcc_aftercuts_pdf_members, "h_ptcc_aftercuts_pdf_up", "h_ptcc_aftercuts_pdf_down");
  auto yccPdfVars = buildPdfVariationHistograms(h_ycc_pdf_members, "h_ycc_pdf_up", "h_ycc_pdf_down");
  auto yccAftercutsPdfVars = buildPdfVariationHistograms(h_ycc_aftercuts_pdf_members, "h_ycc_aftercuts_pdf_up", "h_ycc_aftercuts_pdf_down");

  h_mcc->Write();
  if (mccPdfVars.first)
  {
    mccPdfVars.first->Write();
    mccPdfVars.second->Write();
  }
  h_ptcc->Write();
  h_ycc->Write();
  h_cc_deltaR->Write();
  h_cc_dphi->Write();
  h_c1_pt->Write();
  h_c2_pt->Write();
  h_c1_eta->Write();
  h_c2_eta->Write();

  h_numberofevents_cutflow->Write();
  h_mcc_cut_index->Write();
  for (TH1F *h_thr : h_mcc_threshold_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_mcc_threshold_aftercuts_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_ptcc_threshold_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_ptcc_threshold_aftercuts_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_ycc_threshold_hists)
    h_thr->Write();
  for (TH1F *h_thr : h_ycc_threshold_aftercuts_hists)
    h_thr->Write();
  if (mccAftercutsPdfVars.first)
  {
    mccAftercutsPdfVars.first->Write();
    mccAftercutsPdfVars.second->Write();
  }
  if (ptccPdfVars.first)
  {
    ptccPdfVars.first->Write();
    ptccPdfVars.second->Write();
  }
  if (ptccAftercutsPdfVars.first)
  {
    ptccAftercutsPdfVars.first->Write();
    ptccAftercutsPdfVars.second->Write();
  }
  if (yccAftercutsPdfVars.first)
  {
    yccAftercutsPdfVars.first->Write();
    yccAftercutsPdfVars.second->Write();
  }
  h_ycc_aftercuts->Write();
  h_ptcc_aftercuts->Write();
  h_mcc_aftercuts->Write();
  h_cc_deltaR_aftercuts->Write();
  h_cc_dphi_aftercuts->Write();
  h_c1_pt_aftercuts->Write();
  h_c2_pt_aftercuts->Write();
  h_c1_eta_aftercuts->Write();
  h_c2_eta_aftercuts->Write();
  ;
  if (yccPdfVars.first)
  {
    yccPdfVars.first->Write();
    yccPdfVars.second->Write();
  }
  if (yccAftercutsPdfVars.first)
  {
    yccAftercutsPdfVars.first->Write();
    yccAftercutsPdfVars.second->Write();
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