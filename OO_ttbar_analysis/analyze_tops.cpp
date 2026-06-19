/*
 * Análise de produção de tops em colisões O-O
 * Lê árvore ROOT de LHE e faz análise das propriedades dos tops
 * Inclui reconstrução de tops semileptônicos com FastJet
 *
 * Compilar: g++ -o analyze_tops analyze_tops.cpp `root-config --cflags --libs` `fastjet-config --cxxflags --libs --plugins`
 * Executar: ./analyze_tops input.root output.root
 */

#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLorentzVector.h>
#include <TRandom3.h>
#include <TTree.h>
#include <algorithm>
#include <cmath>
#include <ctime>
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

  unc.up = std::sqrt(up2);
  unc.down = std::sqrt(down2);
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

double computeWeight(const TLorentzVector &top, const TLorentzVector &nu, double met_x, double met_y)
{
  double px_diff = nu.Px() - met_x;
  double py_diff = nu.Py() - met_y;
  return std::exp(-0.5 * (px_diff * px_diff + py_diff * py_diff) / (20.0 * 20.0)); // width 20 GeV, por exemplo
}

bool solveNeutrinoPz(const TLorentzVector &lep, double met_ex, double met_ey,
                     double &pz_sol1, double &pz_sol2, bool &hasSolution)
{
  const double mW = 80.379;
  const double px = met_ex;
  const double py = met_ey;

  double lepE = lep.E();
  double lepPx = lep.Px();
  double lepPy = lep.Py();
  double lepPz = lep.Pz();

  double mu = mW * mW / 2.0 + px * lepPx + py * lepPy;
  double A = lepE * lepE - lepPz * lepPz;
  double B = -2.0 * mu * lepPz;
  double C = lepE * lepE * (px * px + py * py) - mu * mu;

  double disc = B * B - 4.0 * A * C;
  hasSolution = (disc >= 0.0);

  double sqrtD = std::sqrt(std::max(0.0, disc));
  if (fabs(A) < 1e-9)
  {
    // degenerate: linear equation
    if (fabs(B) < 1e-9)
      pz_sol1 = 0.0;
    else
      pz_sol1 = -C / B;
    pz_sol2 = pz_sol1;
  }
  else
  {
    pz_sol1 = (-B + sqrtD) / (2.0 * A);
    pz_sol2 = (-B - sqrtD) / (2.0 * A);
  }
  return true;
}

// weight function: gaussian on (px,py) difference
inline double metWeight(double dx, double dy, double sigma_px = 10.0, double sigma_py = 10.0)
{
  double wx = std::exp(-0.5 * (dx * dx) / (sigma_px * sigma_px));
  double wy = std::exp(-0.5 * (dy * dy) / (sigma_py * sigma_py));
  return wx * wy;
}

double neutrinoDiscriminant(const TLorentzVector &lep, double met_ex, double met_ey)
{
  const double mW = 80.379;
  const double mu = mW * mW / 2.0 + met_ex * lep.Px() + met_ey * lep.Py();
  const double A = lep.E() * lep.E() - lep.Pz() * lep.Pz();
  const double B = -2.0 * mu * lep.Pz();
  const double C = lep.E() * lep.E() * (met_ex * met_ex + met_ey * met_ey) - mu * mu;
  return B * B - 4.0 * A * C;
}

bool correctMETForWConstraint(const TLorentzVector &lep, double met_ex, double met_ey,
                              double &met_ex_corr, double &met_ey_corr)
{
  met_ex_corr = met_ex;
  met_ey_corr = met_ey;

  if (neutrinoDiscriminant(lep, met_ex, met_ey) >= 0.0)
    return false;

  double best_r2 = 1e12;
  bool found = false;

  // Procura o menor deslocamento em MET que restaura solução real para pz(nu).
  const double rmax = 120.0;
  const double dr = 2.0;
  const int nphi = 72;
  for (double r = dr; r <= rmax; r += dr)
  {
    for (int i = 0; i < nphi; ++i)
    {
      const double phi = (2.0 * M_PI * i) / nphi;
      const double test_x = met_ex + r * std::cos(phi);
      const double test_y = met_ey + r * std::sin(phi);

      if (neutrinoDiscriminant(lep, test_x, test_y) < 0.0)
        continue;

      const double dx = test_x - met_ex;
      const double dy = test_y - met_ey;
      const double r2 = dx * dx + dy * dy;
      if (r2 < best_r2)
      {
        best_r2 = r2;
        met_ex_corr = test_x;
        met_ey_corr = test_y;
        found = true;
      }
    }
    if (found)
      break;
  }

  return found;
}

struct NeutrinoSolution
{
  TLorentzVector nu;
  double met_chi2;
};

std::vector<NeutrinoSolution> reconstructNeutrinoCandidates(const TLorentzVector &lep,
                                                            double met_ex,
                                                            double met_ey,
                                                            double sigma_met = 20.0)
{
  double met_ex_fit = met_ex;
  double met_ey_fit = met_ey;
  correctMETForWConstraint(lep, met_ex, met_ey, met_ex_fit, met_ey_fit);

  double pz1 = 0.0;
  double pz2 = 0.0;
  bool hasSol = false;
  solveNeutrinoPz(lep, met_ex_fit, met_ey_fit, pz1, pz2, hasSol);

  const double dx = met_ex_fit - met_ex;
  const double dy = met_ey_fit - met_ey;
  const double met_chi2 = (dx * dx + dy * dy) / (sigma_met * sigma_met);

  std::vector<NeutrinoSolution> solutions;
  const double e1 = std::sqrt(met_ex_fit * met_ex_fit + met_ey_fit * met_ey_fit + pz1 * pz1);
  TLorentzVector nu1;
  nu1.SetPxPyPzE(met_ex_fit, met_ey_fit, pz1, e1);
  solutions.push_back({nu1, met_chi2});

  if (std::fabs(pz2 - pz1) > 1e-6)
  {
    const double e2 = std::sqrt(met_ex_fit * met_ex_fit + met_ey_fit * met_ey_fit + pz2 * pz2);
    TLorentzVector nu2;
    nu2.SetPxPyPzE(met_ex_fit, met_ey_fit, pz2, e2);
    solutions.push_back({nu2, met_chi2});
  }

  return solutions;
}

double leptonEfficiency(double pt, double eta, int absPdg)
{
  const double abseta = std::abs(eta);

  if (absPdg == 11)
  {
    // Electron: include crack veto in efficiency model.
    const bool in_crack = (abseta > 1.37 && abseta < 1.52);
    if (pt < 18.0 || abseta > 2.5 || in_crack)
      return 0.0;
    return (abseta < 1.37) ? 0.80 : 0.70;
  }

  if (absPdg == 13)
  {
    // Muon: no crack region in this simplified model.
    if (pt < 18.0 || abseta > 2.5)
      return 0.0;
    return (abseta < 1.5) ? 0.85 : 0.80;
  }

  return 0.0;
}

double bTagEfficiency(double, double)
{
  // PbPb-like WP from internal note: 85.43% b-tag efficiency in ttbar
  return 0.8543;
}

double mistagCharmRate(double, double)
{
  // PbPb-like WP from internal note
  return 0.346;
}

double mistagLightRate(double, double)
{
  // PbPb-like WP from internal note
  return 0.0248;
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

void smearMET(double &met_x, double &met_y, TRandom3 &rng)
{
  constexpr double sigma_met = 20.0;
  met_x += rng.Gaus(0.0, sigma_met);
  met_y += rng.Gaus(0.0, sigma_met);
}

std::vector<bool> tagBJetsByDeltaR(const std::vector<fastjet::PseudoJet> &jets,
                                   const std::vector<TLorentzVector> &b_quarks,
                                   const std::vector<TLorentzVector> &c_quarks,
                                   TRandom3 &rng,
                                   int &n_bjets)
{
  std::vector<bool> is_bjet(jets.size(), false);
  n_bjets = 0;

  for (size_t iJet = 0; iJet < jets.size(); ++iJet)
  {
    const double jet_eta = jets[iJet].eta();
    const double jet_phi = jets[iJet].phi();
    bool matched_to_b = false;
    bool matched_to_c = false;

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
        matched_to_b = true;
        break;
      }
    }

    if (!matched_to_b)
    {
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
          matched_to_c = true;
          break;
        }
      }
    }

    double tag_probability = mistagLightRate(jets[iJet].pt(), jet_eta);
    if (matched_to_b)
      tag_probability = bTagEfficiency(jets[iJet].pt(), jet_eta);
    else if (matched_to_c)
      tag_probability = mistagCharmRate(jets[iJet].pt(), jet_eta);

    if (rng.Uniform() < tag_probability)
    {
      is_bjet[iJet] = true;
      ++n_bjets;
    }
  }

  return is_bjet;
}

std::vector<TLorentzVector> collectBQuarks(const std::vector<int> &pdgID,
                                           const std::vector<int> &status,
                                           const std::vector<float> &px,
                                           const std::vector<float> &py,
                                           const std::vector<float> &pz,
                                           const std::vector<float> &e)
{
  std::vector<TLorentzVector> b_quarks;
  for (size_t iPart = 0; iPart < pdgID.size(); ++iPart)
  {
    if (std::abs(pdgID[iPart]) == 5 && (status[iPart] == 1 || status[iPart] == 23))
    {
      TLorentzVector b_quark;
      b_quark.SetPxPyPzE(px[iPart], py[iPart], pz[iPart], e[iPart]);
      b_quarks.push_back(b_quark);
    }
  }
  return b_quarks;
}

std::vector<TLorentzVector> collectCQuarks(const std::vector<int> &pdgID,
                                           const std::vector<int> &status,
                                           const std::vector<float> &px,
                                           const std::vector<float> &py,
                                           const std::vector<float> &pz,
                                           const std::vector<float> &e)
{
  std::vector<TLorentzVector> c_quarks;
  for (size_t iPart = 0; iPart < pdgID.size(); ++iPart)
  {
    if (std::abs(pdgID[iPart]) == 4 && (status[iPart] == 1 || status[iPart] == 23))
    {
      TLorentzVector c_quark;
      c_quark.SetPxPyPzE(px[iPart], py[iPart], pz[iPart], e[iPart]);
      c_quarks.push_back(c_quark);
    }
  }
  return c_quarks;
}

std::vector<TLorentzVector> collectSelectedLeptons(const std::vector<int> &pdgID,
                                                   const std::vector<int> &status,
                                                   const std::vector<float> &px,
                                                   const std::vector<float> &py,
                                                   const std::vector<float> &pz,
                                                   const std::vector<float> &e,
                                                   std::vector<int> &selectedLeptonPdg)
{
  selectedLeptonPdg.clear();
  std::vector<TLorentzVector> leptons;
  std::vector<std::pair<TLorentzVector, int>> selected;
  for (size_t iPart = 0; iPart < pdgID.size(); ++iPart)
  {
    if (status[iPart] != 1)
      continue;
    const int absPdg = std::abs(pdgID[iPart]);
    if (absPdg != 11 && absPdg != 13)
      continue;

    TLorentzVector lep;
    lep.SetPxPyPzE(px[iPart], py[iPart], pz[iPart], e[iPart]);

    if (absPdg == 11)
    {
      // Electron: MediumLH-like kinematics from PbPb note.
      const double abseta = std::abs(lep.Eta());
      const bool in_crack = (abseta > 1.37 && abseta < 1.52);
      if (lep.Pt() > 18.0 && lep.Et() > 18.0 && abseta < 2.5 && !in_crack)
        selected.push_back({lep, absPdg});
    }
    else if (absPdg == 13)
    {
      // Muon: Medium-like kinematics from PbPb note.
      if (lep.Pt() > 18.0 && std::abs(lep.Eta()) < 2.5)
        selected.push_back({lep, absPdg});
    }
  }

  std::sort(selected.begin(), selected.end(), [](const std::pair<TLorentzVector, int> &a, const std::pair<TLorentzVector, int> &b)
            { return a.first.Pt() > b.first.Pt(); });

  leptons.reserve(selected.size());
  selectedLeptonPdg.reserve(selected.size());
  for (const auto &lepInfo : selected)
  {
    leptons.push_back(lepInfo.first);
    selectedLeptonPdg.push_back(lepInfo.second);
  }

  return leptons;
}

// Contadores
double weight = 0.0;
double xs = 0.0; // nb isospin nPDFSet0
// double xs = 23.4; // nb isospin nPDFSet3
// double xs = 23.4; // nb isospin nCTeq16_8
double lumi = 8.0; // nb-1
double bestjet1_pt = 0.0;
double bestjet2_pt = 0.0;
double bestbjet_pt = 0.0;
TLorentzVector lepton;

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

  std::string txt_name = (argc >= 5) ? argv[4] : "out_top.txt";
  // Inclui .txt opcional para salvar os resultados dos cortes.

  std::string pdf_weights_csv = (argc >= 6) ? argv[5] : "";
  int pdf_member = (argc >= 7) ? std::atoi(argv[6]) : 0;

  // Abrir arquivo de entrada
  TFile *input_file = TFile::Open(argv[1]);
  if (!input_file || input_file->IsZombie())
  {
    std::cerr << "Erro ao abrir arquivo: " << argv[1] << std::endl;
    return 1;
  }

  // Obter a árvore
  TTree *tree = (TTree *)input_file->Get("lheTree");
  if (!tree)
  {
    std::cerr << "Erro: árvore 'lheTree' não encontrada!" << std::endl;
    input_file->Close();
    return 1;
  }

  // Variáveis para leitura
  std::vector<int> *pdgID = nullptr;
  std::vector<int> *status = nullptr;
  std::vector<float> *px = nullptr;
  std::vector<float> *py = nullptr;
  std::vector<float> *pz = nullptr;
  std::vector<float> *e = nullptr;
  std::vector<float> *m = nullptr;

  tree->SetBranchAddress("pdgID", &pdgID);
  tree->SetBranchAddress("status", &status);
  tree->SetBranchAddress("px", &px);
  tree->SetBranchAddress("py", &py);
  tree->SetBranchAddress("pz", &pz);
  tree->SetBranchAddress("e", &e);
  tree->SetBranchAddress("m", &m);

  // Histogramas - tops reconstruídos com FastJet (j1j2b)
  TH1F *h_njets = new TH1F("h_njets", "Número de jatos (pT > 25 GeV);N_{jatos};Eventos", 20, 0, 20);
  TH1F *h_nbjets = new TH1F("h_nbjets", "Número de b-jatos (#DeltaR < 0.4);N_{b-jatos};Eventos", 10, 0, 10);
  TH1F *h_reco_top_mass = new TH1F("h_reco_top_mass", "Massa do top reconstruido (jjb);M_{jjb} [GeV];Eventos", 100, 100, 250);
  TH1F *h_reco_top_pt = new TH1F("h_reco_top_pt", "pT do top reconstruido;pT [GeV];Eventos", 100, 0, 500);

  // Histogramas - top leptônico
  TH1F *h_nlep = new TH1F("h_nlep", "Número de leptons (e/#mu, pT > 20 GeV);N_{lep};Eventos", 10, 0, 10);
  TH1F *h_met = new TH1F("h_met", "Missing ET;MET [GeV];Eventos", 100, 0, 300);
  TH1F *h_reco_top_lep_mass = new TH1F("h_reco_top_lep_mass", "Massa do top leptonico (l#nub);M_{l#nub} [GeV];Eventos", 100, 100, 250);
  TH1F *h_reco_top_lep_pt = new TH1F("h_reco_top_lep_pt", "pT do top leptonico;pT [GeV];Eventos", 100, 0, 500);
  TH1F *h_reco_W_lep_mass = new TH1F("h_reco_W_lep_mass", "Massa do W leptonico (l#nu);M_{l#nu} [GeV];Eventos", 100, 0, 150);
  TH2F *h_mtop_had_vs_lep = new TH2F("h_mtop_had_vs_lep", "M_{top had} vs M_{top lep};M_{jjb} [GeV];M_{l#nub} [GeV]", 50, 100, 250, 50, 100, 250);
  TH1F *h_mtt = new TH1F("h_mtt", "Massa invariante t#bar{t};M_{t#bar{t}} [GeV];Eventos", 100, 200, 1000);

  // Histogramas sensíveis a nPDFs
  TH1F *h_top_had_y = new TH1F("h_top_had_y", "Rapidez do top hadronico;y;Eventos", 60, -3, 3);
  TH1F *h_top_lep_y = new TH1F("h_top_lep_y", "Rapidez do top leptonico;y;Eventos", 60, -3, 3);
  TH1F *h_ttbar_y = new TH1F("h_ttbar_y", "Rapidez do sistema t#bar{t};y_{t#bar{t}};Eventos", 60, -3, 3);
  TH1F *h_ttbar_pt = new TH1F("h_ttbar_pt", "pT do sistema t#bar{t};pT_{t#bar{t}} [GeV];Eventos", 100, 0, 500);
  TH1F *h_delta_y_tt = new TH1F("h_delta_y_tt", "#Deltay(t, #bar{t});#Deltay;Eventos", 100, -5, 5);
  TH1F *h_HT = new TH1F("h_HT", "H_{T} (soma escalar pT jatos);H_{T} [GeV];Eventos", 100, 0, 800);
  TH2F *h_ttbar_y_vs_pt = new TH2F("h_ttbar_y_vs_pt", "Rapidez vs pT do sistema t#bar{t};y_{t#bar{t}};pT_{t#bar{t}} [GeV]", 60, -3, 3, 100, 0, 500);

  TH1F *h_numberofevents_cutflow = new TH1F("h_numberofevents_cutflow", "", 10.0, 0, 10.0);

  TH1F *h_bestjet1_pt = new TH1F("h_bestjet1_pt", "pT do melhor jato 1;pT [GeV];Eventos", 100, 0, 500);
  TH1F *h_bestjet2_pt = new TH1F("h_bestjet2_pt", "pT do melhor jato 2;pT [GeV];Eventos", 100, 0, 500);
  TH1F *h_bestbjet_pt = new TH1F("h_bestbjet_pt", "pT do melhor b-jato;pT [GeV];Eventos", 100, 0, 500);
  TH1F *h_lep_pt_reco = new TH1F("h_lep_pt_reco", "pT dos leptons;pT [GeV];Leptons", 100, 0, 500);
  TH1F *h_lep_eta_reco = new TH1F("h_lep_eta_reco", "#eta dos leptons;#eta;Leptons", 100, -5, 5);

  //=======================Afeter Cuts plots ===========================
  TH1F *h_lep_pt_aftercuts = new TH1F("h_lep_pt_aftercuts", "pT dos leptons apos cortes;pT [GeV];Leptons", 100, 0, 500);
  TH1F *h_lep_eta_aftercuts = new TH1F("h_lep_eta_aftercuts", "#eta dos leptons apos cortes;#eta;Leptons", 100, -5, 5);
  TH1F *h_bestjet1_pt_aftercuts = new TH1F("h_bestjet1_pt_aftercuts", "pT do melhor jato 1 apos cortes;pT [GeV];Eventos", 100, 0, 500);
  TH1F *h_bestjet2_pt_aftercuts = new TH1F("h_bestjet2_pt_aftercuts", "pT do melhor jato 2 apos cortes;pT [GeV];Eventos", 100, 0, 500);
  TH1F *h_bestbjet_pt_aftercuts = new TH1F("h_bestbjet_pt_aftercuts", "pT do melhor b-jato apos cortes;pT [GeV];Eventos", 100, 0, 500);
  TH1F *h_bestjet1_eta_aftercuts = new TH1F("h_bestjet1_eta_aftercuts", "#eta do melhor jato 1 apos cortes;#eta;Eventos", 100, -5, 5);
  TH1F *h_bestjet2_eta_aftercuts = new TH1F("h_bestjet2_eta_aftercuts", "#eta do melhor jato 2 apos cortes;#eta;Eventos", 100, -5, 5);
  TH1F *h_bestbjet_eta_aftercuts = new TH1F("h_bestbjet_eta_aftercuts", "#eta do melhor b-jato apos cortes;#eta;Eventos", 100, -5, 5);
  // Histogramas nPDF apos cortes
  TH1F *h_top_had_y_aftercuts = new TH1F("h_top_had_y_aftercuts", "Rapidez do top hadronico apos cortes;y;Eventos", 60, -3, 3);
  TH1F *h_top_lep_y_aftercuts = new TH1F("h_top_lep_y_aftercuts", "Rapidez do top leptonico apos cortes;y;Eventos", 60, -3, 3);
  TH1F *h_ttbar_y_aftercuts = new TH1F("h_ttbar_y_aftercuts", "Rapidez do sistema t#bar{t} apos cortes;y_{t#bar{t}};Eventos", 60, -3, 3);
  TH1F *h_ttbar_pt_aftercuts = new TH1F("h_ttbar_pt_aftercuts", "pT do sistema t#bar{t} apos cortes;pT_{t#bar{t}} [GeV];Eventos", 100, 0, 500);
  TH1F *h_delta_y_tt_aftercuts = new TH1F("h_delta_y_tt_aftercuts", "#Deltay(t, #bar{t}) apos cortes;#Deltay;Eventos", 100, -5, 5);
  TH1F *h_mtt_aftercuts = new TH1F("h_mtt_aftercuts", "Massa invariante t#bar{t} apos cortes;M_{t#bar{t}} [GeV];Eventos", 100, 200, 1000);
  TH1F *h_HT_aftercuts = new TH1F("h_HT_aftercuts", "H_{T} apos cortes;H_{T} [GeV];Eventos", 100, 0, 800);
  TH1F *h_reco_top_mass_aftercuts = new TH1F("h_reco_top_mass_aftercuts", "Massa do top hadronico apos cortes;M_{jjb} [GeV];Eventos", 100, 100, 250);
  TH1F *h_reco_top_lep_mass_aftercuts = new TH1F("h_reco_top_lep_mass_aftercuts", "Massa do top leptonico apos cortes;M_{l#nub} [GeV];Eventos", 100, 100, 250);
  TH2F *h_ttbar_y_vs_pt_aftercuts = new TH2F("h_ttbar_y_vs_pt_aftercuts", "Rapidez vs pT do sistema t#bar{t} apos cortes;y_{t#bar{t}};pT_{t#bar{t}} [GeV]", 60, -3, 3, 100, 0, 500);

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

    const int nMembersCsv = static_cast<int>(nColsHeader) - 7;
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
        cols.push_back(cell);

      if (cols.size() < static_cast<size_t>(7 + nMembersCsv))
        continue;

      const size_t eventIdx = static_cast<size_t>(std::stoll(cols[0]));
      if (eventIdx >= static_cast<size_t>(nEntries))
        continue;

      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
      {
        const double wMember = std::stod(cols[7 + iMember]);
        pdf_all_event_weights[eventIdx * static_cast<size_t>(nPdfMembers) + static_cast<size_t>(iMember)] = wMember;
        if (iMember == pdf_member)
          pdf_event_weight[eventIdx] = wMember;
      }
      ++loaded;
    }

    std::cout << "CSV de pesos carregado: " << loaded
              << " eventos, membro " << pdf_member
              << " de " << nMembersCsv << " membros." << std::endl;
  }

  std::vector<double> cutflow_sumw2(4, 0.0);
  std::vector<std::vector<double>> pdf_cutflow_yields(4);
  if (nPdfMembers > 0)
    pdf_cutflow_yields.assign(4, std::vector<double>(static_cast<size_t>(nPdfMembers), 0.0));

  std::vector<TH1F *> h_mtt_pdf_members;
  std::vector<TH1F *> h_ttbar_y_pdf_members;
  std::vector<TH1F *> h_ttbar_pt_pdf_members;
  std::vector<TH1F *> h_mtt_aftercuts_pdf_members;
  std::vector<TH1F *> h_ttbar_y_aftercuts_pdf_members;
  std::vector<TH1F *> h_ttbar_pt_aftercuts_pdf_members;
  if (nPdfMembers > 0)
  {
    h_mtt_pdf_members = createPdfMemberHistograms(h_mtt, "h_mtt", nPdfMembers);
    h_ttbar_y_pdf_members = createPdfMemberHistograms(h_ttbar_y, "h_ttbar_y", nPdfMembers);
    h_ttbar_pt_pdf_members = createPdfMemberHistograms(h_ttbar_pt, "h_ttbar_pt", nPdfMembers);
    h_mtt_aftercuts_pdf_members = createPdfMemberHistograms(h_mtt_aftercuts, "h_mtt_aftercuts", nPdfMembers);
    h_ttbar_y_aftercuts_pdf_members = createPdfMemberHistograms(h_ttbar_y_aftercuts, "h_ttbar_y_aftercuts", nPdfMembers);
    h_ttbar_pt_aftercuts_pdf_members = createPdfMemberHistograms(h_ttbar_pt_aftercuts, "h_ttbar_pt_aftercuts", nPdfMembers);
  }

  // Loop sobre eventos
  for (Long64_t iEvent = 0; iEvent < nEntries; iEvent++)
  {
    tree->GetEntry(iEvent);
    lepton.SetPxPyPzE(0., 0., 0., 0.);
    TRandom3 detector_rng(1000 + iEvent);

    weight = xs / nEntries; // isospin nPDFSet0
    const double pdf_selected_weight = pdf_event_weight[static_cast<size_t>(iEvent)];
    const double event_weight_before_lep = weight * pdf_selected_weight;
    double event_weight = event_weight_before_lep;
    double lepton_weight_factor = 1.0;

    h_numberofevents_cutflow->Fill(0.0, event_weight_before_lep);
    cutflow_sumw2[0] += event_weight_before_lep * event_weight_before_lep;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[0][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)];
    }

    // Loop sobre partículas
    for (size_t iPart = 0; iPart < pdgID->size(); iPart++)
    {
      int pdg = pdgID->at(iPart);
    }
    // Reconstrução com FastJet - usar partículas finais (status == 1)
    std::vector<fastjet::PseudoJet> particles;

    // Primeiro, identificar quarks b gerados (para b-tagging)
    std::vector<TLorentzVector> b_quarks = collectBQuarks(*pdgID, *status, *px, *py, *pz, *e);
    std::vector<TLorentzVector> c_quarks = collectCQuarks(*pdgID, *status, *px, *py, *pz, *e);

    for (size_t iPart = 0; iPart < pdgID->size(); iPart++)
    {
      int pdg = pdgID->at(iPart);
      int stat = status->at(iPart);

      // Selecionar partículas finais carregadas e neutras para clustering
      // Excluir neutrinos (12, 14, 16) e partículas muito pesadas
      if (stat == 1 && abs(pdg) != 12 && abs(pdg) != 14 && abs(pdg) != 16)
      {
        fastjet::PseudoJet particle(px->at(iPart), py->at(iPart), pz->at(iPart), e->at(iPart));
        particle.set_user_index(iPart); // guardar índice original
        particles.push_back(particle);
      }
    }

    // Clustering com FastJet - algoritmo anti-kt, R = 0.4
    double R = 0.4;
    fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
    fastjet::ClusterSequence cs(particles, jet_def);

    // Selecionar jatos com pT > 25 GeV
    double ptmin = 25.0;
    std::vector<fastjet::PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(ptmin));
    std::vector<fastjet::PseudoJet> reco_jets;
    for (const auto &jet : jets)
    {
      fastjet::PseudoJet smeared_jet = smearJet(jet, detector_rng);
      if (smeared_jet.pt() > ptmin)
        reco_jets.push_back(smeared_jet);
    }
    jets = sorted_by_pt(reco_jets);

    h_njets->Fill(jets.size(), event_weight);

    // Identificar b-jets usando ΔR < 0.4 com quarks b gerados
    int n_bjets = 0;
    std::vector<bool> is_bjet = tagBJetsByDeltaR(jets, b_quarks, c_quarks, detector_rng, n_bjets);

    h_nbjets->Fill(n_bjets, event_weight);

    // Reconstrução de top hadrônico: 1 b-jet + 2 jatos leves
    // Procurar a melhor combinação
    TLorentzVector best_top_had;
    std::vector<size_t> best_had_indices;
    bool found_had_top = false;
    bool is_jet_central = false;
    bool n_jets_and_bjets = false;

    if (jets.size() >= 3 && n_bjets >= 1)
    {
      double best_mass_diff = 1e6;
      double m_top = 172.5; // massa do top em GeV

      // Loop sobre todas as combinações de 3 jatos onde pelo menos 1 é b-jet
      for (size_t i = 0; i < jets.size(); i++)
      {
        for (size_t j = i + 1; j < jets.size(); j++)
        {
          for (size_t k = j + 1; k < jets.size(); k++)
          {
            // Verificar se pelo menos um dos jatos é b-jet
            if (!is_bjet[i] && !is_bjet[j] && !is_bjet[k])
              continue;

            // Combinar 3 jatos
            fastjet::PseudoJet trijet = jets[i] + jets[j] + jets[k];

            TLorentzVector top_candidate;
            top_candidate.SetPxPyPzE(trijet.px(), trijet.py(), trijet.pz(), trijet.E());

            // Verificar proximidade à massa do top
            double mass_diff = fabs(top_candidate.M() - m_top);

            if (mass_diff < best_mass_diff)
            {
              best_mass_diff = mass_diff;
              best_top_had = top_candidate;
              best_had_indices = {i, j, k};
              found_had_top = true;
            }
          }
        }
      }

      bestjet1_pt = jets[best_had_indices[0]].pt();
      bestjet2_pt = jets[best_had_indices[1]].pt();
      bestbjet_pt = jets[best_had_indices[2]].pt();

      if (jets.size() >= 3 && n_bjets >= 1)
        n_jets_and_bjets = true;

      if ((fabs(jets[best_had_indices[0]].eta()) < 2.5) && (fabs(jets[best_had_indices[1]].eta()) < 2.5) && (fabs(jets[best_had_indices[2]].eta()) < 2.5) && jets[best_had_indices[0]].pt() > 30.0 && jets[best_had_indices[1]].pt() > 30.0 && jets[best_had_indices[2]].pt() > 30.0)
        is_jet_central = true;

      // Preencher histogramas do top hadrônico reconstruído
      if (found_had_top)
      {
        h_reco_top_mass->Fill(best_top_had.M(), event_weight);
        h_reco_top_pt->Fill(best_top_had.Pt(), event_weight);
        h_top_had_y->Fill(best_top_had.Rapidity(), event_weight);
      }
    }

    // Reconstrução de top leptônico: lépton + neutrino + b-jet
    std::vector<int> selected_lepton_pdg;
    std::vector<TLorentzVector> leptons = collectSelectedLeptons(*pdgID, *status, *px, *py, *pz, *e,
                                                                 selected_lepton_pdg);
    h_nlep->Fill(leptons.size(), event_weight);

    // Calcular MET (Missing ET) - soma de todos os neutrinos
    double met_x = 0, met_y = 0;
    for (size_t iPart = 0; iPart < pdgID->size(); iPart++)
    {
      int pdg = pdgID->at(iPart);
      int stat = status->at(iPart);

      // Neutrinos (12, 14, 16)
      if (stat == 1 && (std::abs(pdg) == 12 || std::abs(pdg) == 14 || std::abs(pdg) == 16))
      {
        met_x += px->at(iPart);
        met_y += py->at(iPart);
      }
    }

    smearMET(met_x, met_y, detector_rng);
    double met = std::sqrt(met_x * met_x + met_y * met_y);
    h_met->Fill(met, event_weight);

    if (!leptons.empty())
    {
      lepton = leptons.front();
      const int lead_abs_pdg = selected_lepton_pdg.empty() ? 0 : selected_lepton_pdg.front();
      const double lep_eff = leptonEfficiency(lepton.Pt(), lepton.Eta(), lead_abs_pdg);
      lepton_weight_factor = lep_eff;
      event_weight *= lep_eff;
    }

    // Reconstruir top lept\u00f4nico: l\u00e9pton + neutrino + b-jet
    TLorentzVector best_top_lep;
    TLorentzVector best_W_lep;
    TLorentzVector ttbar_system;
    bool found_lep_top = false;
    bool found_ttbar = false;

    if (leptons.size() >= 1 && n_bjets >= 1)
    {
      double best_chi2 = 1e12;
      const double mW = 80.379;
      const double mTop = 172.5;
      const double sigmaW = 15.0;
      const double sigmaTop = 20.0;
      const std::vector<NeutrinoSolution> nu_solutions =
          reconstructNeutrinoCandidates(lepton, met_x, met_y, 20.0);

      for (size_t iJet = 0; iJet < jets.size(); iJet++)
      {
        if (!is_bjet[iJet])
          continue;

        bool used_in_had = false;
        for (size_t idx : best_had_indices)
        {
          if (iJet == idx)
          {
            used_in_had = true;
            break;
          }
        }
        if (used_in_had)
          continue;

        TLorentzVector bjet;
        bjet.SetPxPyPzE(jets[iJet].px(), jets[iJet].py(), jets[iJet].pz(), jets[iJet].E());

        for (const auto &nu_sol : nu_solutions)
        {
          const TLorentzVector W_lep = lepton + nu_sol.nu;
          const TLorentzVector top_lep_candidate = W_lep + bjet;

          const double chi2_w = std::pow((W_lep.M() - mW) / sigmaW, 2);
          const double chi2_top = std::pow((top_lep_candidate.M() - mTop) / sigmaTop, 2);
          const double chi2 = chi2_w + chi2_top + nu_sol.met_chi2;

          if (chi2 < best_chi2)
          {
            best_chi2 = chi2;
            best_top_lep = top_lep_candidate;
            best_W_lep = W_lep;
            found_lep_top = true;
          }
        }
      }

      if (found_lep_top)
      {
        h_reco_W_lep_mass->Fill(best_W_lep.M(), event_weight);
        h_reco_top_lep_mass->Fill(best_top_lep.M(), event_weight);
        h_reco_top_lep_pt->Fill(best_top_lep.Pt(), event_weight);
        h_top_lep_y->Fill(best_top_lep.Rapidity(), event_weight);

        if (found_had_top)
        {
          ((TH2F *)h_mtop_had_vs_lep)->Fill(best_top_had.M(), best_top_lep.M(), event_weight);
          ttbar_system = best_top_had + best_top_lep;
          found_ttbar = true;
          h_mtt->Fill(ttbar_system.M(), event_weight);
          h_ttbar_y->Fill(ttbar_system.Rapidity(), event_weight);
          h_ttbar_pt->Fill(ttbar_system.Pt(), event_weight);
          if (nPdfMembers > 0)
          {
            const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
            for (int iMember = 0; iMember < nPdfMembers; ++iMember)
            {
              const double memberWeight = weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)] * lepton_weight_factor;
              h_mtt_pdf_members[static_cast<size_t>(iMember)]->Fill(ttbar_system.M(), memberWeight);
              h_ttbar_y_pdf_members[static_cast<size_t>(iMember)]->Fill(ttbar_system.Rapidity(), memberWeight);
              h_ttbar_pt_pdf_members[static_cast<size_t>(iMember)]->Fill(ttbar_system.Pt(), memberWeight);
            }
          }
          h_ttbar_y_vs_pt->Fill(ttbar_system.Rapidity(), ttbar_system.Pt(), event_weight);
          h_delta_y_tt->Fill(best_top_had.Rapidity() - best_top_lep.Rapidity(), event_weight);
        }
      }
    }

    // HT: soma escalar de pT dos jatos
    double HT = 0;
    for (const auto &jet : jets)
      HT += jet.pt();
    h_HT->Fill(HT, event_weight);

    // No cuts
    if (found_had_top)
    {
      h_bestjet1_pt->Fill(jets[best_had_indices[0]].pt(), event_weight);
      h_bestjet2_pt->Fill(jets[best_had_indices[1]].pt(), event_weight);
      h_bestbjet_pt->Fill(jets[best_had_indices[2]].pt(), event_weight);
    }
    if (!leptons.empty())
    {
      h_lep_pt_reco->Fill(lepton.Pt(), event_weight);
      h_lep_eta_reco->Fill(lepton.Eta(), event_weight);
    }

    // cuts flow
    if (leptons.size() != 1 || lepton.Pt() < 18.0)
      continue;
    h_numberofevents_cutflow->Fill(1.0, event_weight);
    cutflow_sumw2[1] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[1][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)] * lepton_weight_factor;
    }

    if (!n_jets_and_bjets)
      continue;
    h_numberofevents_cutflow->Fill(2.0, event_weight);
    cutflow_sumw2[2] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[2][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)] * lepton_weight_factor;
    }

    if (!is_jet_central)
      continue;
    h_numberofevents_cutflow->Fill(3.0, event_weight);
    cutflow_sumw2[3] += event_weight * event_weight;
    if (nPdfMembers > 0)
    {
      const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
      for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        pdf_cutflow_yields[3][static_cast<size_t>(iMember)] += weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)] * lepton_weight_factor;
    }

    // Histogramas nPDF após todos os cortes
    h_HT_aftercuts->Fill(HT, event_weight);
    if (found_had_top)
    {
      h_reco_top_mass_aftercuts->Fill(best_top_had.M(), event_weight);
      h_top_had_y_aftercuts->Fill(best_top_had.Rapidity(), event_weight);
    }
    if (found_lep_top)
    {
      h_reco_top_lep_mass_aftercuts->Fill(best_top_lep.M(), event_weight);
      h_top_lep_y_aftercuts->Fill(best_top_lep.Rapidity(), event_weight);
    }
    if (found_ttbar)
    {
      h_mtt_aftercuts->Fill(ttbar_system.M(), event_weight);
      h_ttbar_y_aftercuts->Fill(ttbar_system.Rapidity(), event_weight);
      h_ttbar_pt_aftercuts->Fill(ttbar_system.Pt(), event_weight);
      if (nPdfMembers > 0)
      {
        const size_t eventOffset = static_cast<size_t>(iEvent) * static_cast<size_t>(nPdfMembers);
        for (int iMember = 0; iMember < nPdfMembers; ++iMember)
        {
          const double memberWeight = weight * pdf_all_event_weights[eventOffset + static_cast<size_t>(iMember)] * lepton_weight_factor;
          h_mtt_aftercuts_pdf_members[static_cast<size_t>(iMember)]->Fill(ttbar_system.M(), memberWeight);
          h_ttbar_y_aftercuts_pdf_members[static_cast<size_t>(iMember)]->Fill(ttbar_system.Rapidity(), memberWeight);
          h_ttbar_pt_aftercuts_pdf_members[static_cast<size_t>(iMember)]->Fill(ttbar_system.Pt(), memberWeight);
        }
      }
      h_ttbar_y_vs_pt_aftercuts->Fill(ttbar_system.Rapidity(), ttbar_system.Pt(), event_weight);
      h_delta_y_tt_aftercuts->Fill(best_top_had.Rapidity() - best_top_lep.Rapidity(), event_weight);
    }
    h_lep_pt_aftercuts->Fill(lepton.Pt(), event_weight);
    h_lep_eta_aftercuts->Fill(lepton.Eta(), event_weight);
    h_bestjet1_pt_aftercuts->Fill(jets[best_had_indices[0]].pt(), event_weight);
    h_bestjet2_pt_aftercuts->Fill(jets[best_had_indices[1]].pt(), event_weight);
    h_bestbjet_pt_aftercuts->Fill(jets[best_had_indices[2]].pt(), event_weight);
    h_bestjet1_eta_aftercuts->Fill(jets[best_had_indices[0]].eta(), event_weight);
    h_bestjet2_eta_aftercuts->Fill(jets[best_had_indices[1]].eta(), event_weight);
    h_bestbjet_eta_aftercuts->Fill(jets[best_had_indices[2]].eta(), event_weight);

    // Progresso
    if (iEvent % 1000 == 0)
    {
      std::cout << "Processado: " << iEvent << " / " << nEntries
                << " (" << (100.0 * iEvent / nEntries) << "%)" << std::endl;
    }
  }

  // Output em .txt com o numero de eventos esperados para uma luminosidade de 8nb^-1 para cada top e anti-top apos a selecao dos leptons, jatos e b-jatos
  TString path = "/media/danbiajujumiguel/T7/HI_analysis/OO_ttbar_analysis/output/";
  FILE *ttbaranalysis;
  ttbaranalysis = fopen(path + txt_name, "a");
  fprintf(ttbaranalysis, "%s\n", "-----------------------------------------------------------");
  fprintf(ttbaranalysis, "%s\n", "-----------ttbar -> Semi-leptonic analysis-----------------");
  fprintf(ttbaranalysis, "%s\n", "-----------------------------------------------------------");
  fprintf(ttbaranalysis, "%s%.3f%s\n", "Cross Section: ", xs, " nb");
  fprintf(ttbaranalysis, "%s%.1f%s\n", "Integrated Luminosity: ", lumi, " nb^-1");
  fprintf(ttbaranalysis, "%s%.2f\n", "Expected events: ", xs * lumi);
  fprintf(ttbaranalysis, "%s\n", "-----------------------------------------------------------");
  fprintf(ttbaranalysis, "%s\n", "----------------Cut Flow-Events----------------------------");
  fprintf(ttbaranalysis, "%s\n", "-----------------------------------------------------------");
  auto printCutflowLine = [&](const char *label, int stageIndex, int binIndex)
  {
    const double nominal = h_numberofevents_cutflow->GetBinContent(binIndex);
    const double reference = h_numberofevents_cutflow->GetBinContent(1);
    const double efficiency = (reference > 0.0) ? 100.0 * nominal / reference : 0.0;
    const double statUnc = std::sqrt(cutflow_sumw2[static_cast<size_t>(stageIndex)]);
    HessianUncertainty pdfUnc;
    if (!pdf_cutflow_yields.empty())
      pdfUnc = computeHessianUncertainty(pdf_cutflow_yields[static_cast<size_t>(stageIndex)]);

    fprintf(ttbaranalysis,
            "%s%.4f +/- %.4f (stat) +%.4f -%.4f (PDF)\t%.3f%%, #N(8 nb-1): %.4f +/- %.4f (stat) +%.4f -%.4f (PDF)\n",
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

  printCutflowLine("Entries(Total): ", 0, 1);
  printCutflowLine("Entries(1 lepton (pT > 18 GeV)): ", 1, 2);
  printCutflowLine("Entries(>= 3 jets, >=1 bjet): ", 2, 3);
  printCutflowLine("Entries(>= 3 jets centrais, >=1 bjet): ", 3, 4);
  fprintf(ttbaranalysis, "%s\n\n", "-----------------------------------------------------------");
  fclose(ttbaranalysis);

  // Salvar histogramas
  TFile *output_file = new TFile(output_name.c_str(), "RECREATE");

  auto mttPdfVars = buildPdfVariationHistograms(h_mtt_pdf_members, "h_mtt_pdf_up", "h_mtt_pdf_down");
  auto yttbarPdfVars = buildPdfVariationHistograms(h_ttbar_y_pdf_members, "h_ttbar_y_pdf_up", "h_ttbar_y_pdf_down");
  auto ptttbarPdfVars = buildPdfVariationHistograms(h_ttbar_pt_pdf_members, "h_ttbar_pt_pdf_up", "h_ttbar_pt_pdf_down");
  auto mttAftercutsPdfVars = buildPdfVariationHistograms(h_mtt_aftercuts_pdf_members, "h_mtt_aftercuts_pdf_up", "h_mtt_aftercuts_pdf_down");
  auto yttbarAftercutsPdfVars = buildPdfVariationHistograms(h_ttbar_y_aftercuts_pdf_members, "h_ttbar_y_aftercuts_pdf_up", "h_ttbar_y_aftercuts_pdf_down");
  auto ptttbarAftercutsPdfVars = buildPdfVariationHistograms(h_ttbar_pt_aftercuts_pdf_members, "h_ttbar_pt_aftercuts_pdf_up", "h_ttbar_pt_aftercuts_pdf_down");

  // Tops reconstruídos com FastJet
  h_njets->Write();
  h_nbjets->Write();
  h_reco_top_mass->Write();
  h_reco_top_pt->Write();

  // Top leptônico
  h_nlep->Write();
  h_met->Write();
  h_reco_top_lep_mass->Write();
  h_reco_top_lep_pt->Write();
  h_reco_W_lep_mass->Write();
  h_mtop_had_vs_lep->Write();
  h_mtt->Write();

  h_numberofevents_cutflow->Write();

  h_bestjet1_pt->Write();
  h_bestjet2_pt->Write();
  h_bestbjet_pt->Write();

  h_lep_pt_reco->Write();
  h_lep_eta_reco->Write();

  h_lep_pt_aftercuts->Write();
  h_lep_eta_aftercuts->Write();
  h_bestjet1_pt_aftercuts->Write();
  h_bestjet2_pt_aftercuts->Write();
  h_bestbjet_pt_aftercuts->Write();
  h_bestjet1_eta_aftercuts->Write();
  h_bestjet2_eta_aftercuts->Write();
  h_bestbjet_eta_aftercuts->Write();

  // nPDF-sensitive histograms
  h_top_had_y->Write();
  h_top_lep_y->Write();
  h_ttbar_y->Write();
  h_ttbar_pt->Write();
  if (mttPdfVars.first)
  {
    mttPdfVars.first->Write();
    mttPdfVars.second->Write();
    yttbarPdfVars.first->Write();
    yttbarPdfVars.second->Write();
    ptttbarPdfVars.first->Write();
    ptttbarPdfVars.second->Write();
  }
  h_ttbar_y_vs_pt->Write();
  h_delta_y_tt->Write();
  h_HT->Write();

  // nPDF after cuts
  h_top_had_y_aftercuts->Write();
  h_top_lep_y_aftercuts->Write();
  h_ttbar_y_aftercuts->Write();
  h_ttbar_pt_aftercuts->Write();
  if (mttAftercutsPdfVars.first)
  {
    mttAftercutsPdfVars.first->Write();
    mttAftercutsPdfVars.second->Write();
    yttbarAftercutsPdfVars.first->Write();
    yttbarAftercutsPdfVars.second->Write();
    ptttbarAftercutsPdfVars.first->Write();
    ptttbarAftercutsPdfVars.second->Write();
  }
  h_ttbar_y_vs_pt_aftercuts->Write();
  h_delta_y_tt_aftercuts->Write();
  h_mtt_aftercuts->Write();
  h_HT_aftercuts->Write();
  h_reco_top_mass_aftercuts->Write();
  h_reco_top_lep_mass_aftercuts->Write();

  output_file->Close();

  std::cout << "\nHistogramas salvos em: " << output_name << std::endl;

  // Fechar arquivo de entrada
  input_file->Close();
  delete input_file;
  delete output_file;

  return 0;
}
