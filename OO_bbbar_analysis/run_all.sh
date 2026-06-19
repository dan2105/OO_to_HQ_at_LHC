#!/bin/sh


# run all the configurations for OO
input_root_dir="/media/danbiajujumiguel/T7/HI_analysis/OO_bbbar_analysis/input_root"
output_dir="/media/danbiajujumiguel/T7/HI_analysis/OO_bbbar_analysis/output"
csv_dir="/media/danbiajujumiguel/T7/HI_analysis/samples_heavyion/OO_bbbar"

#values in 
#23.8 #Xs ISOSPIN in nanobarns

# Xs in nanobarns
# ./analyze_tops \
#   "${input_root_dir}/HI_bbbar_OO_ISOSPIN.root" \
#   "${output_dir}/analysis_HI_bbbar_OO_ISOSPIN.root" \
#   13.8 \
#   analysis_HI_bbbar_OO_ISOSPIN.txt \
#   "${csv_dir}/HI_bbbar_OO_ISOSPIN_pdf_error.hepmc.pdfweights.csv" \
#   0

./analyze_bb \
	"${input_root_dir}/HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16.root" \
	"${output_dir}/analysis_HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16.root" \
	1.938E+07 \
    ${output_dir}/analysis_HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16.txt \
	"${csv_dir}/HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16.hepmc.pdfweights.csv" \
    0

./analyze_bb \
	"${input_root_dir}/HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16_pp_reference.root" \
	"${output_dir}/analysis_HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16_pp_reference.root" \
	2.474E+07 \
    ${output_dir}/analysis_HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16_pp_reference.txt \
	"${csv_dir}/HI_bbbar_OO_EPPS21nlo_CT18Anlo_O16_pp_reference.hepmc.pdfweights.csv" \
    0

#==========================================
#ja atualizado
 ./analyze_bb \
   "${input_root_dir}/HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8.root" \
   "${output_dir}/analysis_HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8.root" \
   2.275E+07 \
   ${output_dir}/analysis_HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8.txt \
   "${csv_dir}/HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8.hepmc.pdfweights.csv" \
    0

./analyze_bb \
   "${input_root_dir}/HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8_pp_reference.root" \
   "${output_dir}/analysis_HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8_pp_reference.root" \
   2.478E+07 \
   ${output_dir}/analysis_HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8_pp_reference.txt \
   "${csv_dir}/HI_bbbar_OO_nCTEQ15HQ_FullNucleous_16_8_pp_reference.hepmc.pdfweights.csv" \
    0

#==========================================
#TUJU21 

 ./analyze_bb \
   "${input_root_dir}/HI_bbbar_OO_TUJU21_16_8.root" \
   "${output_dir}/analysis_HI_bbbar_OO_TUJU21_16_8.root" \
   2.518E+07 \
   ${output_dir}/analysis_HI_bbbar_OO_TUJU21_16_8.txt \
   "${csv_dir}/HI_bbbar_OO_TUJU21_16_8.hepmc.pdfweights.csv" \
    0

 ./analyze_bb \
   "${input_root_dir}/HI_bbbar_OO_TUJU21_16_8_pp_reference.root" \
   "${output_dir}/analysis_HI_bbbar_OO_TUJU21_16_8_pp_reference.root" \
   2.554E+07 \
   ${output_dir}/analysis_HI_bbbar_OO_TUJU21_16_8_pp_reference.txt \
   "${csv_dir}/HI_bbbar_OO_TUJU21_16_8_pp_reference.hepmc.pdfweights.csv" \
    0



