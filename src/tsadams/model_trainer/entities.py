#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Entities in the Time Series Anomaly Archive
#######################################

# 10% subset of entities in the dataset
ANOMALY_ARCHIVE_10_ENTITIES = [
    '128_UCR_Anomaly_GP711MarkerLFM5z2', '149_UCR_Anomaly_Lab2Cmac011215EPG5',
    '107_UCR_Anomaly_NOISEinsectEPG3', '131_UCR_Anomaly_GP711MarkerLFM5z5',
    '071_UCR_Anomaly_DISTORTEDltstdbs30791AS',
    '161_UCR_Anomaly_WalkingAceleration1', '179_UCR_Anomaly_ltstdbs30791AS',
    '041_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG5', '119_UCR_Anomaly_ECG1',
    '061_UCR_Anomaly_DISTORTEDgait3', '001_UCR_Anomaly_DISTORTED1sddb40',
    '042_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG6',
    '054_UCR_Anomaly_DISTORTEDWalkingAceleration5', '193_UCR_Anomaly_s20101m',
    '181_UCR_Anomaly_park3m', '023_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z5',
    '053_UCR_Anomaly_DISTORTEDWalkingAceleration1',
    '011_UCR_Anomaly_DISTORTEDECG1', '237_UCR_Anomaly_mit14157longtermecg',
    '093_UCR_Anomaly_NOISE1sddb40', '070_UCR_Anomaly_DISTORTEDltstdbs30791AI',
    '099_UCR_Anomaly_NOISEInternalBleeding6',
    '028_UCR_Anomaly_DISTORTEDInternalBleeding17',
    '221_UCR_Anomaly_STAFFIIIDatabase', '085_UCR_Anomaly_DISTORTEDs20101m'
]

# Entities in the entire dataset
ANOMALY_ARCHIVE_ENTITIES = [
    '125_UCR_Anomaly_ECG4', '108_UCR_Anomaly_NOISEresperation2',
    '104_UCR_Anomaly_NOISEapneaecg4', '233_UCR_Anomaly_mit14157longtermecg',
    '128_UCR_Anomaly_GP711MarkerLFM5z2',
    '097_UCR_Anomaly_NOISEGP711MarkerLFM5z3',
    '074_UCR_Anomaly_DISTORTEDqtdbSel1005V', '160_UCR_Anomaly_TkeepThirdMARS',
    '077_UCR_Anomaly_DISTORTEDresperation11', '152_UCR_Anomaly_PowerDemand1',
    '185_UCR_Anomaly_resperation11', '191_UCR_Anomaly_resperation9',
    '068_UCR_Anomaly_DISTORTEDinsectEPG4',
    '022_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z4', '250_UCR_Anomaly_weallwalk',
    '096_UCR_Anomaly_NOISEECG4', '135_UCR_Anomaly_InternalBleeding16',
    '241_UCR_Anomaly_taichidbS0715Master', '180_UCR_Anomaly_ltstdbs30791ES',
    '049_UCR_Anomaly_DISTORTEDTkeepFirstMARS',
    '005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1',
    '073_UCR_Anomaly_DISTORTEDpark3m', '235_UCR_Anomaly_mit14157longtermecg',
    '109_UCR_Anomaly_1sddb40', '134_UCR_Anomaly_InternalBleeding15',
    '234_UCR_Anomaly_mit14157longtermecg',
    '113_UCR_Anomaly_CIMIS44AirTemperature1',
    '217_UCR_Anomaly_STAFFIIIDatabase', '148_UCR_Anomaly_Lab2Cmac011215EPG4',
    '126_UCR_Anomaly_ECG4', '194_UCR_Anomaly_sddb49',
    '094_UCR_Anomaly_NOISEBIDMC1', '198_UCR_Anomaly_tiltAPB2',
    '156_UCR_Anomaly_TkeepFifthMARS', '122_UCR_Anomaly_ECG3',
    '006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2',
    '030_UCR_Anomaly_DISTORTEDInternalBleeding19',
    '114_UCR_Anomaly_CIMIS44AirTemperature2',
    '225_UCR_Anomaly_mit14046longtermecg', '195_UCR_Anomaly_sel840mECG1',
    '149_UCR_Anomaly_Lab2Cmac011215EPG5',
    '019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1',
    '014_UCR_Anomaly_DISTORTEDECG3', '213_UCR_Anomaly_STAFFIIIDatabase',
    '107_UCR_Anomaly_NOISEinsectEPG3',
    '029_UCR_Anomaly_DISTORTEDInternalBleeding18',
    '007_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature3', '124_UCR_Anomaly_ECG4',
    '064_UCR_Anomaly_DISTORTEDgaitHunt3', '057_UCR_Anomaly_DISTORTEDapneaecg4',
    '040_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG4',
    '013_UCR_Anomaly_DISTORTEDECG3', '207_UCR_Anomaly_CHARISten',
    '102_UCR_Anomaly_NOISEMesoplodonDensirostris',
    '203_UCR_Anomaly_CHARISfive', '065_UCR_Anomaly_DISTORTEDinsectEPG1',
    '063_UCR_Anomaly_DISTORTEDgaitHunt2',
    '008_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature4',
    '105_UCR_Anomaly_NOISEgait3', '131_UCR_Anomaly_GP711MarkerLFM5z5',
    '208_UCR_Anomaly_CHARISten', '071_UCR_Anomaly_DISTORTEDltstdbs30791AS',
    '154_UCR_Anomaly_PowerDemand3', '047_UCR_Anomaly_DISTORTEDPowerDemand4',
    '137_UCR_Anomaly_InternalBleeding18', '205_UCR_Anomaly_CHARISfive',
    '069_UCR_Anomaly_DISTORTEDinsectEPG5',
    '078_UCR_Anomaly_DISTORTEDresperation1',
    '115_UCR_Anomaly_CIMIS44AirTemperature3', '157_UCR_Anomaly_TkeepFirstMARS',
    '202_UCR_Anomaly_CHARISfive', '153_UCR_Anomaly_PowerDemand2',
    '106_UCR_Anomaly_NOISEgaitHunt2', '243_UCR_Anomaly_tilt12744mtable',
    '158_UCR_Anomaly_TkeepForthMARS', '090_UCR_Anomaly_DISTORTEDtiltAPB2',
    '170_UCR_Anomaly_gaitHunt1', '161_UCR_Anomaly_WalkingAceleration1',
    '174_UCR_Anomaly_insectEPG2', '056_UCR_Anomaly_DISTORTEDapneaecg3',
    '033_UCR_Anomaly_DISTORTEDInternalBleeding5', '167_UCR_Anomaly_gait1',
    '239_UCR_Anomaly_taichidbS0715Master',
    '009_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature5',
    '088_UCR_Anomaly_DISTORTEDsel840mECG2', '175_UCR_Anomaly_insectEPG3',
    '083_UCR_Anomaly_DISTORTEDresperation9', '190_UCR_Anomaly_resperation4',
    '244_UCR_Anomaly_tilt12754table', '196_UCR_Anomaly_sel840mECG2',
    '155_UCR_Anomaly_PowerDemand4', '017_UCR_Anomaly_DISTORTEDECG4',
    '147_UCR_Anomaly_Lab2Cmac011215EPG3', '192_UCR_Anomaly_s20101mML2',
    '034_UCR_Anomaly_DISTORTEDInternalBleeding6',
    '048_UCR_Anomaly_DISTORTEDTkeepFifthMARS',
    '183_UCR_Anomaly_qtdbSel100MLII', '179_UCR_Anomaly_ltstdbs30791AS',
    '228_UCR_Anomaly_mit14134longtermecg',
    '240_UCR_Anomaly_taichidbS0715Master', '246_UCR_Anomaly_tilt12755mtable',
    '012_UCR_Anomaly_DISTORTEDECG2', '197_UCR_Anomaly_tiltAPB1',
    '146_UCR_Anomaly_Lab2Cmac011215EPG2', '211_UCR_Anomaly_Italianpowerdemand',
    '041_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG5',
    '117_UCR_Anomaly_CIMIS44AirTemperature5', '168_UCR_Anomaly_gait2',
    '187_UCR_Anomaly_resperation2',
    '036_UCR_Anomaly_DISTORTEDInternalBleeding9', '119_UCR_Anomaly_ECG1',
    '136_UCR_Anomaly_InternalBleeding17',
    '100_UCR_Anomaly_NOISELab2Cmac011215EPG1',
    '038_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG2',
    '245_UCR_Anomaly_tilt12754table', '188_UCR_Anomaly_resperation2',
    '052_UCR_Anomaly_DISTORTEDTkeepThirdMARS',
    '151_UCR_Anomaly_MesoplodonDensirostris', '059_UCR_Anomaly_DISTORTEDgait1',
    '206_UCR_Anomaly_CHARISten', '061_UCR_Anomaly_DISTORTEDgait3',
    '210_UCR_Anomaly_Italianpowerdemand', '204_UCR_Anomaly_CHARISfive',
    '081_UCR_Anomaly_DISTORTEDresperation3',
    '130_UCR_Anomaly_GP711MarkerLFM5z4', '133_UCR_Anomaly_InternalBleeding14',
    '169_UCR_Anomaly_gait3', '103_UCR_Anomaly_NOISETkeepThirdMARS',
    '079_UCR_Anomaly_DISTORTEDresperation2',
    '129_UCR_Anomaly_GP711MarkerLFM5z3', '141_UCR_Anomaly_InternalBleeding5',
    '001_UCR_Anomaly_DISTORTED1sddb40', '165_UCR_Anomaly_apneaecg4',
    '016_UCR_Anomaly_DISTORTEDECG4', '003_UCR_Anomaly_DISTORTED3sddb40',
    '042_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG6',
    '046_UCR_Anomaly_DISTORTEDPowerDemand3', '186_UCR_Anomaly_resperation1',
    '024_UCR_Anomaly_DISTORTEDInternalBleeding10',
    '159_UCR_Anomaly_TkeepSecondMARS',
    '025_UCR_Anomaly_DISTORTEDInternalBleeding14',
    '227_UCR_Anomaly_mit14134longtermecg', '248_UCR_Anomaly_weallwalk',
    '118_UCR_Anomaly_CIMIS44AirTemperature6', '163_UCR_Anomaly_apneaecg2',
    '084_UCR_Anomaly_DISTORTEDs20101mML2', '086_UCR_Anomaly_DISTORTEDsddb49',
    '242_UCR_Anomaly_tilt12744mtable',
    '072_UCR_Anomaly_DISTORTEDltstdbs30791ES', '184_UCR_Anomaly_resperation10',
    '062_UCR_Anomaly_DISTORTEDgaitHunt1',
    '230_UCR_Anomaly_mit14134longtermecg',
    '229_UCR_Anomaly_mit14134longtermecg', '214_UCR_Anomaly_STAFFIIIDatabase',
    '212_UCR_Anomaly_Italianpowerdemand',
    '021_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z3',
    '116_UCR_Anomaly_CIMIS44AirTemperature4',
    '037_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG1', '111_UCR_Anomaly_3sddb40',
    '087_UCR_Anomaly_DISTORTEDsel840mECG1', '216_UCR_Anomaly_STAFFIIIDatabase',
    '172_UCR_Anomaly_gaitHunt3', '249_UCR_Anomaly_weallwalk',
    '173_UCR_Anomaly_insectEPG1',
    '095_UCR_Anomaly_NOISECIMIS44AirTemperature4',
    '060_UCR_Anomaly_DISTORTEDgait2',
    '035_UCR_Anomaly_DISTORTEDInternalBleeding8',
    '067_UCR_Anomaly_DISTORTEDinsectEPG3',
    '150_UCR_Anomaly_Lab2Cmac011215EPG6',
    '010_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature6',
    '054_UCR_Anomaly_DISTORTEDWalkingAceleration5',
    '182_UCR_Anomaly_qtdbSel1005V', '076_UCR_Anomaly_DISTORTEDresperation10',
    '004_UCR_Anomaly_DISTORTEDBIDMC1', '171_UCR_Anomaly_gaitHunt2',
    '140_UCR_Anomaly_InternalBleeding4', '144_UCR_Anomaly_InternalBleeding9',
    '177_UCR_Anomaly_insectEPG5', '050_UCR_Anomaly_DISTORTEDTkeepForthMARS',
    '132_UCR_Anomaly_InternalBleeding10',
    '082_UCR_Anomaly_DISTORTEDresperation4', '166_UCR_Anomaly_apneaecg',
    '092_UCR_Anomaly_DISTORTEDtiltAPB4',
    '031_UCR_Anomaly_DISTORTEDInternalBleeding20',
    '055_UCR_Anomaly_DISTORTEDapneaecg2',
    '075_UCR_Anomaly_DISTORTEDqtdbSel100MLII',
    '058_UCR_Anomaly_DISTORTEDapneaecg', '209_UCR_Anomaly_Fantasia',
    '080_UCR_Anomaly_DISTORTEDresperation2', '164_UCR_Anomaly_apneaecg3',
    '020_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z2',
    '101_UCR_Anomaly_NOISELab2Cmac011215EPG4',
    '045_UCR_Anomaly_DISTORTEDPowerDemand2',
    '142_UCR_Anomaly_InternalBleeding6', '145_UCR_Anomaly_Lab2Cmac011215EPG1',
    '098_UCR_Anomaly_NOISEInternalBleeding16',
    '162_UCR_Anomaly_WalkingAceleration5', '193_UCR_Anomaly_s20101m',
    '043_UCR_Anomaly_DISTORTEDMesoplodonDensirostris',
    '247_UCR_Anomaly_tilt12755mtable', '139_UCR_Anomaly_InternalBleeding20',
    '181_UCR_Anomaly_park3m', '110_UCR_Anomaly_2sddb40',
    '218_UCR_Anomaly_STAFFIIIDatabase', '127_UCR_Anomaly_GP711MarkerLFM5z1',
    '023_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z5',
    '178_UCR_Anomaly_ltstdbs30791AI', '176_UCR_Anomaly_insectEPG4',
    '189_UCR_Anomaly_resperation3', '238_UCR_Anomaly_mit14157longtermecg',
    '032_UCR_Anomaly_DISTORTEDInternalBleeding4',
    '018_UCR_Anomaly_DISTORTEDECG4', '223_UCR_Anomaly_mit14046longtermecg',
    '232_UCR_Anomaly_mit14134longtermecg',
    '138_UCR_Anomaly_InternalBleeding19', '215_UCR_Anomaly_STAFFIIIDatabase',
    '236_UCR_Anomaly_mit14157longtermecg', '089_UCR_Anomaly_DISTORTEDtiltAPB1',
    '220_UCR_Anomaly_STAFFIIIDatabase', '112_UCR_Anomaly_BIDMC1',
    '027_UCR_Anomaly_DISTORTEDInternalBleeding16',
    '091_UCR_Anomaly_DISTORTEDtiltAPB3', '219_UCR_Anomaly_STAFFIIIDatabase',
    '015_UCR_Anomaly_DISTORTEDECG4',
    '053_UCR_Anomaly_DISTORTEDWalkingAceleration1',
    '143_UCR_Anomaly_InternalBleeding8', '011_UCR_Anomaly_DISTORTEDECG1',
    '039_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG3',
    '224_UCR_Anomaly_mit14046longtermecg', '120_UCR_Anomaly_ECG2',
    '044_UCR_Anomaly_DISTORTEDPowerDemand1',
    '051_UCR_Anomaly_DISTORTEDTkeepSecondMARS',
    '237_UCR_Anomaly_mit14157longtermecg', '002_UCR_Anomaly_DISTORTED2sddb40',
    '200_UCR_Anomaly_tiltAPB4', '093_UCR_Anomaly_NOISE1sddb40',
    '070_UCR_Anomaly_DISTORTEDltstdbs30791AI',
    '099_UCR_Anomaly_NOISEInternalBleeding6',
    '026_UCR_Anomaly_DISTORTEDInternalBleeding15',
    '028_UCR_Anomaly_DISTORTEDInternalBleeding17',
    '231_UCR_Anomaly_mit14134longtermecg', '199_UCR_Anomaly_tiltAPB3',
    '221_UCR_Anomaly_STAFFIIIDatabase', '123_UCR_Anomaly_ECG4',
    '085_UCR_Anomaly_DISTORTEDs20101m', '222_UCR_Anomaly_mit14046longtermecg',
    '121_UCR_Anomaly_ECG3', '066_UCR_Anomaly_DISTORTEDinsectEPG2',
    '226_UCR_Anomaly_mit14046longtermecg', '201_UCR_Anomaly_CHARISfive'
]

#######################################
# Entities in the Server Machine Dataset
#######################################
MACHINES = [
    'machine-2-8', 'machine-1-8', 'machine-3-4', 'machine-1-7', 'machine-2-2',
    'machine-2-4', 'machine-3-5', 'machine-1-4', 'machine-3-8', 'machine-1-2',
    'machine-2-6', 'machine-1-6', 'machine-3-10', 'machine-1-5',
    'machine-3-11', 'machine-3-7', 'machine-2-9', 'machine-2-5', 'machine-3-1',
    'machine-3-9', 'machine-3-6', 'machine-3-2', 'machine-2-3', 'machine-2-7',
    'machine-3-3', 'machine-1-1', 'machine-1-3', 'machine-2-1'
]

#######################################
# Entities in the NASA MSL Dataset
#######################################
MSL_CHANNELS = [
    'M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4',
    'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14', 'T-9',
    'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7', 'F-8'
]

#######################################
# Entities in the NASA SMAP Dataset
#######################################
SMAP_CHANNELS = [
    'P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8',
    'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-2', 'P-3', 'D-2',
    'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2', 'D-5', 'D-6', 'D-7',
    'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9', 'F-2', 'G-4', 'T-3',
    'D-11', 'D-12', 'B-1', 'G-6', 'G-7', 'P-7', 'R-1', 'A-5', 'A-6', 'A-7',
    'D-13', 'A-8', 'A-9', 'F-3'
]


#######################################
# Entities in the AutoTSAD Dataset
#######################################
AUTOTSAD_ENTITIES = [
    "GutenTAG=cbf-position-middle.semi-supervised", "GutenTAG=ecg-diff-count-1.semi-supervised", "GutenTAG=ecg-diff-count-8.semi-supervised",
    "GutenTAG=ecg-same-count-1.semi-supervised", "GutenTAG=ecg-same-count-2.semi-supervised", "GutenTAG=ecg-type-mean.semi-supervised",
    "GutenTAG=ecg-type-pattern-shift.semi-supervised", "GutenTAG=ecg-type-trend.semi-supervised", "GutenTAG=poly-diff-count-2.semi-supervised",
    "GutenTAG=poly-diff-count-5.semi-supervised", "GutenTAG=poly-length-10.semi-supervised", "GutenTAG=poly-length-500.semi-supervised",
    "GutenTAG=poly-trend-linear.semi-supervised", "GutenTAG=rw-combined-diff-1.semi-supervised", "GutenTAG=rw-combined-diff-2.semi-supervised",
    "GutenTAG=rw-length-100.semi-supervised", "GutenTAG=sinus-diff-count-2.semi-supervised", "GutenTAG=sinus-length-50.semi-supervised",
    "GutenTAG=sinus-position-middle.semi-supervised", "GutenTAG=sinus-type-mean.semi-supervised", "KDD-TSAD=011_UCR_Anomaly_DISTORTEDECG1",
    "KDD-TSAD=022_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z4", "KDD-TSAD=030_UCR_Anomaly_DISTORTEDInternalBleeding19",
    "KDD-TSAD=031_UCR_Anomaly_DISTORTEDInternalBleeding20", "KDD-TSAD=058_UCR_Anomaly_DISTORTEDapneaecg",
    "KDD-TSAD=070_UCR_Anomaly_DISTORTEDltstdbs30791AI", "KDD-TSAD=074_UCR_Anomaly_DISTORTEDqtdbSel1005V",
    "KDD-TSAD=095_UCR_Anomaly_NOISECIMIS44AirTemperature4", "KDD-TSAD=098_UCR_Anomaly_NOISEInternalBleeding16",
    "KDD-TSAD=102_UCR_Anomaly_NOISEMesoplodonDensirostris", "KDD-TSAD=114_UCR_Anomaly_CIMIS44AirTemperature2",
    "KDD-TSAD=131_UCR_Anomaly_GP711MarkerLFM5z5", "KDD-TSAD=147_UCR_Anomaly_Lab2Cmac011215EPG3", "KDD-TSAD=152_UCR_Anomaly_PowerDemand1",
    "KDD-TSAD=154_UCR_Anomaly_PowerDemand3", "KDD-TSAD=163_UCR_Anomaly_apneaecg2", "KDD-TSAD=174_UCR_Anomaly_insectEPG2",
    "KDD-TSAD=202_UCR_Anomaly_CHARISfive", "KDD-TSAD=208_UCR_Anomaly_CHARISten", "KDD-TSAD=232_UCR_Anomaly_mit14134longtermecg",
    "NASA-MSL=C-2", "NASA-MSL=D-14", "NASA-MSL=F-5", "NASA-MSL=F-7", "NASA-MSL=M-6", "NASA-MSL=M-7", "NASA-MSL=P-10",
    "NASA-MSL=P-11", "NASA-MSL=P-14", "NASA-MSL=P-15", "NASA-MSL=S-2", "NASA-MSL=T-12", "NASA-MSL=T-4", "NASA-MSL=T-5",
    "NASA-MSL=T-8", "NASA-SMAP=A-1", "NASA-SMAP=A-2", "NASA-SMAP=A-3", "NASA-SMAP=A-5", "NASA-SMAP=B-1", "NASA-SMAP=D-13",
    "NASA-SMAP=D-5", "NASA-SMAP=D-8", "NASA-SMAP=E-1", "NASA-SMAP=E-5", "NASA-SMAP=E-6", "NASA-SMAP=E-7", "NASA-SMAP=G-1",
    "NASA-SMAP=G-2", "NASA-SMAP=G-3", "NASA-SMAP=G-4", "NASA-SMAP=G-6", "NASA-SMAP=G-7", "NASA-SMAP=P-4", "NASA-SMAP=S-1",
    "synthetic=gt-0", "synthetic=gt-2", "synthetic=gt-3", "synthetic=gt-4",
]

#######################################
# Family of entities in the Anomaly Archive
#######################################

ANOMALY_ARCHIVE_ENTITY_TO_DATA_FAMILY = {
    '125_UCR_Anomaly_ECG4': 'Electrocardiogram (ECG) Arrhythmia',
    '108_UCR_Anomaly_NOISEresperation2': 'Respiration Rate (RESP)',
    '104_UCR_Anomaly_NOISEapneaecg4': 'Electrocardiogram (ECG) Arrhythmia',
    '233_UCR_Anomaly_mit14157longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '128_UCR_Anomaly_GP711MarkerLFM5z2': 'Gait',
    '097_UCR_Anomaly_NOISEGP711MarkerLFM5z3': 'Gait',
    '074_UCR_Anomaly_DISTORTEDqtdbSel1005V':
    'Electrocardiogram (ECG) Arrhythmia',
    '160_UCR_Anomaly_TkeepThirdMARS': 'NASA Data',
    '077_UCR_Anomaly_DISTORTEDresperation11': 'Respiration Rate (RESP)',
    '152_UCR_Anomaly_PowerDemand1': 'Power Demand',
    '185_UCR_Anomaly_resperation11': 'Respiration Rate (RESP)',
    '191_UCR_Anomaly_resperation9': 'Respiration Rate (RESP)',
    '068_UCR_Anomaly_DISTORTEDinsectEPG4':
    'Insect Electrical Penetration Graph (EPG)',
    '022_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z4': 'Gait',
    '250_UCR_Anomaly_weallwalk': 'Gait',
    '096_UCR_Anomaly_NOISEECG4': 'Electrocardiogram (ECG) Arrhythmia',
    '135_UCR_Anomaly_InternalBleeding16': 'Atrial Blood Pressure (ABP)',
    '241_UCR_Anomaly_taichidbS0715Master': 'Gait',
    '180_UCR_Anomaly_ltstdbs30791ES': 'Electrocardiogram (ECG) Arrhythmia',
    '049_UCR_Anomaly_DISTORTEDTkeepFirstMARS': 'NASA Data',
    '005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1': 'Air Temperature',
    '073_UCR_Anomaly_DISTORTEDpark3m': 'Gait',
    '235_UCR_Anomaly_mit14157longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '109_UCR_Anomaly_1sddb40': 'Electrocardiogram (ECG) Arrhythmia',
    '134_UCR_Anomaly_InternalBleeding15': 'Atrial Blood Pressure (ABP)',
    '234_UCR_Anomaly_mit14157longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '113_UCR_Anomaly_CIMIS44AirTemperature1': 'Air Temperature',
    '217_UCR_Anomaly_STAFFIIIDatabase': 'Electrocardiogram (ECG) Arrhythmia',
    '148_UCR_Anomaly_Lab2Cmac011215EPG4':
    'Insect Electrical Penetration Graph (EPG)',
    '126_UCR_Anomaly_ECG4': 'Electrocardiogram (ECG) Arrhythmia',
    '194_UCR_Anomaly_sddb49': 'Electrocardiogram (ECG) Arrhythmia',
    '094_UCR_Anomaly_NOISEBIDMC1': 'Electrocardiogram (ECG) Arrhythmia',
    '198_UCR_Anomaly_tiltAPB2': 'Atrial Blood Pressure (ABP)',
    '156_UCR_Anomaly_TkeepFifthMARS': 'NASA Data',
    '122_UCR_Anomaly_ECG3': 'Electrocardiogram (ECG) Arrhythmia',
    '006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2': 'Air Temperature',
    '030_UCR_Anomaly_DISTORTEDInternalBleeding19':
    'Atrial Blood Pressure (ABP)',
    '114_UCR_Anomaly_CIMIS44AirTemperature2': 'Air Temperature',
    '225_UCR_Anomaly_mit14046longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '195_UCR_Anomaly_sel840mECG1': 'Electrocardiogram (ECG) Arrhythmia',
    '149_UCR_Anomaly_Lab2Cmac011215EPG5':
    'Insect Electrical Penetration Graph (EPG)',
    '019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1': 'Gait',
    '014_UCR_Anomaly_DISTORTEDECG3': 'Electrocardiogram (ECG) Arrhythmia',
    '213_UCR_Anomaly_STAFFIIIDatabase': 'Electrocardiogram (ECG) Arrhythmia',
    '107_UCR_Anomaly_NOISEinsectEPG3':
    'Insect Electrical Penetration Graph (EPG)',
    '029_UCR_Anomaly_DISTORTEDInternalBleeding18':
    'Atrial Blood Pressure (ABP)',
    '007_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature3': 'Air Temperature',
    '124_UCR_Anomaly_ECG4': 'Electrocardiogram (ECG) Arrhythmia',
    '064_UCR_Anomaly_DISTORTEDgaitHunt3': 'Gait',
    '057_UCR_Anomaly_DISTORTEDapneaecg4': 'Electrocardiogram (ECG) Arrhythmia',
    '040_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG4':
    'Insect Electrical Penetration Graph (EPG)',
    '013_UCR_Anomaly_DISTORTEDECG3': 'Electrocardiogram (ECG) Arrhythmia',
    '207_UCR_Anomaly_CHARISten': 'Electrocardiogram (ECG) Arrhythmia',
    '102_UCR_Anomaly_NOISEMesoplodonDensirostris': 'Acceleration Sensor Data',
    '203_UCR_Anomaly_CHARISfive': 'Electrocardiogram (ECG) Arrhythmia',
    '065_UCR_Anomaly_DISTORTEDinsectEPG1':
    'Insect Electrical Penetration Graph (EPG)',
    '063_UCR_Anomaly_DISTORTEDgaitHunt2': 'Gait',
    '008_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature4': 'Air Temperature',
    '105_UCR_Anomaly_NOISEgait3': 'Gait',
    '131_UCR_Anomaly_GP711MarkerLFM5z5': 'Gait',
    '208_UCR_Anomaly_CHARISten': 'Electrocardiogram (ECG) Arrhythmia',
    '071_UCR_Anomaly_DISTORTEDltstdbs30791AS':
    'Electrocardiogram (ECG) Arrhythmia',
    '154_UCR_Anomaly_PowerDemand3': 'Power Demand',
    '047_UCR_Anomaly_DISTORTEDPowerDemand4': 'Power Demand',
    '137_UCR_Anomaly_InternalBleeding18': 'Atrial Blood Pressure (ABP)',
    '205_UCR_Anomaly_CHARISfive': 'Electrocardiogram (ECG) Arrhythmia',
    '069_UCR_Anomaly_DISTORTEDinsectEPG5':
    'Insect Electrical Penetration Graph (EPG)',
    '078_UCR_Anomaly_DISTORTEDresperation1': 'Respiration Rate (RESP)',
    '115_UCR_Anomaly_CIMIS44AirTemperature3': 'Air Temperature',
    '157_UCR_Anomaly_TkeepFirstMARS': 'NASA Data',
    '202_UCR_Anomaly_CHARISfive': 'Electrocardiogram (ECG) Arrhythmia',
    '153_UCR_Anomaly_PowerDemand2': 'Power Demand',
    '106_UCR_Anomaly_NOISEgaitHunt2': 'Gait',
    '243_UCR_Anomaly_tilt12744mtable': 'Atrial Blood Pressure (ABP)',
    '158_UCR_Anomaly_TkeepForthMARS': 'NASA Data',
    '090_UCR_Anomaly_DISTORTEDtiltAPB2': 'Atrial Blood Pressure (ABP)',
    '170_UCR_Anomaly_gaitHunt1': 'Gait',
    '161_UCR_Anomaly_WalkingAceleration1': 'Acceleration Sensor Data',
    '174_UCR_Anomaly_insectEPG2': 'Insect Electrical Penetration Graph (EPG)',
    '056_UCR_Anomaly_DISTORTEDapneaecg3': 'Electrocardiogram (ECG) Arrhythmia',
    '033_UCR_Anomaly_DISTORTEDInternalBleeding5':
    'Atrial Blood Pressure (ABP)',
    '167_UCR_Anomaly_gait1': 'Gait',
    '239_UCR_Anomaly_taichidbS0715Master': 'Gait',
    '009_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature5': 'Air Temperature',
    '088_UCR_Anomaly_DISTORTEDsel840mECG2':
    'Electrocardiogram (ECG) Arrhythmia',
    '175_UCR_Anomaly_insectEPG3': 'Insect Electrical Penetration Graph (EPG)',
    '083_UCR_Anomaly_DISTORTEDresperation9': 'Respiration Rate (RESP)',
    '190_UCR_Anomaly_resperation4': 'Respiration Rate (RESP)',
    '244_UCR_Anomaly_tilt12754table': 'Atrial Blood Pressure (ABP)',
    '196_UCR_Anomaly_sel840mECG2': 'Electrocardiogram (ECG) Arrhythmia',
    '155_UCR_Anomaly_PowerDemand4': 'Power Demand',
    '017_UCR_Anomaly_DISTORTEDECG4': 'Electrocardiogram (ECG) Arrhythmia',
    '147_UCR_Anomaly_Lab2Cmac011215EPG3':
    'Insect Electrical Penetration Graph (EPG)',
    '192_UCR_Anomaly_s20101mML2': 'Electrocardiogram (ECG) Arrhythmia',
    '034_UCR_Anomaly_DISTORTEDInternalBleeding6':
    'Atrial Blood Pressure (ABP)',
    '048_UCR_Anomaly_DISTORTEDTkeepFifthMARS': 'NASA Data',
    '183_UCR_Anomaly_qtdbSel100MLII': 'Electrocardiogram (ECG) Arrhythmia',
    '179_UCR_Anomaly_ltstdbs30791AS': 'Electrocardiogram (ECG) Arrhythmia',
    '228_UCR_Anomaly_mit14134longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '240_UCR_Anomaly_taichidbS0715Master': 'Gait',
    '246_UCR_Anomaly_tilt12755mtable': 'Atrial Blood Pressure (ABP)',
    '012_UCR_Anomaly_DISTORTEDECG2': 'Electrocardiogram (ECG) Arrhythmia',
    '197_UCR_Anomaly_tiltAPB1': 'Atrial Blood Pressure (ABP)',
    '146_UCR_Anomaly_Lab2Cmac011215EPG2':
    'Insect Electrical Penetration Graph (EPG)',
    '211_UCR_Anomaly_Italianpowerdemand': 'Power Demand',
    '041_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG5':
    'Insect Electrical Penetration Graph (EPG)',
    '117_UCR_Anomaly_CIMIS44AirTemperature5': 'Air Temperature',
    '168_UCR_Anomaly_gait2': 'Gait',
    '187_UCR_Anomaly_resperation2': 'Respiration Rate (RESP)',
    '036_UCR_Anomaly_DISTORTEDInternalBleeding9':
    'Atrial Blood Pressure (ABP)',
    '119_UCR_Anomaly_ECG1': 'Electrocardiogram (ECG) Arrhythmia',
    '136_UCR_Anomaly_InternalBleeding17': 'Atrial Blood Pressure (ABP)',
    '100_UCR_Anomaly_NOISELab2Cmac011215EPG1':
    'Insect Electrical Penetration Graph (EPG)',
    '038_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG2':
    'Insect Electrical Penetration Graph (EPG)',
    '245_UCR_Anomaly_tilt12754table': 'Atrial Blood Pressure (ABP)',
    '188_UCR_Anomaly_resperation2': 'Respiration Rate (RESP)',
    '052_UCR_Anomaly_DISTORTEDTkeepThirdMARS': 'NASA Data',
    '151_UCR_Anomaly_MesoplodonDensirostris': 'Acceleration Sensor Data',
    '059_UCR_Anomaly_DISTORTEDgait1': 'Gait',
    '206_UCR_Anomaly_CHARISten': 'Electrocardiogram (ECG) Arrhythmia',
    '061_UCR_Anomaly_DISTORTEDgait3': 'Gait',
    '210_UCR_Anomaly_Italianpowerdemand': 'Power Demand',
    '204_UCR_Anomaly_CHARISfive': 'Electrocardiogram (ECG) Arrhythmia',
    '081_UCR_Anomaly_DISTORTEDresperation3': 'Respiration Rate (RESP)',
    '130_UCR_Anomaly_GP711MarkerLFM5z4': 'Gait',
    '133_UCR_Anomaly_InternalBleeding14': 'Atrial Blood Pressure (ABP)',
    '169_UCR_Anomaly_gait3': 'Gait',
    '103_UCR_Anomaly_NOISETkeepThirdMARS': 'NASA Data',
    '079_UCR_Anomaly_DISTORTEDresperation2': 'Respiration Rate (RESP)',
    '129_UCR_Anomaly_GP711MarkerLFM5z3': 'Gait',
    '141_UCR_Anomaly_InternalBleeding5': 'Atrial Blood Pressure (ABP)',
    '001_UCR_Anomaly_DISTORTED1sddb40': 'Electrocardiogram (ECG) Arrhythmia',
    '165_UCR_Anomaly_apneaecg4': 'Electrocardiogram (ECG) Arrhythmia',
    '016_UCR_Anomaly_DISTORTEDECG4': 'Electrocardiogram (ECG) Arrhythmia',
    '003_UCR_Anomaly_DISTORTED3sddb40': 'Electrocardiogram (ECG) Arrhythmia',
    '042_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG6':
    'Insect Electrical Penetration Graph (EPG)',
    '046_UCR_Anomaly_DISTORTEDPowerDemand3': 'Power Demand',
    '186_UCR_Anomaly_resperation1': 'Respiration Rate (RESP)',
    '024_UCR_Anomaly_DISTORTEDInternalBleeding10':
    'Atrial Blood Pressure (ABP)',
    '159_UCR_Anomaly_TkeepSecondMARS': 'NASA Data',
    '025_UCR_Anomaly_DISTORTEDInternalBleeding14':
    'Atrial Blood Pressure (ABP)',
    '227_UCR_Anomaly_mit14134longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '248_UCR_Anomaly_weallwalk': 'Gait',
    '118_UCR_Anomaly_CIMIS44AirTemperature6': 'Air Temperature',
    '163_UCR_Anomaly_apneaecg2': 'Electrocardiogram (ECG) Arrhythmia',
    '084_UCR_Anomaly_DISTORTEDs20101mML2':
    'Electrocardiogram (ECG) Arrhythmia',
    '086_UCR_Anomaly_DISTORTEDsddb49': 'Electrocardiogram (ECG) Arrhythmia',
    '242_UCR_Anomaly_tilt12744mtable': 'Atrial Blood Pressure (ABP)',
    '072_UCR_Anomaly_DISTORTEDltstdbs30791ES':
    'Electrocardiogram (ECG) Arrhythmia',
    '184_UCR_Anomaly_resperation10': 'Respiration Rate (RESP)',
    '062_UCR_Anomaly_DISTORTEDgaitHunt1': 'Gait',
    '230_UCR_Anomaly_mit14134longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '229_UCR_Anomaly_mit14134longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '214_UCR_Anomaly_STAFFIIIDatabase': 'Electrocardiogram (ECG) Arrhythmia',
    '212_UCR_Anomaly_Italianpowerdemand': 'Power Demand',
    '021_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z3': 'Gait',
    '116_UCR_Anomaly_CIMIS44AirTemperature4': 'Air Temperature',
    '037_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG1':
    'Insect Electrical Penetration Graph (EPG)',
    '111_UCR_Anomaly_3sddb40': 'Electrocardiogram (ECG) Arrhythmia',
    '087_UCR_Anomaly_DISTORTEDsel840mECG1':
    'Electrocardiogram (ECG) Arrhythmia',
    '216_UCR_Anomaly_STAFFIIIDatabase': 'Electrocardiogram (ECG) Arrhythmia',
    '172_UCR_Anomaly_gaitHunt3': 'Gait',
    '249_UCR_Anomaly_weallwalk': 'Gait',
    '173_UCR_Anomaly_insectEPG1': 'Insect Electrical Penetration Graph (EPG)',
    '095_UCR_Anomaly_NOISECIMIS44AirTemperature4': 'Air Temperature',
    '060_UCR_Anomaly_DISTORTEDgait2': 'Gait',
    '035_UCR_Anomaly_DISTORTEDInternalBleeding8':
    'Atrial Blood Pressure (ABP)',
    '067_UCR_Anomaly_DISTORTEDinsectEPG3':
    'Insect Electrical Penetration Graph (EPG)',
    '150_UCR_Anomaly_Lab2Cmac011215EPG6':
    'Insect Electrical Penetration Graph (EPG)',
    '010_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature6': 'Air Temperature',
    '054_UCR_Anomaly_DISTORTEDWalkingAceleration5': 'Acceleration Sensor Data',
    '182_UCR_Anomaly_qtdbSel1005V': 'Electrocardiogram (ECG) Arrhythmia',
    '076_UCR_Anomaly_DISTORTEDresperation10': 'Respiration Rate (RESP)',
    '004_UCR_Anomaly_DISTORTEDBIDMC1': 'Electrocardiogram (ECG) Arrhythmia',
    '171_UCR_Anomaly_gaitHunt2': 'Gait',
    '140_UCR_Anomaly_InternalBleeding4': 'Atrial Blood Pressure (ABP)',
    '144_UCR_Anomaly_InternalBleeding9': 'Atrial Blood Pressure (ABP)',
    '177_UCR_Anomaly_insectEPG5': 'Insect Electrical Penetration Graph (EPG)',
    '050_UCR_Anomaly_DISTORTEDTkeepForthMARS': 'NASA Data',
    '132_UCR_Anomaly_InternalBleeding10': 'Atrial Blood Pressure (ABP)',
    '082_UCR_Anomaly_DISTORTEDresperation4': 'Respiration Rate (RESP)',
    '166_UCR_Anomaly_apneaecg': 'Electrocardiogram (ECG) Arrhythmia',
    '092_UCR_Anomaly_DISTORTEDtiltAPB4': 'Atrial Blood Pressure (ABP)',
    '031_UCR_Anomaly_DISTORTEDInternalBleeding20':
    'Atrial Blood Pressure (ABP)',
    '055_UCR_Anomaly_DISTORTEDapneaecg2': 'Electrocardiogram (ECG) Arrhythmia',
    '075_UCR_Anomaly_DISTORTEDqtdbSel100MLII':
    'Electrocardiogram (ECG) Arrhythmia',
    '058_UCR_Anomaly_DISTORTEDapneaecg': 'Electrocardiogram (ECG) Arrhythmia',
    '209_UCR_Anomaly_Fantasia': 'Electrocardiogram (ECG) Arrhythmia',
    '080_UCR_Anomaly_DISTORTEDresperation2': 'Respiration Rate (RESP)',
    '164_UCR_Anomaly_apneaecg3': 'Electrocardiogram (ECG) Arrhythmia',
    '020_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z2': 'Gait',
    '101_UCR_Anomaly_NOISELab2Cmac011215EPG4':
    'Insect Electrical Penetration Graph (EPG)',
    '045_UCR_Anomaly_DISTORTEDPowerDemand2': 'Power Demand',
    '142_UCR_Anomaly_InternalBleeding6': 'Atrial Blood Pressure (ABP)',
    '145_UCR_Anomaly_Lab2Cmac011215EPG1':
    'Insect Electrical Penetration Graph (EPG)',
    '098_UCR_Anomaly_NOISEInternalBleeding16': 'Atrial Blood Pressure (ABP)',
    '162_UCR_Anomaly_WalkingAceleration5': 'Acceleration Sensor Data',
    '193_UCR_Anomaly_s20101m': 'Electrocardiogram (ECG) Arrhythmia',
    '043_UCR_Anomaly_DISTORTEDMesoplodonDensirostris':
    'Acceleration Sensor Data',
    '247_UCR_Anomaly_tilt12755mtable': 'Atrial Blood Pressure (ABP)',
    '139_UCR_Anomaly_InternalBleeding20': 'Atrial Blood Pressure (ABP)',
    '181_UCR_Anomaly_park3m': 'Gait',
    '110_UCR_Anomaly_2sddb40': 'Electrocardiogram (ECG) Arrhythmia',
    '218_UCR_Anomaly_STAFFIIIDatabase': 'Electrocardiogram (ECG) Arrhythmia',
    '127_UCR_Anomaly_GP711MarkerLFM5z1': 'Gait',
    '023_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z5': 'Gait',
    '178_UCR_Anomaly_ltstdbs30791AI': 'Electrocardiogram (ECG) Arrhythmia',
    '176_UCR_Anomaly_insectEPG4': 'Insect Electrical Penetration Graph (EPG)',
    '189_UCR_Anomaly_resperation3': 'Respiration Rate (RESP)',
    '238_UCR_Anomaly_mit14157longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '032_UCR_Anomaly_DISTORTEDInternalBleeding4':
    'Atrial Blood Pressure (ABP)',
    '018_UCR_Anomaly_DISTORTEDECG4': 'Electrocardiogram (ECG) Arrhythmia',
    '223_UCR_Anomaly_mit14046longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '232_UCR_Anomaly_mit14134longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '138_UCR_Anomaly_InternalBleeding19': 'Atrial Blood Pressure (ABP)',
    '215_UCR_Anomaly_STAFFIIIDatabase': 'Electrocardiogram (ECG) Arrhythmia',
    '236_UCR_Anomaly_mit14157longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '089_UCR_Anomaly_DISTORTEDtiltAPB1': 'Atrial Blood Pressure (ABP)',
    '220_UCR_Anomaly_STAFFIIIDatabase': 'Electrocardiogram (ECG) Arrhythmia',
    '112_UCR_Anomaly_BIDMC1': 'Electrocardiogram (ECG) Arrhythmia',
    '027_UCR_Anomaly_DISTORTEDInternalBleeding16':
    'Atrial Blood Pressure (ABP)',
    '091_UCR_Anomaly_DISTORTEDtiltAPB3': 'Atrial Blood Pressure (ABP)',
    '219_UCR_Anomaly_STAFFIIIDatabase': 'Electrocardiogram (ECG) Arrhythmia',
    '015_UCR_Anomaly_DISTORTEDECG4': 'Electrocardiogram (ECG) Arrhythmia',
    '053_UCR_Anomaly_DISTORTEDWalkingAceleration1': 'Acceleration Sensor Data',
    '143_UCR_Anomaly_InternalBleeding8': 'Atrial Blood Pressure (ABP)',
    '011_UCR_Anomaly_DISTORTEDECG1': 'Electrocardiogram (ECG) Arrhythmia',
    '039_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG3':
    'Insect Electrical Penetration Graph (EPG)',
    '224_UCR_Anomaly_mit14046longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '120_UCR_Anomaly_ECG2': 'Electrocardiogram (ECG) Arrhythmia',
    '044_UCR_Anomaly_DISTORTEDPowerDemand1': 'Power Demand',
    '051_UCR_Anomaly_DISTORTEDTkeepSecondMARS': 'NASA Data',
    '237_UCR_Anomaly_mit14157longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '002_UCR_Anomaly_DISTORTED2sddb40': 'Electrocardiogram (ECG) Arrhythmia',
    '200_UCR_Anomaly_tiltAPB4': 'Atrial Blood Pressure (ABP)',
    '093_UCR_Anomaly_NOISE1sddb40': 'Electrocardiogram (ECG) Arrhythmia',
    '070_UCR_Anomaly_DISTORTEDltstdbs30791AI':
    'Electrocardiogram (ECG) Arrhythmia',
    '099_UCR_Anomaly_NOISEInternalBleeding6': 'Atrial Blood Pressure (ABP)',
    '026_UCR_Anomaly_DISTORTEDInternalBleeding15':
    'Atrial Blood Pressure (ABP)',
    '028_UCR_Anomaly_DISTORTEDInternalBleeding17':
    'Atrial Blood Pressure (ABP)',
    '231_UCR_Anomaly_mit14134longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '199_UCR_Anomaly_tiltAPB3': 'Atrial Blood Pressure (ABP)',
    '221_UCR_Anomaly_STAFFIIIDatabase': 'Electrocardiogram (ECG) Arrhythmia',
    '123_UCR_Anomaly_ECG4': 'Electrocardiogram (ECG) Arrhythmia',
    '085_UCR_Anomaly_DISTORTEDs20101m': 'Electrocardiogram (ECG) Arrhythmia',
    '222_UCR_Anomaly_mit14046longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '121_UCR_Anomaly_ECG3': 'Electrocardiogram (ECG) Arrhythmia',
    '066_UCR_Anomaly_DISTORTEDinsectEPG2':
    'Insect Electrical Penetration Graph (EPG)',
    '226_UCR_Anomaly_mit14046longtermecg':
    'Electrocardiogram (ECG) Arrhythmia',
    '201_UCR_Anomaly_CHARISfive': 'Electrocardiogram (ECG) Arrhythmia'
}
