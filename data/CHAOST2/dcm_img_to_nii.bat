@echo off
REM Convert dicom-like images to nii files in 3D
REM This is the first step for image pre-processing

REM Feed path to the downloaded data here
SET DATAPATH=E:\CodeAchieve\Data\CHAOS\Train_Sets\MR 
REM please put chaos dataset training fold here which contains ground truth

REM Feed path to the output folder here
SET OUTPATH=E:\CodeAchieve\Data\CHAOS\Train_Sets\niis

IF NOT EXIST "%OUTPATH%\T2SPIR" (
    mkdir "%OUTPATH%\T2SPIR"
)

FOR /D %%sid IN ("%DATAPATH%\*") DO (
    E:\Environment\MRIcron\Resources\dcm2niix.exe -o "%%sid\T2SPIR" "%%sid\T2SPIR\DICOM_anon"
    FOR %%f IN ("%%sid\T2SPIR\*.nii.gz") DO (
        move "%%f" "%OUTPATH%\T2SPIR\image_%%~nsid.nii.gz"
    )
)