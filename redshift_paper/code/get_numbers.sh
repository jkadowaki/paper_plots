#!/bin/bash

################################################################################

FILE=../data/kadowaki2019.tsv
DATA=../data/get_number.csv
tmp=tmp.txt
    
COLUMNS=TABLE,NAME,udg,ra,dec,cz,sepDEG,LocalEnv,GlobalEnv,Density,MUg0,Mnuv,Mg,Mr,Mz,Re,b/a,n,UV,g-r,RedSequence
#COLNUM=    1,   2,  3, 4,  5, 6,     7,       8,        9,     10,  11,  12,13,14,15,16,17,18,19,20, 21

cat $FILE | tr '\t' ',' | csvcut -c $COLUMNS | tr ',' ' ' > $DATA

clear
echo "-------------------------------------------------------------------------"

################################################################################

# TOTAL COUNT
echo ""
echo "----------------------------------------------------"
echo "Total Counts"
echo "----------------------------------------------------"
TOTAL_CANDIDATES=$(tail -n+2 $DATA | wc -l)
TOTAL_UDGS=$(awk '($3=="TRUE") {print}' $DATA | wc -l)
echo ""
echo "***ALL***:" $TOTAL_UDGS "UDGs out of" $TOTAL_CANDIDATES "Candidates"
echo ""  

################################################################################

# TABLE COUNTS 
echo ""
echo "----------------------------------------------------"
echo "Candidates Per Table"

for TABLE in 2 3 4;
do
    echo "----------------------------------------------------"
    { head -1 $DATA | awk '{print $1, $2, $11, $16, $6}';
      awk -v TABLE=$TABLE '($1==TABLE && $3=="TRUE") {print $1, $2, $11, $16, $6}' \
          $DATA; } | column -t | tee $tmp
    NUM_UDGS=$(tail -n+2 $tmp | wc -l)
    echo "----------------------------------------------------"
    { head -1 $DATA | awk '{print $1, $2, $11, $16, $6}';
      awk -v TABLE=$TABLE '($1==TABLE && $3=="FALSE") {print $1, $2, $11, $16, $6}' \
          $DATA; } | column -t | tee $tmp
    NUM_NONUDGS=$(tail -n+2 $tmp | wc -l)
    echo ""
    echo "***TABLE"$TABLE"***:" $NUM_UDGS "UDGs +" $NUM_NONUDGS "Non-UDGs"
    echo ""  
done

################################################################################

# Borderline Cases
echo ""
echo "----------------------------------------------------"
echo "Borderline UDGs"
echo "----------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6}';
  awk '($3=="TRUE" && $11>=24.0 && $16>=1.5) \
       {print $1, $2, $11, $16, $6}' $DATA; } | column -t | tee $tmp
echo "----------------------------------------------------"
NUM_UDGS=$(tail -n+2 $tmp | wc -l)
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6}';
  awk '($3=="TRUE" && ($11<24.0 || $16<1.5)) \
       {print $1, $2, $11, $16, $6}' $DATA; } | column -t | tee $tmp
NUM_BORDERLINE=$(tail -n+2 $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS "UDGs +" $NUM_BORDERLINE "Borderline UDGs"
echo ""  

################################################################################

# UDG Candidates with < 1000 km/s
echo ""
echo "----------------------------------------------------"
echo "UDG Candidates with cz<1000 km/s"
echo "----------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6}';
  awk '($6<1000) {print $1, $2, $11, $16, $6}' $DATA; } | column -t | tee $tmp
NUM_UDGS=$(tail -n+2 $tmp | awk  '($3>=23.8 && $4>=1.4) {print}' | wc -l)
NUM_CANDIDATES=$(tail -n+2 $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS "UDGs in" $NUM_CANDIDATES "candidates"
echo "                 " $((TOTAL_CANDIDATES-NUM_CANDIDATES)) "viable candidates > 1000 km/s"
echo ""  

################################################################################

# UDG Candidates with 1000<cz<2000 km/s
echo ""
echo "----------------------------------------------------"
echo "UDG Candidates with 1000<cz<2000"
echo "----------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6}';
  awk '($6>1000 && $6<2000) \
       {print $1, $2, $11, $16, $6}' $DATA; } | column -t | tee $tmp
NUM_UDGS=$(tail -n+2 $tmp | awk  '($3>=23.8 && $4>=1.4) {print}' | wc -l)
NUM_CANDIDATES=$(tail -n+2 $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS "UDGs in" $NUM_CANDIDATES "candidates"
echo ""

################################################################################

# UDG Candidates below cz<4000 km/s
echo ""
echo "----------------------------------------------------"
echo "UDGs under 1000<cz<4000 km/s"
echo "----------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6}'
  awk '($6>1000 && $6<4000) \
       {print $1, $2, $11, $16, $6}' $DATA; } | column -t | tee $tmp
NUM_UDGS=$(tail -n+2 $tmp | awk  '($3>=23.8 && $4>1.4) {print}' | wc -l)
NUM_CANDIDATES=$(tail -n+2 $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS "UDGs in" $NUM_CANDIDATES "candidates"
echo ""

################################################################################

# UDG Candidates in Coma-Region
echo ""
echo "------------------------------------------------------------"
echo "Coma Region UDGs/Candidates"
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
  awk '($7 <= 11.0 && $3=="TRUE" && $6>=1000) \
       {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_UDGS_COMA=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
  awk '($7 <= 11.0 && $3=="FALSE" && $6>=1000) \
       {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_NONUDGS_COMA=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
  awk '($7 <= 11.0 && $6<1000) \
       {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_UNCONSTRAINED_COMA=$(tail -n+2 $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS_COMA "UDGs +" \
                         $NUM_NONUDGS_COMA "Non-UDGs +" \
                         $NUM_UNCONSTRAINED_COMA "Unconstrained Objects in Coma"
echo ""

################################################################################

# UDG Candidates in Off Coma-Region
echo ""
echo "------------------------------------------------------------"
echo "OFF-Coma Region UDGs/Candidates"
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
  awk '($7 > 11.0 && $3=="TRUE" && $6>=1000) \
       {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_UDGS=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
  awk '($7 > 11.0 && $3=="FALSE" && $6>=1000) \
       {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_NONUDGS=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
  awk '($7 > 11.0 && $6<1000) \
       {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_UNCONSTRAINED=$(tail -n+2 $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS "UDGs +" \
                         $NUM_NONUDGS "Non-UDGs +" \
                         $NUM_UNCONSTRAINED "Unconstrained Objects in OFF-Coma Region."
echo ""


################################################################################

# Coma-Region UDGs Candidates in Coma Cluster
echo ""
echo "--------------------------------------------------------------------------"
echo "Coma Region UDGs/Candidates & Velocity Breakdown"
echo "--------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $3, $11, $16, $6, $9, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $6>4425 && $6<9425) \
       {print $1, $2, $3, $11, $16, $6, $9, $8}' $DATA; } | column -t | tee $tmp
NUM_UDGS_in2500=$(tail -n+2 $tmp | wc -l)
echo "--------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $3, $11, $16, $6, $9, $8}';
  awk '($7 <= 11.0 && $3=="FALSE" && $6>4425 && $6<9425) \
       {print $1, $2, $3, $11, $16, $6, $9, $8}' $DATA; } | column -t | tee $tmp
NUM_nonUDGS_in2500=$(tail -n+2 $tmp | wc -l)
echo "--------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $3, $11, $16, $6, $9, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $6>1000 && ($6<4425 || $6>9425)) \
       {print $1, $2, $3, $11, $16, $6, $9, $8}' $DATA; } | column -t | tee $tmp
NUM_UDGS_out2500=$(tail -n+2 $tmp | wc -l)
echo "--------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $3, $11, $16, $6, $9, $8}';
  awk '($7 <= 11.0 && $3=="FALSE" && $6>1000 && ($6<4425 || $6>9425)) \
       {print $1, $2, $3, $11, $16, $6, $9, $8}' $DATA; } | column -t | tee $tmp
NUM_nonUDGS_out2500=$(tail -n+2 $tmp | wc -l)
echo "--------------------------------------------------------------------------"
echo ""
echo "**CONCLUSIONS**:" $NUM_UDGS_in2500 "UDGs + " \
                        $NUM_nonUDGS_in2500 "Non-UDGs w/in 2500km/s +"
echo "                " $NUM_UDGS_out2500 " UDGs + " \
                        $NUM_nonUDGS_out2500 "Non-UDGs outside 2500km/s of Coma Mean Velocity (6925km/s)"

################################################################################

# Coma-Region UDGs Candidates in Coma Cluster
echo ""
echo "--------------------------------------------------------------------------"
echo "Coma Region UDGs/Candidates & Environment Breakdown"
echo "--------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $3, $11, $16, $6, $9, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $9=="Cluster") \
       {print $1, $2, $3, $11, $16, $6, $9, $8}' $DATA; } | column -t | tee $tmp
NUM_UDGS_CLUSTER=$(tail -n+2 $tmp | wc -l)
echo "--------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $3, $11, $16, $6, $9, $8}';
  awk '($7 <= 11.0 && $3=="FALSE" && $9=="Cluster") \
       {print $1, $2, $3, $11, $16, $6, $9, $8}' $DATA; } | column -t | tee $tmp
NUM_NONUDGS_CLUSTER=$(tail -n+2 $tmp | wc -l)
echo "--------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $3, $11, $16, $6, $9, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $6>1000) \
       {print $1, $2, $3, $11, $16, $6, $9, $8}' $DATA; } | column -t | tee $tmp
NUM_UDGS_NONCLUSTER=$(tail -n+2 $tmp | wc -l)
echo "--------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $3, $11, $16, $6, $9, $8}';
  awk '($7 <= 11.0 && $3=="FALSE" && $9=="Non-Cluster" && $6>1000) \
       {print $1, $2, $3, $11, $16, $6, $9, $8}' $DATA; } | column -t | tee $tmp
NUM_NONUDGS_NONCLUSTER=$(tail -n+2 $tmp | wc -l)
echo "--------------------------------------------------------------------------"
echo ""
echo "***CONCLUSION***:" $NUM_UDGS_CLUSTER "Cluster UDGs     +" \
                         $NUM_NONUDGS_CLUSTER "Cluster Non-UDGs     +" 
echo "                 " $NUM_UDGS_NONCLUSTER "Non-Cluster UDGs +" \
                         $NUM_NONUDGS_NONCLUSTER "Non-Cluster Non-UDGs +"
echo "                 " $NUM_UNCONSTRAINED_COMA "Unconstrained Objects in Coma Region."
echo ""

################################################################################

# Coma-Region Foreground & Background UDGs Candidates
echo ""
echo "------------------------------------------------------------"
echo "Coma Foreground UDGs/Candidates"
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $6>1000 && $6<4425) \
     {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_UDGS_FOREGROUND=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
awk '($7 <= 11.0 && $3=="FALSE" && $9=="Non-Cluster" && $6>1000 && $6<4425) \
     {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_NONUDGS_FOREGROUND=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
echo ""


echo "------------------------------------------------------------"
echo "Coma Background UDGs/Candidates"
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $6>9425) \
     {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_UDGS_BACKGROUND=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
  awk '($7 <= 11.0 && $3=="FALSE" && $9=="Non-Cluster" && $6>9425) \
       {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_NONUDGS_BACKGROUND=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
echo ""


echo "------------------------------------------------------------"
echo "Coma UDGs/Candidates Outside Splashback Radius"
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
  awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $6>4425 && $6<9425) \
       {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_UDGS_OUTSPLASH=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7}';
  awk '($7 <= 11.0 && $3=="FALSE" && $9=="Non-Cluster" && $6>4425 && $6<9425) \
       {print $1, $2, $11, $16, $6, $7}' $DATA; } | column -t | tee $tmp
NUM_NONUDGS_OUTSPLASH=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------"
echo ""

echo "***CONCLUSION***:" $NUM_UDGS_FOREGROUND "Foreground UDGs +" \
                         $NUM_NONUDGS_FOREGROUND "Foreground Non-UDGs +" 
echo "                 " $NUM_UDGS_BACKGROUND "Background UDGs +" \
                         $NUM_NONUDGS_BACKGROUND "Background Non-UDGs +"
echo "                 " $NUM_UDGS_OUTSPLASH "Outside Splashback UDGs +" \
                         $NUM_NONUDGS_OUTSPLASH "Outside Splashback Non-UDGs +"
echo "                 " $NUM_UNCONSTRAINED_COMA "Unconstrained Foreground Objects in Coma Region."
echo ""


################################################################################

# Coma-Region Foreground & Background UDGs Candidates
echo ""
echo "------------------------------------------------------------------------"
echo "Local Environments of Coma Cluster UDGs"
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7, $8}';
  awk '($3=="TRUE" && $9=="Cluster" && $8=="Sparse") \
       {print $1, $2, $11, $16, $6, $7, $8}' $DATA; } | column -t | tee $tmp
SPARSE_COMA=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7, $8}';
  awk '($3=="TRUE" && $9=="Cluster" && $8=="Dense") \
       {print $1, $2, $11, $16, $6, $7, $8}' $DATA; } | column -t | tee $tmp
DENSE_COMA=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
echo ""

echo "------------------------------------------------------------------------"
echo "Local Environments of Coma Foreground UDGs"
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $8=="Sparse" && $6>1000 && $6<4425) \
       {print $1, $2, $11, $16, $6, $7, $8}' $DATA; } | column -t | tee $tmp
SPARSE_FOREGROUND=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $8=="Dense" && $6>1000 && $6<4425) \
       {print $1, $2, $11, $16, $6, $7, $8}' $DATA; } | column -t | tee $tmp
DENSE_FOREGROUND=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
echo ""


echo "------------------------------------------------------------------------"
echo "Local Environments of Coma Background UDGs"
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $8=="Sparse" && $6>9425) \
       {print $1, $2, $11, $16, $6, $7, $8}' $DATA; } | column -t | tee $tmp
SPARSE_BACKGROUND=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $8=="Dense" && $6>9425) \
       {print $1, $2, $11, $16, $6, $7, $8}' $DATA; } | column -t | tee $tmp
DENSE_BACKGROUND=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
echo ""


echo "------------------------------------------------------------------------"
echo "Local Environments of Coma UDGs Outside Splashback Radius"
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $8=="Sparse" && $6>4425 && $6<9425) \
       {print $1, $2, $11, $16, $6, $7, $8}' $DATA; } | column -t | tee $tmp
SPARSE_OUTSPLASH=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $7, $8}';
  awk '($7 <= 11.0 && $3=="TRUE" && $9=="Non-Cluster" && $8=="Dense" && $6>4425 && $6<9425) \
       {print $1, $2, $11, $16, $6, $7, $8}' $DATA; } | column -t | tee $tmp
DENSE_OUTSPLASH=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
echo ""

echo "***CONCLUSION***:" $SPARSE_COMA "Sparse +" \
                         $DENSE_COMA "Dense Coma UDGs +"
echo "                 " $SPARSE_FOREGROUND "Sparse +" \
                         $DENSE_FOREGROUND "Dense Foreground UDGs +" 
echo "                 " $SPARSE_BACKGROUND "Sparse +" \
                         $DENSE_BACKGROUND "Dense Background UDGs +"
echo "                 " $SPARSE_OUTSPLASH "Sparse +" \
                         $DENSE_OUTSPLASH "Dense UDGs Outside Splashback Radius"
echo ""


################################################################################

#COLUMNS=TABLE,NAME,udg,ra,dec,cz,sepDEG,LocalEnv,GlobalEnv,Density,MUg0,Mnuv,Mg,Mr,Mz,Re,b/a,n,UV
##COLNUM=    1,   2,  3, 4,  5, 6,     7,       8,        9,     10,  11,  12,13,14,15,16,17,18,19


echo "------------------------------------------------------------------------"
echo "NUV"
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $8}';
  awk '($3=="TRUE" && $8=="Sparse" && $19=="Yes") \
       {print $1, $2, $11, $16, $6, $19, $8}' $DATA; } | column -t | tee $tmp
SPARSE_UDGs_NUV=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $8}';
  awk '($3=="TRUE" && $8=="Dense" && $19=="Yes") \
       {print $1, $2, $11, $16, $6, $19, $8}' $DATA; } | column -t | tee $tmp
DENSE_UDGs_NUV=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $9}';
  awk '($3=="TRUE" && $9=="Non-Cluster" && $19=="Yes") \
       {print $1, $2, $11, $16, $6, $19, $9}' $DATA; } | column -t | tee $tmp
NONCLUSTER_UDGs_NUV=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $9}';
  awk '($3=="TRUE" && $9=="Cluster" && $19=="Yes") \
       {print $1, $2, $11, $16, $6, $19, $9}' $DATA; } | column -t | tee $tmp
CLUSTER_UDGs_NUV=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
echo ""

echo "***CONCLUSION***:" $SPARSE_UDGs_NUV "Sparse +" \
                         $DENSE_UDGs_NUV "Dense UDGs  with UV Detections."
echo "                 " $NONCLUSTER_UDGs_NUV "Non-Cluster +" \
                         $CLUSTER_UDGs_NUV "Cluster UDGs with UV Detections."
echo ""


echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $8}';
  awk '($3=="FALSE" && $8=="Sparse" && $19=="Yes") \
       {print $1, $2, $11, $16, $6, $19, $8}' $DATA; } | column -t | tee $tmp
SPARSE_nonUDGs_NUV=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $8}';
  awk '($3=="FALSE" && $8=="Dense" && $19=="Yes") \
       {print $1, $2, $11, $16, $6, $19, $8}' $DATA; } | column -t | tee $tmp
DENSE_nonUDGs_NUV=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $8}';
  awk '($3=="FALSE" && $8=="Unconstrained" && $19=="Yes") \
       {print $1, $2, $11, $16, $6, $19, $8}' $DATA; } | column -t | tee $tmp
UNCONS_nonUDGs_NUV=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $9}';
  awk '($3=="FALSE" && $9=="Non-Cluster" && $19=="Yes") \
       {print $1, $2, $11, $16, $6, $19, $9}' $DATA; } | column -t | tee $tmp
NONCLUSTER_nonUDGs_NUV=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $9}';
  awk '($3=="FALSE" && $9=="Cluster" && $19=="Yes") \
       {print $1, $2, $11, $16, $6, $19, $9}' $DATA; } | column -t | tee $tmp
CLUSTER_nonUDGs_NUV=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
echo ""

echo "***CONCLUSION***:" $SPARSE_nonUDGs_NUV "Sparse +" \
                         $DENSE_nonUDGs_NUV "Dense +" \
                         $UNCONS_nonUDGs_NUV " Non-UDGs with UV Detections."
echo "                 " $NONCLUSTER_nonUDGs_NUV "Non-Cluster +" \
                         $CLUSTER_nonUDGs_NUV "Cluster Non-UDGs with UV Detections."
echo ""


################################################################################

echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $8, $10, $20,$21}';
  awk '($3=="TRUE" && $10=="Low") \
       {print $1, $2, $11, $16, $6, $19, $8, $10, $20, $21}' $DATA; } | column -t | tee $tmp
LOW_DENSITY_UDGs=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
{ head -1 $DATA | awk '{print $1, $2, $11, $16, $6, $19, $8, $10, $20,$21}';
  awk '($3=="TRUE" && $10=="High") \
       {print $1, $2, $11, $16, $6, $19, $8, $10, $20, $21}' $DATA; } | column -t | tee $tmp
HIGH_DENSITY_UDGs=$(tail -n+2 $tmp | wc -l)
echo "------------------------------------------------------------------------"
echo ""

echo "***CONCLUSION***:" $LOW_DENSITY_UDGs "Low Density +" $HIGH_DENSITY_UDGs "High Density UDGs"




################################################################################

rm $tmp #$DATA

echo "-------------------------------------------------------------------------"
