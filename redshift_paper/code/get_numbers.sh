#!/bin/bash

DATA=../data/kadowaki2019.tsv
tmp=tmp.txt

# COLORS



# IMPORTANT COLUMNS
# $5  = Name
# $21 = mu(g,0)
# $46 = M_{NUV}
# $47 = M_g
# $49 = M_r
# $51 = M_z
# $37 = r_e
# $29 = b/a
# $31 = n
# $35 = cz
# $41 = udg
# $42 = LocalEnv
# $43 = GlobalEnv

################################################################################

# Get Numbers for LaTeX Table for Individual Objects from Data File
# awk -F'\t' '{print $5, $21, $46, $47, $49, $51, $37, $29, $31, $35, $42, $43}' $DATA | grep SMDG1313189
clear
echo "-------------------------------------------------------------------------"

################################################################################

# TOTAL COUNT
echo ""
echo "----------------------------------------------------"
echo "Total Counts"
echo "----------------------------------------------------"
TOTAL_CANDIDATES=$(tail -n+1 $DATA | wc -l)
TOTAL_UDGS=$(awk -F'\t' '($41=="TRUE") {print}' $DATA | wc -l)
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
    head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35}'
    awk -v TABLE=$TABLE -v OFS='\t' -F'\t' '($1==TABLE && $41=="TRUE") \
           {print $1, $5, $21, $37, $35}' $DATA | tee $tmp
    NUM_UDGS=$(cat $tmp | wc -l)
    echo "----------------------------------------------------"
    awk -v TABLE=$TABLE -v OFS='\t' -F'\t' '($1==TABLE && $41=="FALSE") \
           {print $1, $5, $21, $37, $35}' $DATA | tee $tmp
    NUM_NONUDGS=$(cat $tmp | wc -l)
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
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35}'
awk -v OFS='\t' -F'\t' '($41=="TRUE" && $21>=24.0 && $37>=1.5) \
                        {print $1, $5, $21, $37, $35}' $DATA | tee $tmp
echo "----------------------------------------------------"
NUM_UDGS=$(cat $tmp | wc -l)
awk -v OFS='\t' -F'\t' '($41=="TRUE" && ($21<24.0 || $37<1.5)) \
                        {print $1, $5, $21, $37, $35}' $DATA | tee $tmp
NUM_BORDERLINE=$(cat $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS "UDGs +" $NUM_BORDERLINE "Borderline UDGs"
echo ""  

################################################################################

# UDG Candidates with < 1000 km/s
echo ""
echo "----------------------------------------------------"
echo "UDG Candidates with cz<1000 km/s"
echo "----------------------------------------------------"
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35}'
awk -v OFS='\t' -F'\t' '($35<1000) \
                        {print $1, $5, $21, $37, $35}' $DATA | tee $tmp
NUM_UDGS=$(cat $tmp | awk -F'\t' '($3>=23.8 && $4>=1.4) {print}' | wc -l)
NUM_CANDIDATES=$(cat $tmp | wc -l)  
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
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35}'
awk -v OFS='\t' -F'\t' '($35>1000 && $35<2000) \
                        {print $1, $5, $21, $37, $35}' $DATA | tee $tmp
NUM_UDGS=$(cat $tmp | awk -F'\t' '($3>=23.8 && $4>=1.4) {print}' | wc -l) 
NUM_CANDIDATES=$(cat $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS "UDGs in" $NUM_CANDIDATES "candidates"
echo ""

################################################################################

# UDG Candidates below cz<4000 km/s
echo ""
echo "----------------------------------------------------"
echo "UDGs under 1000<cz<4000 km/s"
echo "----------------------------------------------------"
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35}'
awk -v OFS='\t' -F'\t' '($35>1000 && $35<4000) \
                        {print $1, $5, $21, $37, $35}' $DATA | tee $tmp
NUM_UDGS=$(cat $tmp | awk -F'\t' '($3>=23.8 && $4>1.4) {print}' | wc -l) 
NUM_CANDIDATES=$(cat $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS "UDGs in" $NUM_CANDIDATES "candidates"
echo ""

################################################################################

# UDG Candidates in Coma-Region
echo ""
echo "------------------------------------------------------------"
echo "Coma Region UDGs/Candidates"
echo "------------------------------------------------------------"
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39}'
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $35>=1000) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_UDGS_COMA=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="FALSE" && $35>=1000) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_NONUDGS_COMA=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $35<1000) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_UNCONSTRAINED_COMA=$(cat $tmp | wc -l)
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
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39}'
awk -v OFS='\t' -F'\t' '($39 > 11.0 && $41=="TRUE" && $35>=1000) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_UDGS=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 > 11.0 && $41=="FALSE" && $35>=1000) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_NONUDGS=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 > 11.0 && $35<1000) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_UNCONSTRAINED=$(cat $tmp | wc -l)
echo ""
echo "***CONCLUSION***:" $NUM_UDGS "UDGs +" \
                         $NUM_NONUDGS "Non-UDGs +" \
                         $NUM_UNCONSTRAINED "Unconstrained Objects in OFF-Coma Region."
echo ""


################################################################################

# Coma-Region UDGs Candidates in Coma Cluster
echo ""
echo "------------------------------------------------------------"
echo "Coma Region UDGs/Candidates"
echo "------------------------------------------------------------"
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39}'
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Cluster") \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_UDGS_CLUSTER=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="FALSE" && $43=="Cluster") \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_NONUDGS_CLUSTER=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" && $35>1000) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_UDGS_NONCLUSTER=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="FALSE" && $43=="Non-Cluster" && $35>1000) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_NONUDGS_NONCLUSTER=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
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
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39}'
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" && $35>1000 && $35<4425) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_UDGS_FOREGROUND=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="FALSE" && $43=="Non-Cluster" && $35>1000 && $35<4425) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_NONUDGS_FOREGROUND=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
echo ""


echo "------------------------------------------------------------"
echo "Coma Background UDGs/Candidates"
echo "------------------------------------------------------------"
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39}'
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" && $35>9425) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_UDGS_BACKGROUND=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="FALSE" && $43=="Non-Cluster" && $35>9425) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_NONUDGS_BACKGROUND=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
echo ""


echo "------------------------------------------------------------"
echo "Coma UDGs/Candidates Outside Splashback Radius"
echo "------------------------------------------------------------"
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39}'
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" && $35>4425 && $35<9425) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_UDGS_OUTSPLASH=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="FALSE" && $43=="Non-Cluster" && $35>4425 && $35<9425) \
                        {print $1, $5, $21, $37, $35, $39}' $DATA | tee $tmp
NUM_NONUDGS_OUTSPLASH=$(cat $tmp | wc -l) 
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
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39, $42}'
awk -v OFS='\t' -F'\t' '($41=="TRUE" && $43=="Cluster" && $42=="Sparse") \
                        {print $1, $5, $21, $37, $35, $39, $42}' $DATA | tee $tmp
SPARSE_COMA=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($41=="TRUE" && $43=="Cluster" && $42=="Dense") \
                        {print $1, $5, $21, $37, $35, $39, $42}' $DATA | tee $tmp
DENSE_COMA=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------------------"
echo ""

echo "------------------------------------------------------------------------"
echo "Local Environments of Coma Foreground UDGs"
echo "------------------------------------------------------------------------"
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39, $42}'
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" \
                        && $42=="Sparse" && $35>1000 && $35<4425) \
                        {print $1, $5, $21, $37, $35, $39, $42}' $DATA | tee $tmp
SPARSE_FOREGROUND=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" \
                        && $42=="Dense" && $35>1000 && $35<4425) \
                        {print $1, $5, $21, $37, $35, $39, $42}' $DATA | tee $tmp
DENSE_FOREGROUND=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------------------"
echo ""


echo "------------------------------------------------------------------------"
echo "Local Environments of Coma Background UDGs"
echo "------------------------------------------------------------------------"
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39, $42}'
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" \
                        && $42=="Sparse" && $35>9425) \
                        {print $1, $5, $21, $37, $35, $39, $42}' $DATA | tee $tmp
SPARSE_BACKGROUND=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" \
                        && $42=="Dense" && $35>9425) \
                        {print $1, $5, $21, $37, $35, $39, $42}' $DATA | tee $tmp
DENSE_BACKGROUND=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------------------"
echo ""


echo "------------------------------------------------------------------------"
echo "Local Environments of Coma UDGs Outside Splashback Radius"
echo "------------------------------------------------------------------------"
head -1 $DATA | awk -v OFS='\t' -F"\t" '{print $1, $5, FS, $21, $37, $35, $39, $42}'
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" \
                        && $42=="Sparse" && $35>4425 && $35<9425) \
                        {print $1, $5, $21, $37, $35, $39, $42}' $DATA | tee $tmp
SPARSE_OUTSPLASH=$(cat $tmp | wc -l) 
echo "------------------------------------------------------------------------"
awk -v OFS='\t' -F'\t' '($39 <= 11.0 && $41=="TRUE" && $43=="Non-Cluster" \
                        && $42=="Dense" && $35>4425 && $35<9425) \
                        {print $1, $5, $21, $37, $35, $39, $42}' $DATA | tee $tmp
DENSE_OUTSPLASH=$(cat $tmp | wc -l) 
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



rm $tmp

echo "-------------------------------------------------------------------------"
