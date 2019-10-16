PLOT_DIR=../plots
SUB_DIR=$PLOT_DIR/animated
PREFIX=udgs_global
#PREFIX=candidates_global
EXT=.pdf

mkdir -p $SUB_DIR

for GLOBAL in `ls $PLOT_DIR/$PREFIX*$EXT`;
do
    LOCAL=${GLOBAL//global/local};
    TMP=${GLOBAL//_global/};
    BASE=`basename $TMP $EXT`
    
    convert -delay 100 -loop 0 $GLOBAL $LOCAL $SUB_DIR/$BASE.gif
    
done


