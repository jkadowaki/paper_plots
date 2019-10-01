GIF=../plots/udgs_pair.gif
GLOBAL=../plots/udgs_global_pair.pdf
LOCAL=../plots/udgs_local_pair.pdf

convert -delay 200 -loop 0 $GLOBAL $LOCAL $GIF
