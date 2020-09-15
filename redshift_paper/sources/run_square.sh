#!/bin/bash

function square_random {
    # Generates a random 16-bit integer
    rand=$RANDOM
    square=$((rand*rand))

    # Uncomment this line to print result
    #echo $square

    #sleep 1
}

export -f square_random
bash -c "time for i in {1..100000}; do square_random; done" 2>&1 | grep real

