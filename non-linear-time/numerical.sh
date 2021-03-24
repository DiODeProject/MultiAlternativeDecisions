#!/bin/bash
#$ -V
#$ -P rse
# -q rse.q

qflags="-cwd"

gamm=0.8

for utility in linear
do
  for geometric in 0
  do
    for maxval in $(seq 3.0 0.5 5.0)
    do
      for logslope in $(seq 0.25 0.25 1.5) # linear discounting range
      #for logslope in $(seq 1.5 1 6.5) # geometric discounting range
      do 
        qsub $qflags -v utility=$utility,geometric=$geometric,gamm=$gamm,maxval=$maxval,logslope=$logslope numerical.qsub
      done
    done
  done
done
