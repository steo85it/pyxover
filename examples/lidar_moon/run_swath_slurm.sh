#!/usr/bin/env bash

#TOL=1e-7 #$1
orb_freq=2H

#mkdir -p stats/eps_$TOL
rm runtmp
touch runtmp

for p in 5 60
do
#	echo "p = $p"

#	OUTDIR="stats/eps_$TOL/ingersoll_p$p"
	echo python ./swath_h2.py ${orb_freq} $p >> runtmp
#	echo ""
done

/home/sberton2/launchLISTslurm runtmp lidmoo 8 99:99:99 40Gb 20 &
wait

rm -f runtmp
