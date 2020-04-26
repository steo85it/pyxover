#!/bin/sh
mapproject ladata_concat_AGTP_0.txt -Js0/90/1:100000 -R0/360/60/90 -S -C -bo | blockmedian -bi3 -C -r -I1 -R-1500/1500/-1500/1500 -bo3 | surface -bi3 -T0.25 -r -I1 -R-1500/1500/-1500/1500 -Gtopo_AGTP_0.grd

