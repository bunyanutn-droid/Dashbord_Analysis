#!/bin/bash
y=$1; c=$2
out=/c/Users/Acer/Downloads/Dashbord_Analysis/data/raw/${y}_${c}.txt
[ -s "$out" ] && exit 0
curl -sk -m 40 --retry 2 -H "User-Agent: Mozilla/5.0" "https://stat.bora.dopa.go.th/new_stat/file/${y}12/${y}12cc${c}.txt" | head -n 1 > "$out"
