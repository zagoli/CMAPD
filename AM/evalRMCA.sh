#!/bin/bash

# For loop da 0 a 99 sulla variabile i
for i in {0..99}
do
	# Creazione della stringa di comando
	map="instances/maps/$i.map"
	agents="instances/maps/$i.map"
	tasks="instances/tasks/$i.task"
	cmdRMCA="../MCA-RMCA/build/MAPD -m $map -a $agents -t $tasks -c 60 -s PP --capacity 3 --objective total-travel-delay --only-update-top --kiva --regret"
	cmdModel="python eval.py -m $map -t $tasks -s models/transformer_conflicts_small.pth"
	# Esecuzione del comando
	# echo "INSTANCE $i --------------------------------------------------"
	# $cmdRMCA
	$cmdModel
done


