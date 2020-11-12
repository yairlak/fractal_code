#UNIT=0 #23 if zero, then pertubation is based on pca
RELATIVE_PERT='--relative-pert'
#PC='--pc 2'
#RELATIVE_PERT=''

#for CURRENT in -10 -6 -5 -4 0; do
for PC in 2 1 3; do
	for CURRENT in -10 -8 -6 -4 -2 0 2 4 6 8 10; do
		for INJECT_TIME in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
			python -W ignore inject_activity_and_eval.py --unit 0 --current $CURRENT --inject-time $INJECT_TIME $RELATIVE_PERT --pc $PC
		done
	done
done
