# range 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
for num in {1..9}
do 
    python main.py --rank_rate 0.$num
done