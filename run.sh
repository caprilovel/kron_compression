for num in {1..9}
do 
    python main.py --rank_rate 0.$num --device cpu
done

for num in {1..4}
do 
    python main.py --rank_rate $num --device cpu
done

python main.py --model LeNet_5 --device cpu