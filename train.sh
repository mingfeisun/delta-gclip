cd vit_cifar
cat tasks.txt | xargs -n 6 -P 6 sh -c 'python3 cifar_vit.py --optim $0 --lr $1 --gamma $2 --delta $3 --weight_decay $4 --device $5'
cd ../bert_finetuning
cat tasks.txt | xargs -n 6 -P 6 sh -c 'python3 bert_main.py --optim $0 --lr $1 --gamma $2 --delta $3 --weight_decay $4 --device $5'