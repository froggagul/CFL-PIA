set root=C:\ProgramData\Anaconda3
call %root%\Scripts\activate.bat %root%

call conda activate fl
call cd E:\
call cd E:\property-inference-pytorch-ifca\backup_221110 - CFL_Real_Final

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 3
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 3

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 4
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 4

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 5
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 5

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 6
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 6

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 7
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 7

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 8
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 8

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 9
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 9

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 10
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 10

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 11
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 11

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 12
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 12

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 13
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 13

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 14
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 14

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 15
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 15

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 16
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 16

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 17
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 17

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 18
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 18

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 19
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 19

call python .\distributed_sgd_passive_IFCA.py -t gender -a race --pi 2 -ni 5000 -c 0 -nc 3 -ds 54321 -ms 12345 -nw 20
call python .\inference_attack_IFCA_old.py -t gender -a race --pi 2 -nw 20

pause